"""
vln_cache_wrapper.py  –  Hook-based KV-Cache reuse manager for InternVLA-N1.

Provides ``VLNCacheManager`` which attaches forward hooks to the Qwen2.5-VL
decoder layers inside InternVLA-N1 and transparently splices previous-frame KV
entries for reusable vision tokens.  No weight modifications; no retraining.

Integration point (minimal invasion):
    The evaluator / agent instantiates ``VLNCacheManager(model, cfg)`` once
    and calls ``manager.on_new_frame(depth, pos, quat, patch_embeds)`` before
    each ``model.generate()``.  When the episode resets, call ``manager.reset()``.

Transformer layer path:  model.model.layers[ℓ]  (Qwen2_5_VLDecoderLayer)
    Each layer's self_attn produces  (hidden, attn_weights, present_key_value).
    ``present_key_value`` is a DynamicCache entry  (K, V) written *after*
    the attention op.  We intercept the layer output to overwrite the KV
    entries that correspond to reusable vision tokens.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F

from .vln_cache_utils import (
    build_reuse_mask,
    compute_semantic_refresh_gate,
    layer_adaptive_threshold,
    neighborhood_refinement,
    view_aligned_remap,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────

@dataclass
class VLNCacheConfig:
    """Hyper-parameters for VLN-Cache.  All have sensible defaults."""

    # View-Aligned Remap
    occlusion_eps: float = 0.15       # relative depth tolerance (fraction of depth)
    neighbor_k: int = 2               # half-width of refinement window (2 → 5×5)

    # Visual static gate
    tau_v_base: float = 0.0           # base cosine threshold (0 = trust
                                      # geometric matching for static scenes)
    tau_v_range: float = 0.15         # adaptation half-width

    # Semantic refresh gate
    tau_s: float = 0.5                # absolute relevance threshold
    delta_s: float = 0.3              # temporal change threshold

    # Layer-adaptive entropy schedule
    H_low: float = 1.0
    H_high: float = 4.0

    # Camera intrinsics (will be overridden by agent config at runtime).
    fx: float = 0.0
    fy: float = 0.0
    cx: float = 0.0
    cy: float = 0.0
    img_h: int = 480
    img_w: int = 480

    # Patch grid (Qwen2.5-VL default for 480×480 with 14×14 patch_size, 2× merge).
    H_patches: int = 17
    W_patches: int = 17

    # Whether to enable layer-adaptive thresholding (requires attention weights).
    layer_adaptive: bool = False

    # Maximum reuse ratio safety cap.
    max_reuse_ratio: float = 0.8


# ──────────────────────────────────────────────────────────────────────
# Frame-level state cache
# ──────────────────────────────────────────────────────────────────────

@dataclass
class _FrameState:
    """Mutable record for a single frame's data."""
    depth: Optional[torch.Tensor] = None          # [H, W]
    pos: Optional[torch.Tensor] = None            # [3]
    quat: Optional[torch.Tensor] = None           # [4]  (w,x,y,z)
    patch_embeds: Optional[torch.Tensor] = None   # [M, d]
    cross_attn_scores: Optional[torch.Tensor] = None  # [M]
    kv_per_layer: dict = field(default_factory=dict)   # {layer_idx: (K, V)}


# ──────────────────────────────────────────────────────────────────────
# Main manager
# ──────────────────────────────────────────────────────────────────────

class VLNCacheManager:
    """Training-free KV-Cache reuse controller for InternVLA-N1.

    Lifecycle:
        1.  ``__init__(model, cfg)`` – registers hooks (idempotent).
        2.  ``on_new_frame(...)``   – call **before** ``model.generate()``
            each step.  Computes the reuse mask and stores frame state.
        3.  ``reset()``             – call on episode boundary.
        4.  ``remove_hooks()``      – detach when no longer needed.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        cfg: VLNCacheConfig | None = None,
        vision_token_offset: int | None = None,
        n_vision_tokens: int | None = None,
    ):
        self.cfg = cfg or VLNCacheConfig()
        self.model = model

        # Decode layer list – model.model.layers for InternVLAN1ForCausalLM.
        self._layers = self._resolve_layers(model)
        self.n_layers = len(self._layers)

        # Per-episode temporal state.
        self._prev: Optional[_FrameState] = None
        self._curr: Optional[_FrameState] = None

        # Reuse mask: set before each forward; consumed by hooks.
        self._reuse_mask: Optional[torch.Tensor] = None        # [M] bool
        self._refined_idx: Optional[torch.Tensor] = None       # [M] int64
        self._vision_token_offset: Optional[int] = vision_token_offset
        self._n_vision_tokens: Optional[int] = n_vision_tokens

        # Hook handles for cleanup.
        self._hook_handles: list = []

        # Statistics.
        self.stats = {"n_frames": 0, "mean_reuse_ratio": 0.0}

    # ------------------------------------------------------------------ hooks

    @staticmethod
    def _resolve_layers(model):
        """Navigate the HuggingFace model wrapper to find decoder layers."""
        # InternVLAN1ForCausalLM → .model (InternVLAN1Model) → .layers
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return model.model.layers
        if hasattr(model, "layers"):
            return model.layers
        raise AttributeError(
            "Cannot locate decoder layers. Expected model.model.layers "
            "on InternVLAN1ForCausalLM / Qwen2_5_VLForConditionalGeneration."
        )

    def _register_hooks(self):
        """Attach post-forward hooks to each decoder layer."""
        if self._hook_handles:
            return  # already registered
        for layer_idx, layer in enumerate(self._layers):
            handle = layer.register_forward_hook(
                self._make_layer_hook(layer_idx),
                with_kwargs=True,
            )
            self._hook_handles.append(handle)
        logger.info("VLN-Cache: registered %d layer hooks.", len(self._hook_handles))

    def remove_hooks(self):
        """Detach all hooks (restore vanilla behaviour)."""
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()

    # ------------------------------------------------------------------ core API

    def reset(self):
        """Reset temporal state at episode boundary."""
        self._prev = None
        self._curr = None
        self._reuse_mask = None
        self._refined_idx = None
        self.stats = {"n_frames": 0, "mean_reuse_ratio": 0.0}

    def on_new_frame(
        self,
        depth: torch.Tensor,
        pos: torch.Tensor,
        quat: torch.Tensor,
        patch_embeds: torch.Tensor,
        cross_attn_scores: torch.Tensor | None = None,
        vision_token_offset: int | None = None,
        n_vision_tokens: int | None = None,
    ):
        """Prepare reuse mask for the upcoming forward pass.

        Must be called BEFORE ``model.generate()`` / ``model.forward()``. 

        Args:
            depth:        [H, W]  current depth map (metres).
            pos:          [3]     world-frame agent position.
            quat:         [4]     world-frame quaternion (w,x,y,z).
            patch_embeds: [M, d]  current vision patch embeddings
                                  (output of ViT / visual encoder).
            cross_attn_scores: [M] optional relevance proxy.
            vision_token_offset:  index of the first vision token in the
                                  full input_ids / inputs_embeds sequence.
            n_vision_tokens: number of vision tokens (= M).
        """
        if vision_token_offset is not None:
            self._vision_token_offset = vision_token_offset
        if n_vision_tokens is not None:
            self._n_vision_tokens = n_vision_tokens

        cfg = self.cfg
        self._curr = _FrameState(
            depth=depth.detach(),
            pos=pos.detach(),
            quat=quat.detach(),
            patch_embeds=patch_embeds.detach(),
            cross_attn_scores=cross_attn_scores.detach() if cross_attn_scores is not None else None,
        )

        if self._prev is None or self._prev.patch_embeds is None:
            # First frame: no reuse possible.
            self._reuse_mask = None
            self._refined_idx = None
            # Do NOT commit here — hooks must store KV in _curr first.
            return

        M = patch_embeds.shape[0]
        M_prev = self._prev.patch_embeds.shape[0]

        # ---- Dimension compatibility check ----
        # If the patch grid changed (e.g. lookdown image has different
        # resolution), reuse is unsafe → skip this frame entirely.
        # Setting _curr = None ensures the decoder-layer hooks also skip
        # storing KV for this frame, so _prev retains the last
        # resolution-compatible frame for future comparisons.
        if M_prev != M:
            logger.warning(
                "VLN-Cache: patch count mismatch (M_prev=%d, M_curr=%d). "
                "Skipping reuse AND storage for this frame.",
                M_prev, M,
            )
            self._reuse_mask = None
            self._refined_idx = None
            self._curr = None          # hooks will skip (no KV store / commit)
            return

        # §1 View-Aligned Remap.
        matched_idx, valid_geom = view_aligned_remap(
            depth_curr=self._curr.depth,
            depth_prev=self._prev.depth,
            pos_curr=self._curr.pos,
            quat_curr=self._curr.quat,
            pos_prev=self._prev.pos,
            quat_prev=self._prev.quat,
            fx=cfg.fx, fy=cfg.fy, cx=cfg.cx, cy=cfg.cy,
            H_patches=cfg.H_patches, W_patches=cfg.W_patches,
            img_h=cfg.img_h, img_w=cfg.img_w,
            occlusion_eps=cfg.occlusion_eps,
        )

        # Diagnostic logging: breakdown at each stage.
        n_geom_valid = int(valid_geom.sum().item())
        _pos_delta = float((self._curr.pos - self._prev.pos).norm().item())
        logger.debug(
            "VLN-Cache diag: M=%d  geom_valid=%d (%.1f%%)  pos_delta=%.3f",
            M, n_geom_valid, 100.0 * n_geom_valid / max(M, 1), _pos_delta,
        )

        # §1b Neighbourhood refinement.
        refined_idx, m_vis = neighborhood_refinement(
            patch_curr=self._curr.patch_embeds,
            patch_prev=self._prev.patch_embeds,
            matched_idx=matched_idx,
            valid_mask=valid_geom,
            W_patches=cfg.W_patches,
            H_patches=cfg.H_patches,
            k=cfg.neighbor_k,
            tau_v=cfg.tau_v_base,
        )

        n_vis_pass = int(m_vis.sum().item())
        logger.debug(
            "VLN-Cache diag: vis_pass=%d (%.1f%%)  tau_v=%.2f",
            n_vis_pass, 100.0 * n_vis_pass / max(M, 1), cfg.tau_v_base,
        )

        # §2 Semantic Refresh Gate.
        if cross_attn_scores is not None:
            m_sem = compute_semantic_refresh_gate(
                cross_attn_scores,
                self._prev.cross_attn_scores,
                tau_s=cfg.tau_s,
                delta_s=cfg.delta_s,
            )
        else:
            # No relevance signal → disable semantic gate (no forced refresh).
            m_sem = torch.zeros(M, dtype=torch.bool, device=patch_embeds.device)

        # §4 Final mask.
        reuse = build_reuse_mask(m_vis, m_sem)

        # Safety cap: never reuse more than max_reuse_ratio of tokens.
        reuse_ratio = reuse.float().mean().item()
        if reuse_ratio > cfg.max_reuse_ratio:
            # Drop excess reuse tokens randomly.
            n_reuse = reuse.sum().item()
            n_target = int(cfg.max_reuse_ratio * M)
            excess = int(n_reuse - n_target)
            if excess > 0:
                reuse_indices = reuse.nonzero(as_tuple=False).squeeze(-1)
                drop = reuse_indices[torch.randperm(len(reuse_indices))[:excess]]
                reuse[drop] = False
            reuse_ratio = reuse.float().mean().item()

        self._reuse_mask = reuse
        self._refined_idx = refined_idx

        # Update running stats.
        n = self.stats["n_frames"]
        self.stats["mean_reuse_ratio"] = (
            self.stats["mean_reuse_ratio"] * n + reuse_ratio
        ) / (n + 1)
        self.stats["n_frames"] = n + 1

        logger.debug(
            "VLN-Cache frame %d: reuse %.1f%% (%d / %d tokens)",
            self.stats["n_frames"], reuse_ratio * 100,
            reuse.sum().item(), M,
        )

        # Do NOT commit here — the decoder layer hooks must store KV
        # in _curr first.  Commit happens in the last layer's hook.

    def _commit_frame(self):
        """Shift current frame to previous for next iteration."""
        self._prev = self._curr
        self._curr = None

    # ------------------------------------------------------------------ hook factory

    def _make_layer_hook(self, layer_idx: int):
        """Return a post-forward hook for decoder layer ``layer_idx``.

        The hook intercepts the layer output and:
          1. Stores the current frame's vision-token KV for future reuse.
          2. For reusable vision tokens, overwrites the newly computed KV
             with cached values from the previous frame.
          3. On the LAST layer: commits the frame (_curr → _prev) and
             clears the reuse mask so autoregressive steps are unaffected.

        Registered with ``with_kwargs=True`` so the signature is
        ``hook(module, args, kwargs, output)`` and we can access
        ``past_key_value`` directly from the decoder layer's kwargs.
        """
        manager = self  # closure reference
        is_last_layer = (layer_idx == self.n_layers - 1)

        def hook(module, args, kwargs, output):
            # Only act when we have an active current frame (prefill step).
            if manager._curr is None:
                return output  # autoregressive step or not initialised

            offset = manager._vision_token_offset
            n_vis = manager._n_vision_tokens
            if offset is None or n_vis is None:
                if is_last_layer:
                    manager._commit_frame()
                    manager._reuse_mask = None
                    manager._refined_idx = None
                return output

            # ---- Access the KV cache from this layer's kwargs ----
            # The DynamicCache is passed as ``past_key_value`` kwarg to
            # each Qwen2_5_VLDecoderLayer.  By the time the post-hook
            # fires, self_attn has already called cache.update() so
            # cache.key_cache[layer_idx] exists.
            cache = kwargs.get("past_key_value")
            if cache is None:
                if is_last_layer:
                    manager._commit_frame()
                    manager._reuse_mask = None
                    manager._refined_idx = None
                return output

            try:
                K_curr = cache.key_cache[layer_idx]    # [B, heads, S, d_head]
                V_curr = cache.value_cache[layer_idx]  # [B, heads, S, d_head]
            except (IndexError, AttributeError):
                if is_last_layer:
                    manager._commit_frame()
                    manager._reuse_mask = None
                    manager._refined_idx = None
                return output

            seq_len = K_curr.shape[2]
            if seq_len <= offset + n_vis - 1:
                # Sequence too short — probably an unexpected state.
                if is_last_layer:
                    manager._commit_frame()
                    manager._reuse_mask = None
                    manager._refined_idx = None
                return output

            # ---- Step 1: Store current KV in _curr for future reuse ----
            manager._curr.kv_per_layer[layer_idx] = (
                K_curr[:, :, offset:offset + n_vis, :].detach().clone(),
                V_curr[:, :, offset:offset + n_vis, :].detach().clone(),
            )

            # ---- Step 2: Splice previous KV for reusable tokens ----
            reuse = manager._reuse_mask
            if reuse is not None and manager._prev is not None:
                refined_idx = manager._refined_idx
                prev_kv = manager._prev.kv_per_layer.get(layer_idx)

                if prev_kv is not None and refined_idx is not None:
                    K_prev, V_prev = prev_kv
                    M_prev = K_prev.shape[2]

                    reuse_idx = reuse.nonzero(as_tuple=False).squeeze(-1)
                    if reuse_idx.numel() > 0:
                        src_idx = refined_idx[reuse_idx].long()

                        # Bounds check.
                        valid = (src_idx >= 0) & (src_idx < M_prev)
                        reuse_idx = reuse_idx[valid]
                        src_idx = src_idx[valid]

                        if reuse_idx.numel() > 0:
                            dst_pos = offset + reuse_idx

                            # In-place splice.
                            K_curr[:, :, dst_pos, :] = K_prev[:, :, src_idx, :]
                            V_curr[:, :, dst_pos, :] = V_prev[:, :, src_idx, :]

                            # Update stored KV with spliced values.
                            manager._curr.kv_per_layer[layer_idx] = (
                                K_curr[:, :, offset:offset + n_vis, :].detach().clone(),
                                V_curr[:, :, offset:offset + n_vis, :].detach().clone(),
                            )

            # ---- Step 3: Commit on last layer ----
            if is_last_layer:
                manager._commit_frame()
                manager._reuse_mask = None
                manager._refined_idx = None

            return output

        return hook

    # ------------------------------------------------------------------ convenience

    def compute_cross_attn_proxy(
        self,
        hidden_states: torch.Tensor,
        vision_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute a lightweight relevance proxy for semantic gating.

        Uses the L2 norm of hidden states at vision-token positions after the
        last decoder layer as a surrogate for cross-attention contribution.

        Args:
            hidden_states: [B, S, d]  output of the last decoder layer.
            vision_mask:   [S] bool – True at vision-token positions.

        Returns:
            scores: [M]  relevance scores (higher = more relevant).
        """
        # Take first batch element.
        h = hidden_states[0, vision_mask, :]  # [M, d]
        scores = h.norm(dim=-1)  # [M]
        # Normalize to [0, 1].
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        return scores

    @staticmethod
    def configure_from_agent(
        agent_config,
        H_patches: int | None = None,
        W_patches: int | None = None,
    ) -> VLNCacheConfig:
        """Build VLNCacheConfig from an agent's model_settings.

        Reads width, height, hfov from the agent config to derive intrinsics.
        """
        import numpy as np

        w = getattr(agent_config, "width", 480)
        h = getattr(agent_config, "height", 480)
        hfov = getattr(agent_config, "hfov", 90)

        fx = (w / 2.0) / np.tan(np.deg2rad(hfov / 2.0))
        fy = fx
        cx = (w - 1.0) / 2.0
        cy = (h - 1.0) / 2.0

        cfg = VLNCacheConfig(
            fx=fx, fy=fy, cx=cx, cy=cy,
            img_h=h, img_w=w,
        )
        if H_patches is not None:
            cfg.H_patches = H_patches
        if W_patches is not None:
            cfg.W_patches = W_patches
        return cfg
