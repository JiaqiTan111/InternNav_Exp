"""
evaluator_integration.py  –  Bridge between HabitatVLNEvaluator and VLN-Cache.

Provides ``VLNCacheHook``, a lightweight orchestrator that:
    1. hooks the visual encoder to capture patch embeddings,
    2. hooks the inner decoder model to capture the live DynamicCache,
    3. delegates to ``VLNCacheManager`` for geometric matching, gating,
       and KV-splice via per-layer hooks.

Usage (inside habitat_vln_evaluator.py – 4 call sites):
    ┌─ __init__: cache_hook = VLNCacheHook.from_evaluator(self)
    ├─ episode start:  cache_hook.reset_episode()
    ├─ before generate: cache_hook.set_frame(depth_m, pos, quat, inputs)
    └─ episode end:    stats = cache_hook.get_episode_stats()

All heavy logic lives in vln_cache_wrapper.py / vln_cache_utils.py;
this file only handles the evaluator ↔ model glue.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np
import torch

from .vln_cache_wrapper import VLNCacheConfig, VLNCacheManager

logger = logging.getLogger(__name__)

IMAGE_TOKEN_ID = 151655  # Qwen2.5-VL <image> placeholder id


class VLNCacheHook:
    """Orchestrates VLN-Cache integration with an InternVLA-N1 evaluator.

    Lifecycle:
        1. ``VLNCacheHook.from_evaluator(evaluator)``   – factory, call once.
        2. ``hook.reset_episode()``                      – per-episode reset.
        3. ``hook.set_frame(depth_m, pos, quat, inputs)``– before ``generate()``.
        4. ``hook.get_episode_stats()``                  – after episode ends.
        5. ``hook.remove_all_hooks()``                   – cleanup.
    """

    # -------------------------------------------------------------- factory

    @classmethod
    def from_evaluator(cls, evaluator, cache_cfg: VLNCacheConfig | None = None):
        """Construct from a live ``HabitatVLNEvaluator`` instance.

        Reads camera intrinsics and sensor resolution from the evaluator's
        Habitat config, then instantiates the hook + manager.
        """
        sensor_cfg = evaluator.sim_sensors_config
        img_w = sensor_cfg.depth_sensor.width
        img_h = sensor_cfg.depth_sensor.height

        if cache_cfg is None:
            cache_cfg = VLNCacheConfig(
                fx=float(evaluator._fx),
                fy=float(evaluator._fy),
                cx=(img_w - 1) / 2.0,
                cy=(img_h - 1) / 2.0,
                img_h=img_h,
                img_w=img_w,
            )

        return cls(evaluator.model, cache_cfg, evaluator.device)

    # -------------------------------------------------------------- init

    def __init__(
        self,
        model: torch.nn.Module,
        cfg: VLNCacheConfig,
        device: torch.device,
    ):
        self.model = model
        self.device = device
        self.cfg = cfg

        # Qwen2.5-VL spatial merge size  (ViT patches → LLM tokens).
        # model.visual merges merge_size×merge_size raw patches into one
        # token, so the number of IMAGE_TOKEN_IDs in input_ids is
        # prod(image_grid_thw) // merge_size**2.
        self._spatial_merge_size: int = getattr(
            getattr(getattr(model, "config", None), "vision_config", None),
            "spatial_merge_size", 2,
        )

        # Core algorithm manager (handles per-layer hooks + splice logic).
        self.manager = VLNCacheManager(model, cfg)
        self.manager._register_hooks()

        # --- Additional orchestration hooks ---
        self._hook_handles: list = []

        # 1. Visual encoder post-hook: capture patch embeddings.
        self._captured_embeds: Optional[torch.Tensor] = None
        h = model.visual.register_forward_hook(self._on_visual_output)
        self._hook_handles.append(h)

        # NOTE: KV cache access is handled directly by the layer hooks via
        # with_kwargs=True — they read ``past_key_value`` from each decoder
        # layer's kwargs.  No inner-model pre-hook is needed.

        # Pending frame data (set by set_frame, consumed by _trigger).
        self._pending_depth: Optional[torch.Tensor] = None
        self._pending_pos: Optional[torch.Tensor] = None
        self._pending_quat: Optional[torch.Tensor] = None
        self._pending_offset: Optional[int] = None
        self._pending_n_patches: Optional[int] = None

        # Per-step timing.
        self._step_timings: list[dict] = []

        # Whether the current forward is a prefill (has vision tokens).
        self._is_prefill = False

        logger.info(
            "VLNCacheHook initialised: img=%dx%d  fx=%.1f  fy=%.1f  "
            "spatial_merge_size=%d",
            cfg.img_w, cfg.img_h, cfg.fx, cfg.fy,
            self._spatial_merge_size,
        )
        if self._spatial_merge_size not in (1, 2):
            logger.warning(
                "Unexpected spatial_merge_size=%d — expected 1 or 2. "
                "Patch count calculations may be wrong.",
                self._spatial_merge_size,
            )

    # -------------------------------------------------------------- public API

    def reset_episode(self):
        """Call at the start of every episode."""
        self.manager.reset()
        self._captured_embeds = None
        self._pending_depth = None
        self._step_timings = []

    def set_frame(
        self,
        depth_m: torch.Tensor | np.ndarray,
        pos: torch.Tensor | np.ndarray | list,
        quat_wxyz: torch.Tensor | np.ndarray | list,
        inputs,  # processor output BatchEncoding
    ):
        """Store pending frame data; consumed during the upcoming ``generate()``.

        Args:
            depth_m:    [H, W]  depth in **metres**.
            pos:        [3]     world-frame position.
            quat_wxyz:  [4]     quaternion (w, x, y, z).
            inputs:     processor output (must have ``input_ids``,
                        ``image_grid_thw``).
        """
        t0 = time.perf_counter()

        self._pending_depth = torch.as_tensor(
            np.asarray(depth_m, dtype=np.float32) if isinstance(depth_m, np.ndarray) else depth_m,
            dtype=torch.float32, device=self.device,
        )
        self._pending_pos = torch.as_tensor(
            pos if not isinstance(pos, list) else np.array(pos, dtype=np.float32),
            dtype=torch.float32, device=self.device,
        )
        self._pending_quat = torch.as_tensor(
            quat_wxyz if not isinstance(quat_wxyz, list) else np.array(quat_wxyz, dtype=np.float32),
            dtype=torch.float32, device=self.device,
        )

        # Compute vision-token offset for the **last** (= current) image.
        input_ids = inputs.input_ids  # [B, S]
        image_grid_thw = inputs.image_grid_thw  # [N_images, 3]

        image_mask = (input_ids[0] == IMAGE_TOKEN_ID)  # [S]
        image_positions = image_mask.nonzero(as_tuple=False).squeeze(-1)

        if len(image_positions) == 0 or image_grid_thw is None or len(image_grid_thw) == 0:
            self._pending_offset = None
            logger.debug("set_frame: no vision tokens found; skipping frame.")
            return

        last_thw = image_grid_thw[-1]  # [3] = (t, h_patches, w_patches)
        merge = self._spatial_merge_size
        # image_grid_thw stores raw ViT patch dims; actual LLM tokens are
        # reduced by merge_size**2 due to the spatial merger in the ViT.
        n_last_tokens = int(
            last_thw[0].item() * last_thw[1].item() * last_thw[2].item()
        ) // (merge * merge)

        if len(image_positions) >= n_last_tokens:
            self._pending_offset = int(image_positions[-n_last_tokens].item())
            self._pending_n_patches = n_last_tokens

            # Update patch grid dims in the config (merged resolution).
            self.cfg.H_patches = int(last_thw[1].item()) // merge
            self.cfg.W_patches = int(last_thw[2].item()) // merge
            self.manager.cfg.H_patches = self.cfg.H_patches
            self.manager.cfg.W_patches = self.cfg.W_patches
        else:
            self._pending_offset = None
            logger.warning(
                "set_frame: image_positions (%d) < n_last_tokens (%d) "
                "(merge=%d, raw_thw=%s)",
                len(image_positions), n_last_tokens,
                merge, last_thw.tolist(),
            )

        dt = time.perf_counter() - t0
        logger.debug("set_frame done in %.2f ms", dt * 1000)

    def set_frame_if_standard_resolution(
        self,
        depth_m: torch.Tensor | np.ndarray,
        pos: torch.Tensor | np.ndarray | list,
        quat_wxyz: torch.Tensor | np.ndarray | list,
        inputs,
        expected_n_patches: int | None = None,
    ):
        """Like ``set_frame`` but silently skips lookdown / non-standard images.

        If *expected_n_patches* is given (e.g. 196 for 384×384 images with
        merge=2), any frame whose last-image patch count differs is skipped,
        preventing M_prev ≠ M_curr mismatches from ever polluting the cache.
        """
        self.set_frame(depth_m, pos, quat_wxyz, inputs)

        if expected_n_patches is not None and self._pending_n_patches is not None:
            if self._pending_n_patches != expected_n_patches:
                logger.debug(
                    "Skipping VLN-Cache for this frame: n_patches=%d != expected %d",
                    self._pending_n_patches, expected_n_patches,
                )
                # Clear pending so _trigger_on_new_frame becomes a no-op.
                self._pending_depth = None
                self._pending_offset = None
                self._pending_n_patches = None

    def get_episode_stats(self) -> dict:
        """Aggregate cache statistics for the completed episode."""
        mgr = self.manager.stats
        result = {
            "n_frames": mgr.get("n_frames", 0),
            "mean_reuse_ratio": mgr.get("mean_reuse_ratio", 0.0),
        }
        if self._step_timings:
            ratios = [s.get("reuse_ratio", 0.0) for s in self._step_timings]
            result["per_step_reuse_ratios"] = ratios
            result["cache_overhead_ms_mean"] = float(
                np.mean([s["cache_overhead_ms"] for s in self._step_timings])
            )
        return result

    def remove_all_hooks(self):
        """Detach every hook (restores vanilla behaviour)."""
        self.manager.remove_hooks()
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()

    # -------------------------------------------------------------- hooks

    def _on_visual_output(self, module, input, output):
        """Post-hook on ``model.visual``: capture ViT embeddings + trigger."""
        self._captured_embeds = output.detach()

        # Trigger on_new_frame NOW — before decoder layers run.
        self._trigger_on_new_frame()

    # -------------------------------------------------------------- internal

    def _trigger_on_new_frame(self):
        """Called from the visual hook to feed the manager."""
        if self._pending_depth is None or self._captured_embeds is None:
            return
        if self._pending_offset is None:
            self._pending_depth = None
            self._captured_embeds = None
            return

        t0 = time.perf_counter()

        n_patches = self._pending_n_patches
        patch_embeds = self._captured_embeds[-n_patches:]  # [M, d]

        self.manager.on_new_frame(
            depth=self._pending_depth,
            pos=self._pending_pos,
            quat=self._pending_quat,
            patch_embeds=patch_embeds,
            vision_token_offset=self._pending_offset,
            n_vision_tokens=n_patches,
        )

        dt = time.perf_counter() - t0

        reuse_ratio = 0.0
        if self.manager._reuse_mask is not None:
            reuse_ratio = float(self.manager._reuse_mask.float().mean().item())

        self._step_timings.append({
            "cache_overhead_ms": dt * 1000,
            "reuse_ratio": reuse_ratio,
        })

        logger.debug(
            "on_new_frame: reuse=%.1f%%  overhead=%.2f ms",
            reuse_ratio * 100, dt * 1000,
        )

        # Clear pending state (consumed).
        self._pending_depth = None
        self._captured_embeds = None
