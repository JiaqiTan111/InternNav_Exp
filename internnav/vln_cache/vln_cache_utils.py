"""
vln_cache_utils.py  –  Vectorised geometry, gating, and mask primitives
                       for VLN-Cache temporal KV reuse.

All public functions operate on **batched** tensors (B, …) and avoid
Python-level loops over patches.  Pure math: no model dependency.

Coordinate convention (Habitat default):
    Camera looks along -Z, Y points down, X points right.
    Depth sensor returns positive depth = distance along -Z.

Notation used in comments:
    M  = number of vision-encoder patches (e.g. 196 for 14×14)
    d  = hidden dimension of patch embeddings
    L  = number of Transformer decoder layers
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from internnav.utils.geometry_utils import quat_to_rot_matrix  # reuse existing


# ──────────────────────────────────────────────────────────────────────
# §0  Shared helpers
# ──────────────────────────────────────────────────────────────────────

def _quat_wxyz_to_rotmat_batch(quats: torch.Tensor) -> torch.Tensor:
    """Batch quaternion (w, x, y, z) → 3×3 rotation matrix.

    Args:
        quats:  [B, 4]  (w, x, y, z) unit quaternions.

    Returns:
        R:  [B, 3, 3]  rotation matrices.
    """
    # Normalize for numerical safety.
    quats = F.normalize(quats, dim=-1)
    w, x, y, z = quats.unbind(-1)

    # Pre-compute pairwise products.
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    R = torch.stack([
        1 - 2 * (yy + zz),     2 * (xy - wz),     2 * (xz + wy),
            2 * (xy + wz), 1 - 2 * (xx + zz),     2 * (yz - wx),
            2 * (xz - wy),     2 * (yz + wx), 1 - 2 * (xx + yy),
    ], dim=-1).reshape(-1, 3, 3)  # [B, 3, 3]
    return R


def compute_patch_centers(
    H_patches: int,
    W_patches: int,
    img_h: int,
    img_w: int,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Return the centre pixel coordinate (u, v) for each patch in raster order.

    The image is divided into H_patches × W_patches non-overlapping tiles.
    Each tile's centre pixel is returned.

    Returns:
        centers:  [M, 2]  float tensor of (u_col, v_row).
    """
    patch_h = img_h / H_patches
    patch_w = img_w / W_patches
    # Centre of tile [i, j]:  u = (j + 0.5) * patch_w,  v = (i + 0.5) * patch_h
    rows = torch.arange(H_patches, device=device, dtype=torch.float32)
    cols = torch.arange(W_patches, device=device, dtype=torch.float32)
    grid_v, grid_u = torch.meshgrid(rows, cols, indexing="ij")  # each [Hp, Wp]
    centers_u = (grid_u + 0.5) * patch_w   # [Hp, Wp]
    centers_v = (grid_v + 0.5) * patch_h   # [Hp, Wp]
    centers = torch.stack([centers_u.reshape(-1), centers_v.reshape(-1)], dim=-1)
    return centers  # [M, 2]


# ──────────────────────────────────────────────────────────────────────
# §1  View-Aligned Remap
# ──────────────────────────────────────────────────────────────────────

def _build_relative_transform(
    pos_curr: torch.Tensor,
    quat_curr: torch.Tensor,
    pos_prev: torch.Tensor,
    quat_prev: torch.Tensor,
) -> torch.Tensor:
    r"""Compute ΔT_{curr→prev} = T_prev^{-1} · T_curr  (4×4).

    Transforms a point from *current* camera coords to *previous* camera coords.

    Args:
        pos_curr, pos_prev:   [3]  world-frame positions.
        quat_curr, quat_prev: [4]  world-frame orientations (w, x, y, z).

    Returns:
        delta_T:  [4, 4]  rigid transform  curr → prev.
    """
    R_curr = _quat_wxyz_to_rotmat_batch(quat_curr.unsqueeze(0))[0]  # [3,3]
    R_prev = _quat_wxyz_to_rotmat_batch(quat_prev.unsqueeze(0))[0]  # [3,3]

    # World→camera:  X_cam = R^T (X_world - t)
    # T_world2cam = [[R^T, -R^T t], [0, 1]]
    T_curr = torch.eye(4, device=pos_curr.device, dtype=pos_curr.dtype)
    T_curr[:3, :3] = R_curr
    T_curr[:3, 3] = pos_curr

    T_prev = torch.eye(4, device=pos_prev.device, dtype=pos_prev.dtype)
    T_prev[:3, :3] = R_prev
    T_prev[:3, 3] = pos_prev

    # T_prev_inv · T_curr  →  transforms world-frame coords expressed in
    # curr camera to prev camera.
    delta_T = torch.linalg.inv(T_prev) @ T_curr  # [4, 4]
    return delta_T


def view_aligned_remap(
    depth_curr: torch.Tensor,
    depth_prev: torch.Tensor,
    pos_curr: torch.Tensor,
    quat_curr: torch.Tensor,
    pos_prev: torch.Tensor,
    quat_prev: torch.Tensor,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    H_patches: int,
    W_patches: int,
    img_h: int,
    img_w: int,
    occlusion_eps: float = 0.25,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Geometric reprojection of current patch centres into previous frame.

    Pipeline (fully vectorised, no Python loops over patches):
        1. Unproject each patch centre to 3-D using depth_curr.
        2. Transform to previous camera via ΔT_{curr→prev}.
        3. Reproject onto previous image plane.
        4. Z-buffer check against depth_prev.

    Args:
        depth_curr:  [H, W]  depth map of current frame (metres, >0).
        depth_prev:  [H, W]  depth map of previous frame.
        pos_curr, pos_prev:   [3]  agent position (world frame).
        quat_curr, quat_prev: [4]  agent quaternion (w,x,y,z).
        fx, fy, cx, cy:  camera intrinsics (pixels).
        H_patches, W_patches:  patch grid resolution (e.g. 14, 14).
        img_h, img_w:  raw image resolution in pixels.
        occlusion_eps: depth tolerance (metres) for Z-buffer check.

    Returns:
        matched_idx:  [M]  int64 – coarse matched patch index in prev frame
                           (-1 if invalid / occluded).
        valid_mask:    [M]  bool  – True where geometric match is trustworthy.
    """
    device = depth_curr.device
    M = H_patches * W_patches

    # Step 1: patch-centre pixel coords  →  3-D points in current camera frame.
    centers = compute_patch_centers(H_patches, W_patches, img_h, img_w, device)  # [M, 2]
    u, v = centers[:, 0], centers[:, 1]  # [M]

    # Bilinear-sample depth at sub-pixel patch centres.
    # grid_sample expects input [1,1,H,W] and grid [1,1,M,2] in [-1,1].
    grid_u = 2.0 * u / (img_w - 1) - 1.0
    grid_v = 2.0 * v / (img_h - 1) - 1.0
    grid = torch.stack([grid_u, grid_v], dim=-1).reshape(1, 1, M, 2)  # [1,1,M,2]
    z_curr = F.grid_sample(
        depth_curr.unsqueeze(0).unsqueeze(0),  # [1,1,H,W]
        grid, mode="bilinear", align_corners=True,
    ).reshape(M)  # [M]

    # Validity: depth must be positive and finite.
    depth_valid = (z_curr > 1e-3) & torch.isfinite(z_curr)  # [M]

    # Unproject (Habitat convention: camera looks along -Z).
    # x_cam = (u - cx) * z / fx,  y_cam = (v - cy) * z / fy,  z_cam = z
    # But Habitat depth = distance along the -Z axis → actual Z_cam = -z.
    # We work in the *world-aligned camera frame* where positive depth = positive Z.
    X_cam = torch.stack([
        (u - cx) * z_curr / fx,
        (v - cy) * z_curr / fy,
        z_curr,
    ], dim=-1)  # [M, 3]

    # Step 2: transform to previous camera frame.
    delta_T = _build_relative_transform(pos_curr, quat_curr, pos_prev, quat_prev)  # [4, 4]
    X_hom = torch.cat([X_cam, torch.ones(M, 1, device=device)], dim=-1)  # [M, 4]
    X_prev = (delta_T @ X_hom.T).T[:, :3]  # [M, 3]

    # Step 3: reproject into previous image plane.
    z_prev_proj = X_prev[:, 2]  # [M]
    z_safe = z_prev_proj.clamp(min=1e-6)
    u_prev = fx * X_prev[:, 0] / z_safe + cx  # [M]
    v_prev = fy * X_prev[:, 1] / z_safe + cy  # [M]

    # Bounds check.
    in_bounds = (
        (u_prev >= 0) & (u_prev < img_w) &
        (v_prev >= 0) & (v_prev < img_h) &
        (z_prev_proj > 1e-3)
    )  # [M]

    # Map projected pixel to patch index in prev frame.
    patch_w = img_w / W_patches
    patch_h = img_h / H_patches
    col_prev = (u_prev / patch_w).long().clamp(0, W_patches - 1)  # [M]
    row_prev = (v_prev / patch_h).long().clamp(0, H_patches - 1)  # [M]
    matched_idx = row_prev * W_patches + col_prev  # [M]

    # Step 4: Z-buffer occlusion check.
    # Sample depth_prev at the projected pixel.
    grid_u2 = 2.0 * u_prev / (img_w - 1) - 1.0
    grid_v2 = 2.0 * v_prev / (img_h - 1) - 1.0
    grid2 = torch.stack([grid_u2, grid_v2], dim=-1).reshape(1, 1, M, 2)
    # Clamp grid to valid range to avoid out-of-bound during grid_sample.
    grid2 = grid2.clamp(-1.0, 1.0)
    z_prev_sampled = F.grid_sample(
        depth_prev.unsqueeze(0).unsqueeze(0),
        grid2, mode="bilinear", align_corners=True,
    ).reshape(M)  # [M]

    depth_consistent = (z_prev_proj - z_prev_sampled).abs() < (
        occlusion_eps * z_curr.clamp(min=1.0)
    )  # [M]  – relative tolerance: eps × max(depth, 1m)

    valid_mask = depth_valid & in_bounds & depth_consistent  # [M]

    # Mark invalid entries.
    matched_idx[~valid_mask] = -1

    return matched_idx, valid_mask


# ──────────────────────────────────────────────────────────────────────
# §1b  Neighbourhood Refinement (3×3 local cosine search)
# ──────────────────────────────────────────────────────────────────────

def neighborhood_refinement(
    patch_curr: torch.Tensor,
    patch_prev: torch.Tensor,
    matched_idx: torch.Tensor,
    valid_mask: torch.Tensor,
    W_patches: int,
    H_patches: int,
    k: int = 1,
    tau_v: float = 0.85,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Refine coarse geometric matches via local cosine similarity.

    For each valid coarse match j(i), search a (2k+1)×(2k+1) neighbourhood
    in the previous frame's patch grid and select the token with highest
    cosine similarity.

    Args:
        patch_curr:  [M, d]  current-frame patch embeddings.
        patch_prev:  [M, d]  previous-frame patch embeddings.
        matched_idx: [M]     coarse match indices from view_aligned_remap.
        valid_mask:  [M]     bool, True where coarse match exists.
        W_patches, H_patches: patch grid dimensions.
        k: neighbourhood half-width (k=1 → 3×3).
        tau_v: cosine-similarity threshold for visual static gating.

    Returns:
        refined_idx:   [M]   refined best-match index (-1 if invalid).
        m_vis:         [M]   bool – visual static gate (True = reusable).
    """
    M, d = patch_curr.shape
    device = patch_curr.device

    # Build relative offsets for the (2k+1)² neighbourhood.
    offsets = []
    for dr in range(-k, k + 1):
        for dc in range(-k, k + 1):
            offsets.append((dr, dc))
    N_neigh = len(offsets)
    dr_offsets = torch.tensor([o[0] for o in offsets], device=device)  # [N_neigh]
    dc_offsets = torch.tensor([o[1] for o in offsets], device=device)  # [N_neigh]

    # Convert coarse match index → (row, col) in patch grid.
    row_c = matched_idx // W_patches  # [M]
    col_c = matched_idx % W_patches   # [M]

    # Expand to [M, N_neigh].
    rows_nb = row_c.unsqueeze(1) + dr_offsets.unsqueeze(0)  # [M, N_neigh]
    cols_nb = col_c.unsqueeze(1) + dc_offsets.unsqueeze(0)  # [M, N_neigh]

    # Clamp to grid boundary.
    rows_nb = rows_nb.clamp(0, H_patches - 1)
    cols_nb = cols_nb.clamp(0, W_patches - 1)

    # Flat indices for gathering.
    nb_idx = rows_nb * W_patches + cols_nb  # [M, N_neigh]

    # Gather neighbour embeddings from previous frame.
    nb_embeds = patch_prev[nb_idx.long()]  # [M, N_neigh, d]

    # Cosine similarity between current patch and each neighbour.
    # Normalize embeddings.
    curr_norm = F.normalize(patch_curr, dim=-1).unsqueeze(1)  # [M, 1, d]
    nb_norm = F.normalize(nb_embeds, dim=-1)                  # [M, N_neigh, d]
    cos_sim = (curr_norm * nb_norm).sum(dim=-1)               # [M, N_neigh]

    # Select best neighbour.
    best_sim, best_local = cos_sim.max(dim=-1)  # [M], [M]
    refined_idx = nb_idx[torch.arange(M, device=device), best_local]  # [M]

    # Visual static gate.
    m_vis = (best_sim > tau_v) & valid_mask  # [M]

    # Invalidate entries without a valid coarse match.
    refined_idx[~valid_mask] = -1

    return refined_idx, m_vis


# ──────────────────────────────────────────────────────────────────────
# §2  Semantic Refresh Gate
# ──────────────────────────────────────────────────────────────────────

def compute_semantic_refresh_gate(
    cross_attn_contrib: torch.Tensor,
    prev_cross_attn_contrib: torch.Tensor | None,
    tau_s: float = 0.5,
    delta_s: float = 0.3,
) -> torch.Tensor:
    r"""Semantic refresh gate that forces recomputation for task-relevant tokens.

    Uses cross-attention output contribution norm as a proxy for language-vision
    relevance (compatible with FlashAttention where raw weights are unavailable).

    Gate fires (True → must refresh) when the token's relevance is high
    or its relevance has changed significantly since last frame.

    Formula:
        m_sem(i) = 𝟙[ s(i) > τ_s  ∨  |s(i) − s_{t−1}(i)| > δ_s ]

    Args:
        cross_attn_contrib:       [M]  current relevance scores  s_t(i).
        prev_cross_attn_contrib:  [M]  or None, previous scores  s_{t−1}(i).
        tau_s:   absolute relevance threshold.
        delta_s: temporal change threshold.

    Returns:
        m_sem:  [M]  bool – True means token MUST be refreshed (recomputed).
    """
    above_threshold = cross_attn_contrib > tau_s  # [M]

    if prev_cross_attn_contrib is not None:
        delta = (cross_attn_contrib - prev_cross_attn_contrib).abs()
        changed = delta > delta_s  # [M]
    else:
        # First frame: refresh everything.
        changed = torch.ones_like(above_threshold)

    m_sem = above_threshold | changed  # [M]
    return m_sem


# ──────────────────────────────────────────────────────────────────────
# §3  Layer-Adaptive Reuse
# ──────────────────────────────────────────────────────────────────────

def compute_layer_entropy(attn_weights: torch.Tensor) -> torch.Tensor:
    """Compute mean attention entropy for a layer.

    Higher entropy → more uniform attention → safer to reuse aggressively.

    When FlashAttention is used and raw weights are unavailable, the caller
    should pass a proxy (e.g. softmax of QK^T for a single head computed
    on a small token subset).

    Args:
        attn_weights:  [B, n_heads, S, S]  or  [B, S, S]  attention probs.

    Returns:
        H:  scalar, mean entropy in nats across batch and heads.
    """
    # Clamp for numerical stability in log.
    p = attn_weights.clamp(min=1e-8)
    ent = -(p * p.log()).sum(dim=-1)  # [B, n_heads, S] or [B, S]
    return ent.mean()


def layer_adaptive_threshold(
    H_ell: torch.Tensor,
    tau_base: float = 0.85,
    tau_range: float = 0.15,
    H_low: float = 1.0,
    H_high: float = 4.0,
) -> float:
    r"""Compute per-layer cosine threshold τ_v^ℓ = f(H_t^ℓ).

    Strategy: low entropy → attention is peaked → reuse is risky → higher τ.
              high entropy → uniform attention → reuse is safe → lower τ.

    Linear schedule clamped to [tau_base - tau_range, tau_base + tau_range]:
        τ_v^ℓ = tau_base + tau_range · (1 − clamp((H − H_low)/(H_high − H_low)))

    Args:
        H_ell:    scalar tensor – layer entropy.
        tau_base: baseline cosine threshold.
        tau_range: half-width of adaptation range.
        H_low, H_high: entropy anchors for the linear schedule.

    Returns:
        tau_v:  float – adapted threshold for this layer.
    """
    h = H_ell.item() if isinstance(H_ell, torch.Tensor) else float(H_ell)
    t = (h - H_low) / max(H_high - H_low, 1e-6)
    t = max(0.0, min(1.0, t))
    # High entropy → low threshold (aggressive reuse).
    tau_v = tau_base + tau_range * (1.0 - t)
    return tau_v


# ──────────────────────────────────────────────────────────────────────
# §4  Final Mask Assembly
# ──────────────────────────────────────────────────────────────────────

def build_reuse_mask(
    m_vis: torch.Tensor,
    m_sem: torch.Tensor,
) -> torch.Tensor:
    r"""Combine visual static gate and semantic refresh gate.

    A token is reusable iff:  m_vis is True  AND  m_sem is False.
        reuse(i) = m_vis(i) ∧ ¬m_sem(i)

    Args:
        m_vis:  [M]  bool – True if visually static (ok to reuse).
        m_sem:  [M]  bool – True if semantically important (must refresh).

    Returns:
        reuse:  [M]  bool – True where previous KV can be copied.
    """
    return m_vis & (~m_sem)
