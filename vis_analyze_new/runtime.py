import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import torch


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def append_jsonl(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(payload) + "\n")


def _cosine_similarity_per_patch(prev_img: np.ndarray, curr_img: np.ndarray, patch_size: int) -> np.ndarray:
    h, w, _ = curr_img.shape
    rows = h // patch_size
    cols = w // patch_size
    prev_crop = prev_img[: rows * patch_size, : cols * patch_size]
    curr_crop = curr_img[: rows * patch_size, : cols * patch_size]

    prev_patches = prev_crop.reshape(rows, patch_size, cols, patch_size, 3).transpose(0, 2, 1, 3, 4)
    curr_patches = curr_crop.reshape(rows, patch_size, cols, patch_size, 3).transpose(0, 2, 1, 3, 4)

    prev_vec = prev_patches.reshape(rows * cols, -1).astype(np.float32)
    curr_vec = curr_patches.reshape(rows * cols, -1).astype(np.float32)
    prev_vec /= np.linalg.norm(prev_vec, axis=1, keepdims=True) + 1e-6
    curr_vec /= np.linalg.norm(curr_vec, axis=1, keepdims=True) + 1e-6
    return np.sum(prev_vec * curr_vec, axis=1)


def _extract_patch_vectors(img: np.ndarray, patch_size: int) -> Tuple[np.ndarray, int, int]:
    h, w, _ = img.shape
    rows = h // patch_size
    cols = w // patch_size
    crop = img[: rows * patch_size, : cols * patch_size]
    patches = crop.reshape(rows, patch_size, cols, patch_size, 3).transpose(0, 2, 1, 3, 4)
    vec = patches.reshape(rows * cols, -1).astype(np.float32)
    vec /= np.linalg.norm(vec, axis=1, keepdims=True) + 1e-6
    return vec, rows, cols


def _yaw_homography(delta_yaw: float, intrinsic: np.ndarray) -> np.ndarray:
    c, s = np.cos(delta_yaw), np.sin(delta_yaw)
    rotation = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32)
    inv_k = np.linalg.inv(intrinsic)
    return intrinsic @ rotation @ inv_k


def _wrap_to_pi(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def _quat_wxyz_to_rotmat(quat_wxyz: np.ndarray) -> np.ndarray:
    q = np.asarray(quat_wxyz, dtype=np.float64)
    if q.shape[0] != 4:
        raise ValueError("quat_wxyz must have shape (4,)")
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = q / n
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _parse_pose(pose: Optional[Dict[str, Any]]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if pose is None:
        return None
    pos = np.asarray(pose.get("position", []), dtype=np.float64)
    quat = np.asarray(pose.get("quaternion_wxyz", []), dtype=np.float64)
    if pos.shape[0] != 3 or quat.shape[0] != 4:
        return None
    return _quat_wxyz_to_rotmat(quat), pos


def _geometric_patch_alignment(
    prev_rgb: np.ndarray,
    curr_rgb: np.ndarray,
    prev_depth: np.ndarray,
    curr_depth: np.ndarray,
    intrinsic: np.ndarray,
    patch_size: int,
    resize_hw: Tuple[int, int],
    prev_pose: Dict[str, Any],
    curr_pose: Dict[str, Any],
    neighbor_radius: int,
    occlusion_eps_mm: float,
) -> Tuple[np.ndarray, np.ndarray]:
    prev_rt = _parse_pose(prev_pose)
    curr_rt = _parse_pose(curr_pose)
    if prev_rt is None or curr_rt is None:
        raise ValueError("invalid_pose")

    prev_r, prev_t = prev_rt
    curr_r, curr_t = curr_rt

    prev_vec, rows, cols = _extract_patch_vectors(prev_rgb, patch_size)
    curr_vec, _, _ = _extract_patch_vectors(curr_rgb, patch_size)
    n_patches = rows * cols

    prev_d = prev_depth
    curr_d = curr_depth
    if prev_d.ndim == 3:
        prev_d = prev_d[..., 0]
    if curr_d.ndim == 3:
        curr_d = curr_d[..., 0]
    prev_d = cv2.resize(prev_d.astype(np.float32), resize_hw[::-1], interpolation=cv2.INTER_NEAREST)
    curr_d = cv2.resize(curr_d.astype(np.float32), resize_hw[::-1], interpolation=cv2.INTER_NEAREST)

    fx, fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
    cx, cy = float(intrinsic[0, 2]), float(intrinsic[1, 2])
    h, w = curr_d.shape

    aligned_cos = np.full((n_patches,), -1.0, dtype=np.float32)
    valid_mask = np.zeros((n_patches,), dtype=bool)

    for idx in range(n_patches):
        py = idx // cols
        px = idx % cols
        u = (px + 0.5) * patch_size - 0.5
        v = (py + 0.5) * patch_size - 0.5
        uu = int(np.clip(round(u), 0, w - 1))
        vv = int(np.clip(round(v), 0, h - 1))
        z = float(curr_d[vv, uu])
        if (not np.isfinite(z)) or z <= 1e-3:
            continue

        x = (u - cx) * z / max(fx, 1e-6)
        y = (v - cy) * z / max(fy, 1e-6)
        pt_curr = np.array([x, y, z], dtype=np.float64)

        pt_world = curr_r @ pt_curr + curr_t
        pt_prev = prev_r.T @ (pt_world - prev_t)
        z_prev = float(pt_prev[2])
        if (not np.isfinite(z_prev)) or z_prev <= 1e-3:
            continue

        u_prev = fx * float(pt_prev[0]) / z_prev + cx
        v_prev = fy * float(pt_prev[1]) / z_prev + cy
        if not (0.0 <= u_prev < w and 0.0 <= v_prev < h):
            continue

        if occlusion_eps_mm > 0:
            pu = int(np.clip(round(u_prev), 0, w - 1))
            pv = int(np.clip(round(v_prev), 0, h - 1))
            z_prev_depth = float(prev_d[pv, pu])
            if np.isfinite(z_prev_depth) and z_prev_depth > 1e-3:
                if abs(z_prev_depth - z_prev) > float(occlusion_eps_mm):
                    continue

        base_px = int(np.floor(u_prev / patch_size))
        base_py = int(np.floor(v_prev / patch_size))
        best = -1.0
        r0 = max(0, base_py - neighbor_radius)
        r1 = min(rows - 1, base_py + neighbor_radius)
        c0 = max(0, base_px - neighbor_radius)
        c1 = min(cols - 1, base_px + neighbor_radius)
        if r0 > r1 or c0 > c1:
            continue

        cur_vec = curr_vec[idx]
        for ry in range(r0, r1 + 1):
            for rx in range(c0, c1 + 1):
                cand = ry * cols + rx
                score = float(np.dot(cur_vec, prev_vec[cand]))
                if score > best:
                    best = score

        if best < -0.5:
            continue
        aligned_cos[idx] = best
        valid_mask[idx] = True

    return aligned_cos, valid_mask


def compute_patch_cosine_stats(
    prev_rgb: np.ndarray,
    curr_rgb: np.ndarray,
    patch_size: int = 28,
    resize_hw: Tuple[int, int] = (392, 392),
    delta_yaw: Optional[float] = None,
    intrinsic: Optional[np.ndarray] = None,
    prev_depth: Optional[np.ndarray] = None,
    curr_depth: Optional[np.ndarray] = None,
    prev_pose: Optional[Dict[str, Any]] = None,
    curr_pose: Optional[Dict[str, Any]] = None,
    alignment_method: str = "depth_pose",
    neighbor_radius: int = 1,
    occlusion_eps_mm: float = 250.0,
    tau: float = 0.7,
) -> Dict[str, float]:
    prev_resize = cv2.resize(prev_rgb, resize_hw[::-1], interpolation=cv2.INTER_LINEAR)
    curr_resize = cv2.resize(curr_rgb, resize_hw[::-1], interpolation=cv2.INTER_LINEAR)

    raw_cos = _cosine_similarity_per_patch(prev_resize, curr_resize, patch_size)
    raw_reuse = raw_cos > tau
    result = {
        "raw_mean": float(np.mean(raw_cos)),
        "raw_median": float(np.median(raw_cos)),
        "raw_p10": float(np.percentile(raw_cos, 10)),
        "raw_p90": float(np.percentile(raw_cos, 90)),
        "raw_reuse_frac": float(np.mean(raw_reuse)),
    }

    if intrinsic is not None:
        intrinsic = intrinsic.astype(np.float32).copy()
        src_h, src_w = prev_rgb.shape[:2]
        dst_h, dst_w = resize_hw
        if src_w > 0 and src_h > 0:
            sx = float(dst_w) / float(src_w)
            sy = float(dst_h) / float(src_h)
            intrinsic[0, 0] *= sx
            intrinsic[0, 2] *= sx
            intrinsic[1, 1] *= sy
            intrinsic[1, 2] *= sy

    can_depth_pose = (
        alignment_method == "depth_pose"
        and intrinsic is not None
        and prev_depth is not None
        and curr_depth is not None
        and prev_pose is not None
        and curr_pose is not None
    )
    can_yaw_h = alignment_method == "yaw_homography" and (delta_yaw is not None) and (intrinsic is not None)

    if can_depth_pose or can_yaw_h:
        if can_depth_pose:
            try:
                aligned_cos, valid_mask = _geometric_patch_alignment(
                    prev_rgb=prev_resize,
                    curr_rgb=curr_resize,
                    prev_depth=prev_depth,
                    curr_depth=curr_depth,
                    intrinsic=intrinsic,
                    patch_size=patch_size,
                    resize_hw=resize_hw,
                    prev_pose=prev_pose,
                    curr_pose=curr_pose,
                    neighbor_radius=max(int(neighbor_radius), 0),
                    occlusion_eps_mm=float(occlusion_eps_mm),
                )
                aligned_cos = np.where(valid_mask, aligned_cos, raw_cos)
                aligned_mode = "depth_pose"
            except Exception:
                can_depth_pose = False

        if not can_depth_pose:
            h = _yaw_homography(_wrap_to_pi(float(delta_yaw)), intrinsic)
            aligned_prev = cv2.warpPerspective(
                prev_resize,
                h,
                (curr_resize.shape[1], curr_resize.shape[0]),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )
            aligned_cos = _cosine_similarity_per_patch(aligned_prev, curr_resize, patch_size)
            valid_mask = np.ones_like(aligned_cos, dtype=bool)
            aligned_mode = "yaw_homography"

        aligned_reuse = aligned_cos > tau
        pseudo_dynamic = np.logical_and(aligned_reuse, np.logical_not(raw_reuse))
        result.update(
            {
                "aligned_mean": float(np.mean(aligned_cos)),
                "aligned_median": float(np.median(aligned_cos)),
                "aligned_p10": float(np.percentile(aligned_cos, 10)),
                "aligned_p90": float(np.percentile(aligned_cos, 90)),
                "aligned_reuse_frac": float(np.mean(aligned_reuse)),
                "delta_mean": float(np.mean(aligned_cos) - np.mean(raw_cos)),
                "pseudo_dynamic_rate": float(np.mean(pseudo_dynamic)),
                "lost_reuse": float(np.mean(aligned_reuse) - np.mean(raw_reuse)),
                "aligned_valid_frac": float(np.mean(valid_mask.astype(np.float32))),
                "aligned_mode": aligned_mode,
            }
        )

    return result


def _to_numpy_tokens(output: Any) -> Optional[np.ndarray]:
    if torch.is_tensor(output):
        tensor = output
    elif isinstance(output, (tuple, list)) and len(output) > 0 and torch.is_tensor(output[0]):
        tensor = output[0]
    else:
        return None

    with torch.no_grad():
        array = tensor.detach().float().cpu().numpy()

    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    return array


class VisualTokenHook:
    def __init__(self, model: torch.nn.Module, enabled: bool = False):
        self.enabled = enabled
        self._handle = None
        self._latest = None
        if not enabled:
            return

        module = getattr(model, "visual", None)
        if module is None and hasattr(model, "get_model"):
            inner = model.get_model()
            module = getattr(inner, "visual", None)
        if module is None:
            self.enabled = False
            return

        def _hook(_module, _inputs, output):
            arr = _to_numpy_tokens(output)
            if arr is not None:
                self._latest = arr.astype(np.float16)

        self._handle = module.register_forward_hook(_hook)

    def consume(self) -> Optional[np.ndarray]:
        value = self._latest
        self._latest = None
        return value

    def close(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


def _normalize_prob(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    v = np.clip(v, eps, None)
    return v / v.sum()


def _entropy(p: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(p, eps, None)
    return float(-(p * np.log(p)).sum())


def probe_language_vision_saliency(
    model: torch.nn.Module,
    inputs: Any,
    image_token_id: Optional[int] = None,
    topk: int = 32,
) -> Dict[str, Any]:
    if image_token_id is None:
        image_token_id = int(getattr(model.config, "image_token_id", 151655))

    fw_kwargs: Dict[str, Any] = {
        "input_ids": inputs.input_ids,
        "attention_mask": getattr(inputs, "attention_mask", None),
        "output_attentions": True,
        "return_dict": True,
        "use_cache": False,
    }
    if hasattr(inputs, "pixel_values"):
        fw_kwargs["pixel_values"] = inputs.pixel_values
    if hasattr(inputs, "image_grid_thw"):
        image_grid = inputs.image_grid_thw
        if isinstance(image_grid, (tuple, list)):
            image_grid = torch.cat([x.unsqueeze(0) for x in image_grid], dim=0)
        fw_kwargs["image_grid_thw"] = image_grid

    try:
        with torch.no_grad():
            outputs = model(**fw_kwargs)
    except Exception as exc:
        return {"ok": False, "error": f"probe_forward_failed: {exc}"}

    attentions = getattr(outputs, "attentions", None)
    if attentions is None or len(attentions) == 0:
        return {"ok": False, "error": "no_attentions_returned"}

    last_attn = attentions[-1]
    if last_attn is None or last_attn.ndim != 4:
        return {"ok": False, "error": "invalid_attention_tensor"}

    input_ids = inputs.input_ids[0]
    attention_mask = getattr(inputs, "attention_mask", None)
    valid_mask = torch.ones_like(input_ids, dtype=torch.bool)
    if attention_mask is not None:
        valid_mask = attention_mask[0] > 0

    vision_idx = torch.where((input_ids == image_token_id) & valid_mask)[0]
    lang_idx = torch.where((input_ids != image_token_id) & valid_mask)[0]
    if vision_idx.numel() == 0 or lang_idx.numel() == 0:
        return {"ok": False, "error": "missing_language_or_vision_tokens"}

    attn_slice = last_attn[0][:, lang_idx][:, :, vision_idx]
    saliency = attn_slice.mean(dim=(0, 1)).float().cpu().numpy()
    saliency = _normalize_prob(saliency)

    k = int(min(topk, saliency.shape[0]))
    top_idx = np.argpartition(-saliency, kth=k - 1)[:k]
    top_idx = top_idx[np.argsort(-saliency[top_idx])]

    return {
        "ok": True,
        "num_vision_tokens": int(saliency.shape[0]),
        "entropy": _entropy(saliency),
        "topk_indices": [int(x) for x in top_idx.tolist()],
        "topk_scores": [float(x) for x in saliency[top_idx].tolist()],
        "saliency_vector": saliency.astype(np.float32),
    }


@dataclass
class ObservationDataWriterV2:
    root_dir: str
    rank: int
    mode: str
    save_patch_tokens: bool = False
    save_rgbd: bool = False

    def __post_init__(self) -> None:
        ensure_dir(self.root_dir)
        self.step_log_path = os.path.join(self.root_dir, f"step_log_rank{self.rank}.jsonl")
        self.s2_log_path = os.path.join(self.root_dir, f"s2_log_rank{self.rank}.jsonl")
        self.episode_log_path = os.path.join(self.root_dir, f"episode_log_rank{self.rank}.jsonl")
        self.saliency_log_path = os.path.join(self.root_dir, f"saliency_log_rank{self.rank}.jsonl")

        self.patch_dir = os.path.join(self.root_dir, "patch_tokens")
        self.rgb_dir = os.path.join(self.root_dir, "rgb")
        self.depth_dir = os.path.join(self.root_dir, "depth")
        self.saliency_dir = os.path.join(self.root_dir, "saliency")
        if self.save_patch_tokens:
            ensure_dir(self.patch_dir)
        if self.save_rgbd:
            ensure_dir(self.rgb_dir)
            ensure_dir(self.depth_dir)
        ensure_dir(self.saliency_dir)

        manifest_path = os.path.join(self.root_dir, "manifest.json")
        if not os.path.exists(manifest_path):
            with open(manifest_path, "w") as f:
                json.dump(
                    {
                        "created_at": datetime.now().isoformat(),
                        "mode": self.mode,
                        "files": {
                            "step": os.path.basename(self.step_log_path),
                            "s2": os.path.basename(self.s2_log_path),
                            "episode": os.path.basename(self.episode_log_path),
                            "saliency": os.path.basename(self.saliency_log_path),
                        },
                    },
                    f,
                    indent=2,
                )

    def log_step(self, payload: Dict[str, Any]) -> None:
        append_jsonl(self.step_log_path, payload)

    def log_s2(self, payload: Dict[str, Any]) -> None:
        append_jsonl(self.s2_log_path, payload)

    def log_episode(self, payload: Dict[str, Any]) -> None:
        append_jsonl(self.episode_log_path, payload)

    def log_saliency(self, payload: Dict[str, Any]) -> None:
        append_jsonl(self.saliency_log_path, payload)

    def dump_patch_tokens(self, scene_id: str, episode_id: int, step_id: int, tokens: Optional[np.ndarray]) -> Optional[str]:
        if (not self.save_patch_tokens) or tokens is None:
            return None
        filename = f"scene_{scene_id}_ep_{episode_id:04d}_step_{step_id:04d}.npz"
        path = os.path.join(self.patch_dir, filename)
        np.savez_compressed(path, tokens=tokens.astype(np.float16))
        return os.path.relpath(path, self.root_dir)

    def dump_rgbd(self, scene_id: str, episode_id: int, step_id: int, rgb: np.ndarray, depth: np.ndarray) -> Dict[str, Optional[str]]:
        if not self.save_rgbd:
            return {"rgb": None, "depth": None}
        rgb_name = f"scene_{scene_id}_ep_{episode_id:04d}_step_{step_id:04d}.jpg"
        depth_name = f"scene_{scene_id}_ep_{episode_id:04d}_step_{step_id:04d}.npy"
        rgb_path = os.path.join(self.rgb_dir, rgb_name)
        depth_path = os.path.join(self.depth_dir, depth_name)
        cv2.imwrite(rgb_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        np.save(depth_path, depth)
        return {
            "rgb": os.path.relpath(rgb_path, self.root_dir),
            "depth": os.path.relpath(depth_path, self.root_dir),
        }

    def dump_saliency(self, scene_id: str, episode_id: int, step_id: int, saliency: np.ndarray) -> str:
        filename = f"scene_{scene_id}_ep_{episode_id:04d}_step_{step_id:04d}.npz"
        path = os.path.join(self.saliency_dir, filename)
        np.savez_compressed(path, saliency=saliency.astype(np.float32))
        return os.path.relpath(path, self.root_dir)
