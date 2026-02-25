import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional, Tuple

import cv2
import numpy as np


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def append_jsonl(file_path: str, payload: Dict) -> None:
    with open(file_path, 'a') as f:
        f.write(json.dumps(payload) + '\n')


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

    prev_norm = np.linalg.norm(prev_vec, axis=1, keepdims=True) + 1e-6
    curr_norm = np.linalg.norm(curr_vec, axis=1, keepdims=True) + 1e-6
    cos = np.sum((prev_vec / prev_norm) * (curr_vec / curr_norm), axis=1)
    return cos


def _yaw_homography(delta_yaw: float, intrinsic: np.ndarray) -> np.ndarray:
    c, s = np.cos(delta_yaw), np.sin(delta_yaw)
    rotation = np.array(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        dtype=np.float32,
    )
    inv_k = np.linalg.inv(intrinsic)
    return intrinsic @ rotation @ inv_k


def compute_patch_cosine_stats(
    prev_rgb: np.ndarray,
    curr_rgb: np.ndarray,
    patch_size: int = 28,
    resize_hw: Tuple[int, int] = (392, 392),
    delta_yaw: Optional[float] = None,
    intrinsic: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    prev_resize = cv2.resize(prev_rgb, resize_hw[::-1], interpolation=cv2.INTER_LINEAR)
    curr_resize = cv2.resize(curr_rgb, resize_hw[::-1], interpolation=cv2.INTER_LINEAR)

    raw_cos = _cosine_similarity_per_patch(prev_resize, curr_resize, patch_size)
    result = {
        "raw_mean": float(np.mean(raw_cos)),
        "raw_median": float(np.median(raw_cos)),
        "raw_p90": float(np.percentile(raw_cos, 90)),
    }

    if delta_yaw is not None and intrinsic is not None:
        homography = _yaw_homography(float(delta_yaw), intrinsic.astype(np.float32))
        aligned_prev = cv2.warpPerspective(
            prev_resize,
            homography,
            (curr_resize.shape[1], curr_resize.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        aligned_cos = _cosine_similarity_per_patch(aligned_prev, curr_resize, patch_size)
        result.update(
            {
                "aligned_mean": float(np.mean(aligned_cos)),
                "aligned_median": float(np.median(aligned_cos)),
                "aligned_p90": float(np.percentile(aligned_cos, 90)),
                "delta_mean": float(np.mean(aligned_cos) - np.mean(raw_cos)),
            }
        )

    return result


@dataclass
class ObservationDataWriter:
    root_dir: str
    rank: int
    mode: str

    def __post_init__(self) -> None:
        ensure_dir(self.root_dir)
        self.step_log_path = os.path.join(self.root_dir, f"step_log_rank{self.rank}.jsonl")
        self.s2_log_path = os.path.join(self.root_dir, f"s2_log_rank{self.rank}.jsonl")
        self.episode_log_path = os.path.join(self.root_dir, f"episode_log_rank{self.rank}.jsonl")
        self.manifest_path = os.path.join(self.root_dir, "manifest.json")
        if not os.path.exists(self.manifest_path):
            manifest = {
                "created_at": datetime.now().isoformat(),
                "mode": self.mode,
                "files": {
                    "step": os.path.basename(self.step_log_path),
                    "s2": os.path.basename(self.s2_log_path),
                    "episode": os.path.basename(self.episode_log_path),
                },
            }
            with open(self.manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2)

    def log_step(self, payload: Dict) -> None:
        append_jsonl(self.step_log_path, payload)

    def log_s2(self, payload: Dict) -> None:
        append_jsonl(self.s2_log_path, payload)

    def log_episode(self, payload: Dict) -> None:
        append_jsonl(self.episode_log_path, payload)
