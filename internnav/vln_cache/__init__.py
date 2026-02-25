# VLN-Cache: Training-free KV-Cache reuse for Vision-Language Navigation.
#
# This module implements view-aligned temporal token reuse to accelerate
# Transformer inference in VLN-CE without retraining. Three sub-algorithms
# are provided:
#   1. View-Aligned Remap   – depth+pose geometric reprojection
#   2. Semantic Refresh Gate – task-relevance-driven forced refresh
#   3. Layer-Adaptive Reuse  – per-layer entropy-based threshold scheduling
#
# Reference layout:
#   vln_cache_utils.py          – pure-math vectorised geometry & mask generation
#   vln_cache_wrapper.py        – hook-based KV reuse manager
#   evaluator_integration.py    – bridge between evaluator and VLNCacheManager
#   run_comparison.py           – baseline vs VLN-Cache comparison script

from .vln_cache_utils import (
    build_reuse_mask,
    compute_layer_entropy,
    compute_patch_centers,
    compute_semantic_refresh_gate,
    layer_adaptive_threshold,
    neighborhood_refinement,
    view_aligned_remap,
)
from .vln_cache_wrapper import VLNCacheConfig, VLNCacheManager
from .evaluator_integration import VLNCacheHook

__all__ = [
    "build_reuse_mask",
    "compute_layer_entropy",
    "compute_patch_centers",
    "compute_semantic_refresh_gate",
    "layer_adaptive_threshold",
    "neighborhood_refinement",
    "view_aligned_remap",
    "VLNCacheConfig",
    "VLNCacheManager",
    "VLNCacheHook",
]
