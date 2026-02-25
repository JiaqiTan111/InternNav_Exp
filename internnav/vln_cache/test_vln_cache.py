"""
test_vln_cache.py – Comprehensive test suite for VLN-Cache module.

Tests cover:
    §0  Infrastructure (import, config, patch centres)
    §1  View-Aligned Remap (identity, small motion)
    §2  Neighbourhood Refinement + visual gating
    §3  Semantic Refresh Gate
    §4  Build Reuse Mask
    §5  Layer-Adaptive Threshold
    §6  VLNCacheManager lifecycle (frame commit, stats)
    §7  Hook splice logic (mock cache)
    §8  VLNCacheHook integration (mock model)
    §9  Performance benchmark

Usage:
    cd /home/iflab-zzh-intern/Zhihao_VLN/InternNav_Exp
    python -m internnav.vln_cache.test_vln_cache
"""

from __future__ import annotations

import sys
import time
import unittest

import numpy as np
import torch

# ── §0 ──────────────────────────────────────────────────────────────────

class TestImportsAndConfig(unittest.TestCase):
    def test_01_imports(self):
        from internnav.vln_cache import (
            VLNCacheConfig,
            VLNCacheManager,
            VLNCacheHook,
            compute_patch_centers,
            view_aligned_remap,
            neighborhood_refinement,
            compute_semantic_refresh_gate,
            build_reuse_mask,
            layer_adaptive_threshold,
            compute_layer_entropy,
        )
        self.assertTrue(True)

    def test_02_patch_centers(self):
        from internnav.vln_cache import compute_patch_centers
        centers = compute_patch_centers(14, 14, 480, 480)
        self.assertEqual(centers.shape, (196, 2))
        # First patch centre should be near (17.1, 17.1)
        self.assertAlmostEqual(centers[0, 0].item(), 480 / 14 / 2, places=1)

    def test_03_config_defaults(self):
        from internnav.vln_cache import VLNCacheConfig
        cfg = VLNCacheConfig()
        self.assertEqual(cfg.occlusion_eps, 0.25)
        self.assertEqual(cfg.tau_v_base, 0.85)
        self.assertFalse(cfg.layer_adaptive)


# ── §1 ──────────────────────────────────────────────────────────────────

class TestViewAlignedRemap(unittest.TestCase):
    def _make_frame(self, H=480, W=480):
        depth = torch.ones(H, W) * 2.0
        pos = torch.zeros(3)
        quat = torch.tensor([1.0, 0.0, 0.0, 0.0])  # identity
        return depth, pos, quat

    def test_identity_pose(self):
        from internnav.vln_cache import view_aligned_remap
        depth, pos, quat = self._make_frame()
        fx = fy = 240.0
        cx = cy = 239.5
        matched, valid = view_aligned_remap(
            depth, depth, pos, quat, pos, quat,
            fx, fy, cx, cy, 14, 14, 480, 480,
        )
        # Identity → every patch maps to itself.
        self_map = (matched == torch.arange(196)).float().mean()
        self.assertGreater(self_map.item(), 0.95)


# ── §2 ──────────────────────────────────────────────────────────────────

class TestNeighborhoodRefinement(unittest.TestCase):
    def test_similar_patches_reuse(self):
        from internnav.vln_cache import neighborhood_refinement
        M, d = 196, 64
        embeds = torch.randn(M, d)
        matched_idx = torch.arange(M)
        valid = torch.ones(M, dtype=torch.bool)
        noisy = embeds + torch.randn(M, d) * 0.01
        _, m_vis = neighborhood_refinement(noisy, embeds, matched_idx, valid, 14, 14, tau_v=0.85)
        self.assertGreater(m_vis.float().mean().item(), 0.8)

    def test_dissimilar_patches_no_reuse(self):
        from internnav.vln_cache import neighborhood_refinement
        M, d = 196, 64
        embeds_a = torch.randn(M, d)
        embeds_b = torch.randn(M, d)
        matched_idx = torch.arange(M)
        valid = torch.ones(M, dtype=torch.bool)
        _, m_vis = neighborhood_refinement(embeds_a, embeds_b, matched_idx, valid, 14, 14, tau_v=0.85)
        self.assertLess(m_vis.float().mean().item(), 0.3)


# ── §3 ──────────────────────────────────────────────────────────────────

class TestSemanticRefreshGate(unittest.TestCase):
    def test_gate(self):
        from internnav.vln_cache import compute_semantic_refresh_gate
        scores = torch.tensor([0.1, 0.6, 0.3, 0.9])
        prev = torch.tensor([0.1, 0.5, 0.8, 0.2])
        m = compute_semantic_refresh_gate(scores, prev, tau_s=0.5, delta_s=0.3)
        # idx 1: score > 0.5 → refresh; idx 2: delta > 0.3 → refresh
        # idx 3: score > 0.5 AND delta > 0.3 → refresh
        self.assertTrue(m[1].item())
        self.assertTrue(m[2].item())
        self.assertTrue(m[3].item())
        self.assertFalse(m[0].item())


# ── §4 ──────────────────────────────────────────────────────────────────

class TestBuildReuseMask(unittest.TestCase):
    def test_mask(self):
        from internnav.vln_cache import build_reuse_mask
        m_vis = torch.tensor([True, True, False, True])
        m_sem = torch.tensor([False, True, False, False])
        reuse = build_reuse_mask(m_vis, m_sem)
        # reuse = m_vis AND NOT m_sem
        self.assertEqual(reuse.tolist(), [True, False, False, True])


# ── §5 ──────────────────────────────────────────────────────────────────

class TestLayerAdaptive(unittest.TestCase):
    def test_entropy_threshold(self):
        from internnav.vln_cache import layer_adaptive_threshold
        tau_low = layer_adaptive_threshold(torch.tensor(0.5), tau_base=0.85)
        tau_high = layer_adaptive_threshold(torch.tensor(5.0), tau_base=0.85)
        # Higher entropy → lower threshold (more aggressive reuse)
        self.assertGreater(tau_low, tau_high)


# ── §6 ──────────────────────────────────────────────────────────────────

class TestVLNCacheManager(unittest.TestCase):
    def _make_dummy_model(self):
        """Create a minimal dummy model with model.model.layers structure."""
        layer = torch.nn.Identity()
        inner = torch.nn.Module()
        inner.layers = torch.nn.ModuleList([layer, layer])
        model = torch.nn.Module()
        model.model = inner
        return model

    def test_lifecycle(self):
        from internnav.vln_cache import VLNCacheConfig, VLNCacheManager
        model = self._make_dummy_model()
        cfg = VLNCacheConfig(fx=240, fy=240, cx=239.5, cy=239.5, img_h=480, img_w=480, H_patches=14, W_patches=14)
        mgr = VLNCacheManager(model, cfg)

        # First frame
        depth1 = torch.ones(480, 480) * 2.0
        pos1 = torch.zeros(3)
        quat1 = torch.tensor([1.0, 0, 0, 0])
        embeds1 = torch.randn(196, 64)
        mgr.on_new_frame(depth1, pos1, quat1, embeds1)
        self.assertIsNone(mgr._reuse_mask)  # First frame: no reuse
        self.assertIsNotNone(mgr._curr)  # _curr not committed yet

        # Simulate commit (normally done by last layer hook)
        mgr._commit_frame()
        self.assertIsNone(mgr._curr)
        self.assertIsNotNone(mgr._prev)

        # Second frame with identical pose
        embeds2 = embeds1 + torch.randn(196, 64) * 0.01
        mgr.on_new_frame(depth1, pos1, quat1, embeds2)
        self.assertIsNotNone(mgr._reuse_mask)
        reuse_ratio = mgr._reuse_mask.float().mean().item()
        self.assertGreater(reuse_ratio, 0.5)
        self.assertEqual(mgr.stats["n_frames"], 1)

        # Reset
        mgr.reset()
        self.assertIsNone(mgr._prev)
        self.assertIsNone(mgr._curr)
        self.assertIsNone(mgr._reuse_mask)

    def test_safety_cap(self):
        from internnav.vln_cache import VLNCacheConfig, VLNCacheManager
        model = self._make_dummy_model()
        cfg = VLNCacheConfig(
            fx=240, fy=240, cx=239.5, cy=239.5,
            img_h=480, img_w=480, H_patches=14, W_patches=14, max_reuse_ratio=0.5,
        )
        mgr = VLNCacheManager(model, cfg)

        depth = torch.ones(480, 480) * 2.0
        pos = torch.zeros(3)
        quat = torch.tensor([1.0, 0, 0, 0])
        embeds = torch.randn(196, 64)

        mgr.on_new_frame(depth, pos, quat, embeds)
        mgr._commit_frame()

        mgr.on_new_frame(depth, pos, quat, embeds + torch.randn(196, 64) * 0.01)
        if mgr._reuse_mask is not None:
            ratio = mgr._reuse_mask.float().mean().item()
            self.assertLessEqual(ratio, 0.5 + 0.05)  # Allow small float margin


# ── §7 ──────────────────────────────────────────────────────────────────

class TestHookSpliceLogic(unittest.TestCase):
    """Test the layer hook with a mock DynamicCache."""

    def test_hook_stores_and_splices(self):
        from internnav.vln_cache import VLNCacheConfig, VLNCacheManager

        # Mock decoder layer that accepts past_key_value kwarg.
        class MockDecoderLayer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(8, 8)
            def forward(self, x, past_key_value=None, **kwargs):
                return (self.linear(x),)

        layer0 = MockDecoderLayer()
        layer1 = MockDecoderLayer()
        inner = torch.nn.Module()
        inner.layers = torch.nn.ModuleList([layer0, layer1])
        model = torch.nn.Module()
        model.model = inner

        cfg = VLNCacheConfig(fx=240, fy=240, cx=239.5, cy=239.5, img_h=480, img_w=480, H_patches=14, W_patches=14)
        mgr = VLNCacheManager(model, cfg)
        mgr._register_hooks()

        # Mock DynamicCache
        class MockCache:
            def __init__(self):
                S = 20  # seq len
                self.key_cache = [torch.randn(1, 4, S, 8), torch.randn(1, 4, S, 8)]
                self.value_cache = [torch.randn(1, 4, S, 8), torch.randn(1, 4, S, 8)]

        mock_cache = MockCache()
        mgr._vision_token_offset = 5
        mgr._n_vision_tokens = 10

        # Frame 1: on_new_frame then simulate layer forwards
        depth = torch.ones(480, 480) * 2.0
        pos = torch.zeros(3)
        quat = torch.tensor([1.0, 0, 0, 0])
        embeds = torch.randn(196, 64)
        mgr.on_new_frame(depth, pos, quat, embeds)

        # Simulate layer 0 forward (pass cache as kwarg)
        x = torch.randn(1, 8)
        _ = layer0(x, past_key_value=mock_cache)
        # Simulate layer 1 forward (last layer -> commit)
        _ = layer1(x, past_key_value=mock_cache)

        # After both layers: _curr should be committed
        self.assertIsNone(mgr._curr)
        self.assertIsNotNone(mgr._prev)
        # KV should be stored
        self.assertIn(0, mgr._prev.kv_per_layer)
        self.assertIn(1, mgr._prev.kv_per_layer)

        # Frame 2: same pose (expect reuse)
        mock_cache2 = MockCache()
        embeds2 = embeds + torch.randn(196, 64) * 0.01
        mgr.on_new_frame(depth, pos, quat, embeds2)
        self.assertIsNotNone(mgr._reuse_mask)

        # Simulate layer forwards again
        _ = layer0(x, past_key_value=mock_cache2)
        _ = layer1(x, past_key_value=mock_cache2)

        self.assertIsNone(mgr._curr)
        self.assertIsNone(mgr._reuse_mask)  # cleared by last layer

        mgr.remove_hooks()


# ── §8 ──────────────────────────────────────────────────────────────────

class TestVLNCacheHookIntegration(unittest.TestCase):
    """Test VLNCacheHook with a mock evaluator-like structure."""

    def test_from_evaluator_mock(self):
        """Test that VLNCacheHook can be created from mock evaluator attrs."""
        from internnav.vln_cache.evaluator_integration import VLNCacheHook
        from internnav.vln_cache.vln_cache_wrapper import VLNCacheConfig

        # Build minimal mock model
        layer = torch.nn.Identity()
        inner = torch.nn.Module()
        inner.layers = torch.nn.ModuleList([layer])
        model = torch.nn.Module()
        model.model = inner
        model.visual = torch.nn.Identity()

        device = torch.device("cpu")
        cfg = VLNCacheConfig(fx=240, fy=240, cx=239.5, cy=239.5, img_h=480, img_w=480, H_patches=14, W_patches=14)

        hook = VLNCacheHook(model, cfg, device)
        self.assertIsNotNone(hook.manager)
        self.assertEqual(hook.manager.n_layers, 1)

        hook.reset_episode()
        stats = hook.get_episode_stats()
        self.assertEqual(stats["n_frames"], 0)

        hook.remove_all_hooks()


# ── §9 ──────────────────────────────────────────────────────────────────

class TestPerformanceBenchmark(unittest.TestCase):
    def test_full_pipeline_speed(self):
        """End-to-end benchmark: on_new_frame should be fast."""
        from internnav.vln_cache import VLNCacheConfig, VLNCacheManager

        layer = torch.nn.Identity()
        inner = torch.nn.Module()
        inner.layers = torch.nn.ModuleList([layer for _ in range(28)])
        model = torch.nn.Module()
        model.model = inner

        cfg = VLNCacheConfig(
            fx=240, fy=240, cx=239.5, cy=239.5,
            img_h=480, img_w=480, H_patches=14, W_patches=14,
        )
        mgr = VLNCacheManager(model, cfg)

        N_FRAMES = 20
        depth = torch.ones(480, 480) * 2.0
        pos_base = torch.zeros(3)
        quat = torch.tensor([1.0, 0, 0, 0])

        times = []
        for i in range(N_FRAMES):
            pos = pos_base + torch.randn(3) * 0.05
            embeds = torch.randn(196, 64)
            t0 = time.perf_counter()
            mgr.on_new_frame(depth, pos, quat, embeds)
            mgr._commit_frame()  # Simulate hook commit
            dt = time.perf_counter() - t0
            times.append(dt * 1000)

        avg_ms = np.mean(times[1:])  # skip first frame
        print(f"\n  [Bench] on_new_frame avg: {avg_ms:.2f} ms  "
              f"(mean reuse: {mgr.stats['mean_reuse_ratio']:.1%})")
        # Should be under 50ms per frame
        self.assertLess(avg_ms, 50.0)


# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    unittest.main(verbosity=2)
