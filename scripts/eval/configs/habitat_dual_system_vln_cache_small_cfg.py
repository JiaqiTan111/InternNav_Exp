"""
Small-scale VLN-Cache config — 5 episodes, minimal analysis overhead.

Usage:
    python scripts/eval/eval.py --config scripts/eval/configs/habitat_dual_system_vln_cache_small_cfg.py
"""

from internnav.configs.agent import AgentCfg
from internnav.configs.evaluator import EnvCfg, EvalCfg

eval_cfg = EvalCfg(
    agent=AgentCfg(
        model_name='internvla_n1',
        model_settings={
            "mode": "dual_system",
            "model_path": "checkpoints/InternVLA-N1",
            "num_history": 8,
            "resize_w": 384,
            "resize_h": 384,
            "max_new_tokens": 1024,
        },
    ),
    env=EnvCfg(
        env_type='habitat',
        env_settings={
            'config_path': 'scripts/eval/configs/vln_r2r.yaml',
        },
    ),
    eval_type='habitat_vln',
    eval_settings={
        "output_path": "./logs/habitat/test_dual_system_vln_cache_small",
        "save_video": False,
        "epoch": 0,
        "max_steps_per_episode": 500,
        # Lightweight analysis (no attention probing)
        "analysis_enable_step_log": True,
        "analysis_collect_s2_log": True,
        "analysis_alignment_enable": True,
        "analysis_dump_dir": "./vis_analyze_new/data/raw_vln_cache_small",
        "analysis_patch_size": 28,
        "analysis_similarity_resize": 392,
        "analysis_similarity_tau": 0.7,
        "analysis_probe_attn": False,
        "analysis_attn_implementation": "flash_attention_2",
        "analysis_probe_every_n_steps": 1,
        "analysis_save_patch_tokens": False,
        "analysis_save_rgbd": False,
        "max_episodes": 100,
        # distributed
        "port": "2333",
        "dist_url": "env://",
        # ── VLN-Cache ──
        "vln_cache_enabled": True,
    },
)
