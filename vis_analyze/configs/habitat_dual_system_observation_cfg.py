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
        "output_path": "./vis_analyze/data/eval_output",
        "save_video": False,
        "epoch": 0,
        "max_episodes": 100,
        "max_steps_per_episode": 500,
        "port": "2333",
        "dist_url": "env://",
        "analysis_enable_step_log": True,
        "analysis_dump_dir": "./vis_analyze/data/raw",
        "analysis_patch_size": 28,
        "analysis_similarity_resize": 392,
        "analysis_alignment_enable": True,
        "analysis_collect_s2_log": True,
    },
)
