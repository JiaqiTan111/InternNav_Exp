from internnav.configs.agent import AgentCfg
from internnav.configs.evaluator import EnvCfg, EvalCfg

eval_cfg = EvalCfg(
    agent=AgentCfg(
        model_name='internvla_n1',
        model_settings={
            "mode": "system2",  # inference mode: dual_system or system2
            "model_path": "checkpoints/InternVLA-N1-System2",  # path to model checkpoint
            "num_history": 8,
            "resize_w": 384,  # image resize width
            "resize_h": 384,  # image resize height
            "max_new_tokens": 1024,  # maximum number of tokens for generation
        },
    ),
    env=EnvCfg(
        env_type='habitat',
        env_settings={
            # habitat sim specifications - agent, sensors, tasks, measures etc. are defined in the habitat config file
            'config_path': 'scripts/eval/configs/vln_r2r.yaml',
        },
    ),
    eval_type='habitat_vln',
    eval_settings={
        # all current parse args
        "output_path": "./logs/habitat/test_s2",  # output directory for logs/results
        "save_video": False,  # whether to save videos
        "epoch": 0,  # epoch number for logging
        "max_steps_per_episode": 500,  # maximum steps per episode
        # vis_analyze_new logging / hooks
        "analysis_enable_step_log": True,
        "analysis_collect_s2_log": True,
        "analysis_alignment_enable": True,
        "analysis_dump_dir": "./vis_analyze_new/data/raw",
        "analysis_patch_size": 28,
        "analysis_similarity_resize": 392,
        "analysis_similarity_tau": 0.7,
        "analysis_probe_attn": True,
        "analysis_attn_implementation": "eager",
        "analysis_probe_every_n_steps": 1,
        "analysis_save_patch_tokens": True,
        "analysis_save_rgbd": False,
        "max_episodes": 2,
        # distributed settings
        "port": "2333",  # communication port
        "dist_url": "env://",  # url for distributed setup
    },
)
