# RULES

## Environment
- Always activate conda env first:
  - `conda activate mzh-habitataenv`

## GPU
- Use `cuda1` by default for this project.
- Recommended shell setting before running training/eval:
  - `export CUDA_VISIBLE_DEVICES=1`

## Run Template
- Example:
  - `conda activate mzh-habitataenv && export CUDA_VISIBLE_DEVICES=1 && python scripts/eval/eval.py --config scripts/eval/configs/h1_internvla_n1_async_cfg.py`

## Notes
- After setting `CUDA_VISIBLE_DEVICES=1`, code-side `cuda:0` maps to physical GPU 1.
- Keep this rule consistent for analysis logging and reproducible experiments.
