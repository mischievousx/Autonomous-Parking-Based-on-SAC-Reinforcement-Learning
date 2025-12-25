# Autonomous-Parking-Based-on-SAC-Reinforcement-Learning

This repository provides a unified evaluation entry for four traffic control scenes (`scene1`–`scene4`). It consolidates model loading, environment setup, agent execution, and visualization so you don't need to maintain per-scene scripts. The project includes reinforcement learning with Soft Actor-Critic (SAC) and supervised Behavior Cloning (BC) baselines, along with training logs and saved checkpoints for reproducibility.

## Project Introduction

- **Goal:** A consistent evaluation workflow across multiple scenes for intelligent control tasks.
- **Methods:** SAC (RL) and BC (supervised), with ready-to-evaluate checkpoints.
- **Scenes:** Each `sceneX/` contains environment definitions, training and evaluation scripts, models, and plotting utilities.
- **Final package:** The `final/` module orchestrates cross-scene loading, automatic model selection, and result saving.

### Repository Layout

- `sceneX/environment/`: Environment components like `scene.py`, `vehicle.py`.
- `sceneX/control_new/`: Training (`train.py`) and evaluation (`eval.py`) for SAC.
- `sceneX/models_sac/`: Saved SAC checkpoints named `sac_ep_*.pth`.
- `sceneX/BC/`: Behavior Cloning code and policies (`bc_policy.pth`, `bc_critic.pth`).
- `sceneX/plot/`: Visualization helpers.
- `final/`: Unified evaluator (`eval.py`) and dynamic scene loader (`loader.py`).

## Quick Start

Automatically pick the latest model (max episode in `sceneX/models_sac/sac_ep_*.pth`) and run evaluation:

```bash
python -m final.eval --scene 1 --episodes 6
python -m final.eval --scene 2 --episodes 6
python -m final.eval --scene 3 --episodes 6
python -m final.eval --scene 4 --episodes 6
```

Use a specific checkpoint and disable rendering:

```bash
python -m final.eval --scene 2 --model_path scene2/models_sac/sac_ep_3000.pth --no_render
```

Customize the output directory:

```bash
python -m final.eval --scene 3 --save_dir out/scene3_eval
```

## How It Works

- The dynamic loader in [final/loader.py](final/loader.py) imports the environment, Agent, and visualization modules for a given `--scene`.
- If a scene uses absolute imports like `from environment.*`, the loader adds that scene directory to `sys.path` so modules resolve correctly.
- When `--model_path` is not provided, the evaluator selects the checkpoint with the largest episode number from `sceneX/models_sac` (`sac_ep_*.pth`).
- Evaluation outputs (PNG and GIF) are saved under `sceneX/eval_results` by default.

## Notes

- Run commands from the workspace root so `sceneX.*` imports resolve properly.

- The final tool does not modify original scene code; it only coordinates cross-scene loading and evaluation.
