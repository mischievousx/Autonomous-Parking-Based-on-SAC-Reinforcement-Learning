import importlib
import os
import sys
from typing import Tuple, Optional


def _workspace_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def ensure_sys_path() -> None:
    """Ensure workspace root is on sys.path for dynamic imports like scene1.*"""
    root = _workspace_root()
    if root not in sys.path:
        sys.path.insert(0, root)


def scene_modules(scene: int):
    """Return module paths for control_new.envs, control_new.train, plot.visualization for a scene."""
    if scene not in (1, 2, 3, 4):
        raise ValueError("scene must be one of 1, 2, 3, 4")
    prefix = f"scene{scene}"
    return (
        f"{prefix}.control_new.envs",
        f"{prefix}.control_new.train",
        f"{prefix}.plot.visualization",
    )


def _ensure_scene_path(scene: int) -> str:
    """Ensure the selected scene directory is at the front of sys.path."""
    root = _workspace_root()
    scene_dir = os.path.join(root, f"scene{scene}")
    if scene_dir not in sys.path:
        # Prepend for highest precedence so `environment.*` resolves to this scene.
        sys.path.insert(0, scene_dir)
    return scene_dir


def load_scene(scene: int):
    """
    Dynamically import the environment, agent, and visualizer for a given scene.

    Returns: (ParkingEnvClass, SACAgentClass, ParkingVisualizerClass)
    """
    ensure_sys_path()
    env_mod, train_mod, vis_mod = scene_modules(scene)
    # Some scene modules use absolute imports like `from environment.vehicle import Vehicle`.
    # Add the scene directory itself to sys.path so `environment` resolves to sceneX/environment.
    _ensure_scene_path(scene)
    env = importlib.import_module(env_mod)
    train = importlib.import_module(train_mod)
    vis = importlib.import_module(vis_mod)
    return env.ParkingEnv, train.SACAgent, vis.ParkingVisualizer


def default_model_path(scene: int) -> Optional[str]:
    """Pick the latest SAC checkpoint in sceneX/models_sac, if any."""
    ensure_sys_path()
    root = _workspace_root()
    models_dir = os.path.join(root, f"scene{scene}", "models_sac")
    if not os.path.isdir(models_dir):
        return None
    try:
        candidates = [f for f in os.listdir(models_dir) if f.endswith(".pth") and f.startswith("sac_ep_")]
        if not candidates:
            return None
        def ep_num(name: str) -> int:
            # sac_ep_XXXX.pth
            try:
                return int(name.split("sac_ep_")[1].split(".")[0])
            except Exception:
                return -1
        latest = max(candidates, key=ep_num)
        return os.path.join(models_dir, latest)
    except Exception:
        return None


def default_save_dir(scene: int) -> str:
    """Default evaluation output directory under the scene folder."""
    root = _workspace_root()
    out = os.path.join(root, f"scene{scene}", "eval_results")
    os.makedirs(out, exist_ok=True)
    return out
