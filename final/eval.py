import argparse
import os
import sys
import time
import torch
import numpy as np

from .loader import load_scene, ensure_sys_path, default_model_path, default_save_dir


def parse_args():
    p = argparse.ArgumentParser(description="Unified evaluator for Parking SAC across scenes")
    p.add_argument("--scene", type=int, default=1, choices=[1, 2, 3, 4], help="Scene index to evaluate (1-4)")
    p.add_argument("--model_path", type=str, default=None, help="Path to model .pth. If not set, auto-pick latest in sceneX/models_sac")
    p.add_argument("--episodes", type=int, default=5, help="Number of episodes to evaluate")
    p.add_argument("--no_render", action="store_true", help="Disable visualization")
    p.add_argument("--save_dir", type=str, default=None, help="Directory to save frames/GIFs. Defaults to sceneX/eval_results")
    return p.parse_args()


def evaluate(scene: int, model_path: str | None, episodes: int, no_render: bool, save_dir: str | None):
    ensure_sys_path()
    ParkingEnv, SACAgent, ParkingVisualizer = load_scene(scene)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Environment and agent
    env = ParkingEnv()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    agent = SACAgent(obs_dim, act_dim, device)

    # Model path resolution
    resolved_model = model_path or default_model_path(scene)
    if not resolved_model or not os.path.exists(resolved_model):
        print(f"❌ Model not found. Provided: {model_path} | Auto: {resolved_model}")
        return 1
    print(f"Loading model: {resolved_model}")
    try:
        state_dict = torch.load(resolved_model, map_location=device)
        agent.actor.load_state_dict(state_dict)
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return 1

    # Visualization
    visualizer = None
    out_dir = save_dir or default_save_dir(scene)
    if not no_render:
        visualizer = ParkingVisualizer()

    # Episodes
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0.0
        steps = 0
        traj_points: list[tuple[float, float]] = []

        print(f"\n--- Scene {scene} | Episode {ep + 1}/{episodes} ---")
        slot = env.scene.parking_spaces[env.scene.target_idx]
        goal_pose = (slot["center"][0], slot["center"][1], slot["theta"]) 
        sx, sy, stheta, *_ = env.vehicle.get_state()
        start_pose = (sx, sy, stheta)

        if visualizer:
            visualizer.draw_scene(env.scene, start_pose=start_pose, goal_pose=goal_pose)
            visualizer.draw_vehicle(env.scene, sx, sy, stheta, delta=0.0)
            visualizer.capture_frame()

        while not done:
            # Deterministic action
            action = agent.select_action(obs, evaluate=True)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            steps += 1
            obs = next_obs

            vx, vy, vtheta, *_ = env.vehicle.get_state()
            traj_points.append((vx, vy))

            if visualizer:
                visualizer.draw_vehicle(
                    env.scene,
                    vx, vy, vtheta,
                    delta=env.vehicle.delta,
                    collided=info.get('is_collided', False),
                    success=info.get('is_success', False),
                )
                visualizer.draw_trajectory(traj_points)

                info_str = (
                    f"Step: {steps}\n"
                    f"Reward: {reward:.2f}\n"
                    f"Total Reward: {episode_reward:.2f}\n"
                    f"Velocity: {env.vehicle.v:.2f} m/s\n"
                    f"Steering: {np.degrees(env.vehicle.delta):.1f}°"
                )
                visualizer.draw_info(info_str)
                visualizer.update(pause=0.01)
                visualizer.capture_frame()

        result = "Success" if info.get('is_success', False) else "Failure"
        print(f"Episode finished. Result: {result} | Steps: {steps} | Total Reward: {episode_reward:.2f}")
        if visualizer:
            os.makedirs(out_dir, exist_ok=True)
            png = os.path.join(out_dir, f"scene{scene}_ep_{ep+1}_{result}.png")
            gif = os.path.join(out_dir, f"scene{scene}_ep_{ep+1}_{result}.gif")
            visualizer.save(png)
            visualizer.save_gif(gif)
            time.sleep(0.5)

    if visualizer:
        visualizer.show()
    return 0


if __name__ == "__main__":
    args = parse_args()
    sys.exit(evaluate(args.scene, args.model_path, args.episodes, args.no_render, args.save_dir))
