import argparse
import os
import sys
import torch
import numpy as np
import time
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from control_new.envs import ParkingEnv
from control_new.train import SACAgent
from plot.visualization import ParkingVisualizer

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SAC Agent for Parking")
    parser.add_argument("--model_path", type=str, default="./models_sac/sac_ep_1200.pth", help="Path to the model file (e.g., models_sac/sac_ep_1000.pth)")
    parser.add_argument("--episodes", type=int, default=6, help="Number of episodes to test")
    parser.add_argument("--no_render", action="store_true", help="Disable visualization")
    parser.add_argument("--save_dir", type=str, default="eval_results", help="Directory to save visualization results")
    return parser.parse_args()

def evaluate(args):
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Environment
    env = ParkingEnv()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Agent
    agent = SACAgent(obs_dim, act_dim, device)
    
    # Load Model
    if os.path.exists(args.model_path):
        print(f"Loading model from {args.model_path}...")
        try:
            state_dict = torch.load(args.model_path, map_location=device)
            agent.actor.load_state_dict(state_dict)
            print("✅ Model loaded successfully.")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return
    else:
        print(f"❌ Model file not found: {args.model_path}")
        return

    # Visualization
    visualizer = None
    if not args.no_render:
        visualizer = ParkingVisualizer()

    # Evaluation Loop
    for ep in range(args.episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        traj_points = []

        print(f"\n--- Episode {ep + 1}/{args.episodes} ---")
        slot = env.scene.parking_spaces[env.scene.target_idx]
        goal_pose = (slot["center"][0], slot["center"][1], slot["theta"])
        start = env.vehicle.get_state()
        start_pose = (start[0], start[1], start[2])
        if visualizer:
            visualizer.draw_scene(env.scene, start_pose=start_pose, goal_pose=goal_pose)
            # Capture initial state
            visualizer.draw_vehicle(env.scene, start[0], start[1], start[2], delta=0.0)
            visualizer.capture_frame()

        while not done:
            # Select Action (Deterministic)
            action = agent.select_action(obs, evaluate=True)
            
            # Step
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            steps += 1
            obs = next_obs

            # Record trajectory
            pose = env.vehicle.get_state()
            traj_points.append((pose[0], pose[1]))

            # Visualization
            if visualizer:
                visualizer.draw_vehicle(
                    env.scene, 
                    pose[0], pose[1], pose[2], 
                    delta=env.vehicle.delta, 
                    collided=info.get('is_collided', False),
                    success=info.get('is_success', False)
                )
                visualizer.draw_trajectory(traj_points)
                
                info_str = (f"Step: {steps}\n"
                            f"Reward: {reward:.2f}\n"
                            f"Total Reward: {episode_reward:.2f}\n"
                            f"Velocity: {env.vehicle.v:.2f} m/s\n"
                            f"Steering: {np.degrees(env.vehicle.delta):.1f}°")
                visualizer.draw_info(info_str)
                
                visualizer.update(pause=0.01)
                visualizer.capture_frame()

            if done:
                result = "Success" if info.get('is_success', False) else "Failure"
                print(f"Episode finished. Result: {result} | Steps: {steps} | Total Reward: {episode_reward:.2f}")
                if visualizer:
                    # Save result
                    os.makedirs(args.save_dir, exist_ok=True)
                    save_path = os.path.join(args.save_dir, f"ep_{ep+1}_{result}.png")
                    visualizer.save(save_path)
                    
                    gif_path = os.path.join(args.save_dir, f"ep_{ep+1}_{result}.gif")
                    visualizer.save_gif(gif_path)

                    # Keep the final frame for a moment
                    time.sleep(1.0)
                break

    if visualizer:
        visualizer.show()

if __name__ == "__main__":
    args = parse_args()
    evaluate(args)

