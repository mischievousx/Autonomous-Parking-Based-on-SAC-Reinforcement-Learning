import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pickle
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import deque
from torch.utils.tensorboard import SummaryWriter

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from control_new.envs import ParkingEnv
from BC.policy import Actor

# ==================================================
# 1. Define components (ReplayBuffer, Critic, SAC)
# ==================================================

class ReplayBuffer:
    def __init__(self, capacity, obs_dim, act_dim):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.acts = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rews = np.zeros((capacity, 1), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

    def add(self, obs, act, rew, next_obs, done):
        self.obs[self.ptr] = obs
        self.next_obs[self.ptr] = next_obs
        self.acts[self.ptr] = act
        self.rews[self.ptr] = rew
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.tensor(self.obs[idxs], dtype=torch.float32),
            torch.tensor(self.acts[idxs], dtype=torch.float32),
            torch.tensor(self.rews[idxs], dtype=torch.float32),
            torch.tensor(self.next_obs[idxs], dtype=torch.float32),
            torch.tensor(self.dones[idxs], dtype=torch.float32)
        )
    
    def load_demos(self, demo_dir, reward_scale=1.0):
        """Load expert data from demo_data and inject into buffer."""
        if not os.path.exists(demo_dir):
            print(f"⚠️ Demo directory {demo_dir} not found.")
            return 0
            
        pkl_files = glob.glob(os.path.join(demo_dir, "*.pkl"))
        count = 0
        print(f"Found {len(pkl_files)} demo files. Injecting data...")
        
        for pkl_file in pkl_files:
            try:
                with open(pkl_file, 'rb') as f:
                    episode_data = pickle.load(f)
                    for step in episode_data:
                        # (obs, action, reward, next_obs, terminated)
                        s, a, r, ns, done = step
                        
                        # Scale reward.
                        r_scaled = float(r) / reward_scale
                        
                        self.add(s, a, r_scaled, ns, float(done))
                        count += 1
            except Exception as e:
                print(f"Error loading {pkl_file}: {e}")
                
        print(f"✅ Injected {count} demo samples into ReplayBuffer.")
        return count

class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=256):
        super().__init__()
        # Q1
        self.q1_net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        # Q2
        self.q2_net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=1)
        return self.q1_net(x), self.q2_net(x)

class SACAgent:
    def __init__(self, obs_dim, act_dim, device, lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        
        # Automatic Entropy Tuning
        self.target_entropy = -float(act_dim)
        self.log_alpha = torch.tensor(np.log(alpha), requires_grad=True, device=device, dtype=torch.float32)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=lr)
        self.alpha = alpha
        
        # Networks
        self.actor = Actor(obs_dim, act_dim, action_limit=1.0).to(device)
        self.critic = Critic(obs_dim, act_dim).to(device)
        self.target_critic = Critic(obs_dim, act_dim).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr)
        
    def select_action(self, obs, evaluate=False):
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if evaluate:
                action = self.actor(obs_t) # Deterministic (forward)
            else:
                action, _, _ = self.actor.sample(obs_t) # Stochastic
        return action.cpu().numpy()[0]
    
    def update(self, buffer, batch_size, update_actor=True, bc_weight=0.0):
        if buffer.size < batch_size:
            return
            
        b_obs, b_acts, b_rews, b_next_obs, b_dones = [x.to(self.device) for x in buffer.sample(batch_size)]
        
        # Get current alpha
        alpha = self.log_alpha.exp().item()
        self.alpha = alpha
        
        # 1. Critic Update
        with torch.no_grad():
            next_actions, next_log_probs, _ = self.actor.sample(b_next_obs)
            target_q1, target_q2 = self.target_critic(b_next_obs, next_actions)
            target_q = torch.min(target_q1, target_q2) - alpha * next_log_probs
            target_q = b_rews + self.gamma * (1 - b_dones) * target_q
            
        current_q1, current_q2 = self.critic(b_obs, b_acts)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()
        
        # 2. Actor Update
        if update_actor:
            actions, log_probs, _ = self.actor.sample(b_obs)
            q1, q2 = self.critic(b_obs, actions)
            q = torch.min(q1, q2)
            
            # SAC Loss
            sac_loss = (alpha * log_probs - q).mean()
            
            # BC Regularization Loss
            if bc_weight > 0:
                bc_loss = F.mse_loss(actions, b_acts)
                actor_loss = sac_loss + bc_weight * bc_loss
            else:
                actor_loss = sac_loss
            
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()
            
            # 3. Alpha Update
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()
            
            # 4. Target Update
            for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def load_bc_weights(self, actor_path, critic_path=None):
        # Load Actor
        if os.path.exists(actor_path):
            print(f"Loading BC Actor weights from {actor_path}...")
            self.actor.load_state_dict(torch.load(actor_path, map_location=self.device))
            
            # CRITICAL FIX: Initialize log_std to low value (-3.0 -> std ~ 0.05)
            # BC training does not train log_std, so it's random/default.
            # High initial std causes random exploration, ignoring BC mean.
            with torch.no_grad():
                self.actor.log_std_layer.weight.fill_(0.0)
                self.actor.log_std_layer.bias.fill_(-3.0)
            print("✅ BC Actor weights loaded. (log_std initialized to -3.0 for stability)")
        else:
            print("⚠️ BC Actor weights not found.")
            
        # Load Critic (Optional)
        if critic_path and os.path.exists(critic_path):
            print(f"Loading BC Critic weights from {critic_path}...")
            self.critic.load_state_dict(torch.load(critic_path, map_location=self.device))
            self.target_critic.load_state_dict(self.critic.state_dict())
            print("✅ BC Critic weights loaded.")
        else:
            print("ℹ️ BC Critic weights not found (will use random init or warm-up).")

    def save(self, path):
        torch.save(self.actor.state_dict(), path)

# ==================================================
# 2. Training pipeline (reference train_utils.py)
# ==================================================

def train():
    # Configuration
    ENV_NAME = "ParkingEnv"
    MAX_EPISODES = 2000 # Count based on episodes.
    BATCH_SIZE = 256
    REWARD_SCALE = 1.0 
    WARMUP_STEPS = 5000
    LOG_DIR = "logs_sac"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize TensorBoard.
    writer = SummaryWriter(LOG_DIR)
    
    # Environment.
    env = ParkingEnv()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    
    # Agent
    agent = SACAgent(obs_dim, act_dim, device)
    
    # 1) Load BC pretrained models (Actor and Critic).
    agent.load_bc_weights("bc_policy.pth", "bc_critic.pth")
    
    # 2) Inject demo data.
    buffer = ReplayBuffer(capacity=1_000_000, obs_dim=obs_dim, act_dim=act_dim)
    demo_count = buffer.load_demos("demo_data_vertical", reward_scale=REWARD_SCALE)
    
    # 3) Critic warm-up (if demo data injected).
    # Always warm up; even with pretrained critic, adapt to buffer data.
    if demo_count > 0:
        print(f"\n>>> [Phase 1] Critic Warm-up ({WARMUP_STEPS} steps)...")
        for _ in tqdm(range(WARMUP_STEPS)):
            agent.update(buffer, BATCH_SIZE, update_actor=False)
        print(">>> Critic Warm-up finished.")
    else:
        print("\n>>> [Phase 1] Skipped Critic Warm-up (No demo data).")
        
    print("\n>>> [Phase 2] Start Online Training...")
    
    total_steps = 0
    
    # Statistics containers (moving averages over last 20 episodes).
    recent_rewards = deque(maxlen=20)
    recent_steps = deque(maxlen=20)
    recent_success = deque(maxlen=20)
    
    for episode in range(MAX_EPISODES):
        # BC weight decay.
        progress = episode / MAX_EPISODES
        if progress < 0.2:
            curr_bc_weight = 5.0
        else:
            decay_ratio = (progress - 0.2) / 0.8
            curr_bc_weight = max(0.1, 5.0 * (1.0 - decay_ratio))
            
        obs, _ = env.reset()
        episode_reward = 0
        current_ep_steps = 0
        
        while True:
            total_steps += 1
            
            # Action
            if total_steps < 1000 and demo_count == 0:
                action = env.action_space.sample() # Pure random (when no demos).
            else:
                action = agent.select_action(obs, evaluate=False)
                
            # Step
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Add to buffer
            buffer.add(obs, action, reward / REWARD_SCALE, next_obs, float(terminated))
            
            obs = next_obs
            episode_reward += reward
            current_ep_steps += 1
            
            # Update
            agent.update(buffer, BATCH_SIZE, update_actor=True, bc_weight=curr_bc_weight)
            
            if done:
                break
        
        # End of episode processing.
        is_success = info.get('is_success', False)
        
        # 1. Log this episode to TensorBoard.
        writer.add_scalar("Train/Episode_Reward", episode_reward, episode)
        writer.add_scalar("Train/Episode_Steps", current_ep_steps, episode)
        writer.add_scalar("Train/Success", int(is_success), episode)
        writer.add_scalar("Train/BC_Weight", curr_bc_weight, episode)
        
        # 2. Update statistics deques.
        recent_rewards.append(episode_reward)
        recent_steps.append(current_ep_steps)
        recent_success.append(1 if is_success else 0)

        # Compute moving averages.
        avg_reward = sum(recent_rewards) / len(recent_rewards)
        avg_steps = sum(recent_steps) / len(recent_steps)
        success_rate = sum(recent_success) / len(recent_success)

        # Log moving averages to TensorBoard (each episode).
        writer.add_scalar("Rollout/Avg_Reward", avg_reward, episode)
        writer.add_scalar("Rollout/Avg_Steps", avg_steps, episode)
        writer.add_scalar("Rollout/Success_Rate", success_rate, episode)
        
        # 3. Print stats every 20 episodes (console).
        if (episode + 1) % 20 == 0:
            print(f"Episode {episode + 1} | Total Steps: {total_steps} | "
                  f"Avg Reward: {avg_reward:.2f} | Avg Steps: {avg_steps:.1f} | "
                  f"Success Rate: {success_rate:.2%}")
            
        # 4. Save model every 100 episodes.
        if (episode + 1) % 100 == 0:
            agent.save(f"models_sac/sac_ep_{episode + 1}.pth")
            
    agent.save("models_sac/sac_final.pth")
    writer.close()
    print("Training finished.")

if __name__ == "__main__":
    # Ensure directories exist.
    os.makedirs("models_sac", exist_ok=True)
    os.makedirs("logs_sac", exist_ok=True)
    train()