# bc/train_bc.py
import numpy as np
import torch
import torch.nn.functional as F
import pickle
import glob
import os
from torch.utils.data import TensorDataset, DataLoader
from BC.policy import BCPolicy, Critic

def load_demo_data(data_dir):
    pkl_files = glob.glob(os.path.join(data_dir, "*.pkl"))
    all_obs = []
    all_acts = []
    all_rews = []
    all_next_obs = []
    all_dones = []
    
    print(f"Found {len(pkl_files)} demonstration files in {data_dir}")
    
    for pkl_file in pkl_files:
        with open(pkl_file, 'rb') as f:
            episode_data = pickle.load(f)
            # episode_data is a list of (obs, action, reward, next_obs, terminated)
            for step in episode_data:
                obs, action, reward, next_obs, done = step
                all_obs.append(obs)
                all_acts.append(action)
                all_rews.append(reward)
                all_next_obs.append(next_obs)
                all_dones.append(float(done))
                
    if not all_obs:
        raise ValueError("No data found in the specified directory.")
        
    return (np.array(all_obs), np.array(all_acts), 
            np.array(all_rews), np.array(all_next_obs), np.array(all_dones))

# Load expert data
DATA_DIR = "demo_data_vertical"
obs_np, act_np, rew_np, next_obs_np, done_np = load_demo_data(DATA_DIR)

print(f"Loaded dataset: {obs_np.shape[0]} samples")

# Convert to tensors
obs = torch.tensor(obs_np, dtype=torch.float32)
act = torch.tensor(act_np, dtype=torch.float32)
rew = torch.tensor(rew_np, dtype=torch.float32).unsqueeze(1)
next_obs = torch.tensor(next_obs_np, dtype=torch.float32)
done = torch.tensor(done_np, dtype=torch.float32).unsqueeze(1)

dataset = TensorDataset(obs, act, rew, next_obs, done)
loader = DataLoader(dataset, batch_size=256, shuffle=True)

# Initialize Networks
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Actor (BC Policy)
policy = BCPolicy(obs_dim=8, act_dim=2, action_limit=1.0).to(device)
actor_opt = torch.optim.Adam(policy.parameters(), lr=3e-4)

# Critic (Q-Function)
critic = Critic(obs_dim=8, act_dim=2).to(device)
target_critic = Critic(obs_dim=8, act_dim=2).to(device)
target_critic.load_state_dict(critic.state_dict())
critic_opt = torch.optim.Adam(critic.parameters(), lr=3e-4)

# Hyperparameters
GAMMA = 0.99
TAU = 0.005
ALPHA = 0.2 # Entropy coefficient (fixed for simplicity in BC phase)

loss_fn_bc = torch.nn.MSELoss()

print("Starting BC + Critic Pre-training...")

for epoch in range(2000):
    actor_losses = []
    critic_losses = []
    
    for b_obs, b_act, b_rew, b_next_obs, b_done in loader:
        b_obs, b_act, b_rew, b_next_obs, b_done = b_obs.to(device), b_act.to(device), b_rew.to(device), b_next_obs.to(device), b_done.to(device)
        
        # ----------------------------
        # 1. Train Actor (Behavior Cloning)
        # ----------------------------
        pred_act = policy(b_obs) # Deterministic
        actor_loss = loss_fn_bc(pred_act, b_act)
        
        actor_opt.zero_grad()
        actor_loss.backward()
        actor_opt.step()
        actor_losses.append(actor_loss.item())
        
        # ----------------------------
        # 2. Train Critic (Offline SAC Update)
        # ----------------------------
        with torch.no_grad():
            # Target Action from Actor (using current policy to estimate next state value)
            # Note: In pure offline RL (CQL), we might be careful here. 
            # But for simple pre-training, using the BC-constrained policy is fine.
            next_actions, next_log_probs, _ = policy.sample(b_next_obs)
            target_q1, target_q2 = target_critic(b_next_obs, next_actions)
            target_q = torch.min(target_q1, target_q2) - ALPHA * next_log_probs
            target_q = b_rew + GAMMA * (1 - b_done) * target_q
            
        current_q1, current_q2 = critic(b_obs, b_act) # Evaluate EXPERT action
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        critic_opt.zero_grad()
        critic_loss.backward()
        critic_opt.step()
        critic_losses.append(critic_loss.item())
        
        # Target Update
        for param, target_param in zip(critic.parameters(), target_critic.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

    if epoch % 100 == 0:
        print(f"Epoch {epoch} | Actor Loss: {sum(actor_losses)/len(actor_losses):.6f} | Critic Loss: {sum(critic_losses)/len(critic_losses):.6f}")

# Save both models
torch.save(policy.state_dict(), "bc_policy.pth")
torch.save(critic.state_dict(), "bc_critic.pth")
print("Models saved: bc_policy.pth, bc_critic.pth")
