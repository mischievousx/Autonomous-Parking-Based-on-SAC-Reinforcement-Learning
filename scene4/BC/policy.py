import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_STD_MAX = 2
LOG_STD_MIN = -20

class Actor(nn.Module):
    """
    SAC Actor Network (Gaussian Policy)
    Designed for Behavior Cloning (BC) pre-training and subsequent SAC fine-tuning.
    """
    def __init__(self, obs_dim=8, act_dim=2, hidden_dim=256, action_limit=1.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.mean_layer = nn.Linear(hidden_dim, act_dim)
        self.log_std_layer = nn.Linear(hidden_dim, act_dim)
        
        self.action_limit = action_limit

    def forward(self, obs):
        """
        Forward pass for BC training (Deterministic).
        Returns the mean action, which is used to minimize MSE against expert actions.
        """
        x = self.net(obs)
        mean = self.mean_layer(x)
        # Tanh squashing to keep action within [-1, 1] * limit
        # Assuming expert actions are also within this range or normalized
        action = torch.tanh(mean) * self.action_limit
        return action

    def sample(self, obs):
        """
        Sample action for SAC training (Stochastic).
        Returns: action, log_prob, mean
        """
        x = self.net(obs)
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        normal = Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization trick
        y_t = torch.tanh(x_t)
        action = y_t * self.action_limit
        
        # Calculate log_prob
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound correction
        # log(1 - tanh(x)^2) + epsilon
        log_prob -= torch.log(self.action_limit * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        mean = torch.tanh(mean) * self.action_limit
        return action, log_prob, mean

# Alias for compatibility with existing BC scripts
BCPolicy = Actor

class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=256):
        super().__init__()
        # Q1 architecture
        self.q1_net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        # Q2 architecture
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