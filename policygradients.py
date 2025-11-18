import torch
import torch.nn as nn
from torch.distributions import Categorical
import gymnasium as gym
import matplotlib.pyplot as plt
import time
import numpy as np

env_train = gym.make("CartPole-v1")
env_test = gym.make("CartPole-v1", render_mode="human")

device = "cpu"


class QPolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.network(x)

total_episodes = 10000
learning_rate = 0.001
discount_factor = 0.99

policy_network = QPolicyNetwork().to(device)
optimizer = torch.optim.Adam(policy_network.parameters(), lr = learning_rate)

losses = []

for episode in range(total_episodes):
    saved_log_probs = []
    rewards = []

    state, info = env_train.reset()
    done = False
    truncated = False

    while not (done or truncated):
        done = True

        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
        logits = policy_network(state_tensor)
        dist = Categorical(logits=logits)
        action = dist.sample()

        log_prob = dist.log_prob(action)
        saved_log_probs.append(log_prob)

        state, reward, done, truncated, info = env_train.step(int(action.item()))

        rewards.append(reward)

    returns = []
    current_return = 0

    for r in reversed(rewards):
        current_return = r + discount_factor * current_return
        returns.insert(0, current_return)

    returns_tensor = torch.tensor(returns).to(device)
    returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-9)
    log_probs_tensor = torch.stack(saved_log_probs).squeeze().to(device)

    loss = (-log_probs_tensor * returns_tensor).sum()
    losses.append(loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(episode)

plt.plot(losses)
plt.xlabel('Episode')
plt.ylabel('Average Loss (REINFORCE)')
plt.title('Policy Gradients: Average Loss vs. Episode')
plt.grid(True)
plt.savefig("training_results.jpg")

for i in range(5):
    state, info = env_test.reset()
    done = False
    truncated = False

    total_reward = 0

    while not (done or truncated):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = policy_network(state_tensor)

        dist = Categorical(logits=logits)

        action = dist.sample()
        next_state, reward, done, truncated, info = env_test.step(int(action.item()))

        state = next_state
        total_reward += reward

        print(i)

        time.sleep(0.02)
    

env_train.close()
env_test.close()
