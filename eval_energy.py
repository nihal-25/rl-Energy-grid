import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from energy_env import EnergyEnv
import matplotlib.pyplot as plt

# Load environment + normalization + model
env = DummyVecEnv([lambda: EnergyEnv("germany_energy.csv")])
env = VecNormalize.load("vecnormalize.pkl", env)
model = PPO.load("ppo_energy", env=env)

obs = env.reset()
rewards = []

for _ in range(500):
    action, _ = model.predict(obs, deterministic=True)

    # VecEnv step returns: obs, reward, done, info
    obs, reward, done, info = env.step(action)

    rewards.append(reward[0])  # reward is a vector, take first element

    if done:
        obs = env.reset()

# Plot rewards
plt.figure(figsize=(8,4))
plt.plot(rewards)
plt.title("Reward per Step (Evaluation)")
plt.xlabel("Step")
plt.ylabel("Reward")
plt.grid(True)
plt.savefig("rewards.png")
print("Saved rewards.png")
