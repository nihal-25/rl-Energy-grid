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
    obs, reward, done, info = env.step(action)   # <-- only 4 values
    rewards.append(reward[0])
    if done:
        obs = env.reset()

# Plot rewards
plt.plot(rewards)
plt.title("Reward per step")
plt.xlabel("Step")
plt.ylabel("Reward")
plt.savefig("rewards.png")
print("Saved rewards.png")
