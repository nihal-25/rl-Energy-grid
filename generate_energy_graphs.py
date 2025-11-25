import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from energy_env import EnergyEnv

# -----------------------------
# Load environment + model
# -----------------------------
env = DummyVecEnv([lambda: EnergyEnv("germany_energy.csv")])
env = VecNormalize.load("vecnormalize.pkl", env)
model = PPO.load("ppo_energy", env=env)

# -----------------------------
# Run one full episode
# -----------------------------
obs = env.reset()

steps = 500  # adjust if needed

soc_list = []
dispatch_list = []
supply_list = []
demand_list = []
solar_list = []
wind_list = []

for _ in range(steps):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)

    # Extract raw values from normalized obs
    load = obs[0][0]
    solar = obs[0][1]
    wind = obs[0][2]
    soc = obs[0][3]
    time = obs[0][4]

    dispatch = float(action[0])

    # Store logs
    soc_list.append(soc)
    dispatch_list.append(dispatch)
    demand_list.append(load)
    solar_list.append(solar)
    wind_list.append(wind)
    supply_list.append(solar + wind + max(dispatch, 0))  # approx supply

    if done:
        break

# -----------------------------
# 1. Battery State of Charge
# -----------------------------
plt.figure(figsize=(10,4))
plt.plot(soc_list)
plt.title("Battery State of Charge over Time")
plt.xlabel("Step")
plt.ylabel("SOC (GW)")
plt.grid(True)
plt.savefig("soc.png")
print("Saved soc.png")

# -----------------------------
# 2. Dispatch (Action) Graph
# -----------------------------
plt.figure(figsize=(10,4))
plt.plot(dispatch_list)
plt.title("Battery Dispatch Action over Time")
plt.xlabel("Step")
plt.ylabel("Dispatch (scaled)")
plt.grid(True)
plt.savefig("dispatch.png")
print("Saved dispatch.png")

# -----------------------------
# 3. Supply vs Demand
# -----------------------------
plt.figure(figsize=(10,4))
plt.plot(demand_list, label="Demand")
plt.plot(supply_list, label="Supply (Solar + Wind + Battery Discharge)")
plt.title("Supply vs Demand")
plt.xlabel("Step")
plt.ylabel("Power (GW)")
plt.legend()
plt.grid(True)
plt.savefig("supply_vs_demand.png")
print("Saved supply_vs_demand.png")

# -----------------------------
# 4. Solar / Wind / Load Comparison
# -----------------------------
plt.figure(figsize=(10,4))
plt.plot(demand_list, label="Demand")
plt.plot(solar_list, label="Solar")
plt.plot(wind_list, label="Wind")
plt.title("Solar, Wind, and Load Comparison")
plt.xlabel("Step")
plt.ylabel("Power (GW)")
plt.legend()
plt.grid(True)
plt.savefig("renewables_vs_load.png")
print("Saved renewables_vs_load.png")

# -----------------------------
# 5. Before vs After RL (Typical Day)
# -----------------------------
df = pd.read_csv("germany_energy.csv")
day_df = df.iloc[:24]   # first 24 hours

plt.figure(figsize=(10,4))
plt.plot(day_df["load"], label="Demand (No Control)")
plt.plot(day_df["solar"] + day_df["wind"], label="Renewable Supply (No Control)")
plt.title("Before RL Control (Typical Day)")
plt.xlabel("Hour")
plt.ylabel("Power (GW)")
plt.legend()
plt.grid(True)
plt.savefig("before_rl.png")
print("Saved before_rl.png")

plt.figure(figsize=(10,4))
plt.plot(demand_list[:24], label="Demand (After RL)")
plt.plot(supply_list[:24], label="Supply After RL")
plt.title("After RL Control (Typical Day)")
plt.xlabel("Hour")
plt.ylabel("Power (GW)")
plt.legend()
plt.grid(True)
plt.savefig("after_rl.png")
print("Saved after_rl.png")
