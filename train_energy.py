import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from energy_env import EnergyEnv

def make_env():
    return EnergyEnv("germany_energy.csv")

if __name__ == "__main__":
    env = DummyVecEnv([make_env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True)

    model = PPO(
    "MlpPolicy", 
    env, 
    verbose=1,
    n_steps=2048,          # increased for stability
    batch_size=128,        # bigger batch
    learning_rate=1e-4,    # smaller LR for stability
    gamma=0.995,           # longer horizon
    ent_coef=0.005,        # mild exploration
)

    model.learn(total_timesteps=2500000)

    model.save("ppo_energy")
    env.save("vecnormalize.pkl")
    print("Training done and model saved.")
