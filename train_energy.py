import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from energy_env import EnergyEnv

def make_env():
    return EnergyEnv("germany_energy.csv")

if __name__ == "__main__":
    env = DummyVecEnv([make_env])
    env = VecNormalize(env, norm_obs=True, norm_reward=False)

    model = PPO("MlpPolicy", env, verbose=1,
                n_steps=1024, batch_size=64, learning_rate=3e-4,
                gamma=0.99, ent_coef=0.01)

    model.learn(total_timesteps=200000)
    model.save("ppo_energy")
    env.save("vecnormalize.pkl")
    print("Training done and model saved.")
