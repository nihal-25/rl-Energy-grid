# api.py
from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from energy_env import EnergyEnv

app = Flask(__name__, static_url_path="", static_folder=".")
CORS(app)  # harmless if same-origin; helpful if you ever open index.html elsewhere

# ---- Load environment + normalization + model ----
# IMPORTANT: This must match how you trained the model
env = DummyVecEnv([lambda: EnergyEnv("germany_energy.csv")])

# If you trained with VecNormalize, load it and attach to env
# (if you didn't train with VecNormalize, comment these two lines)
env = VecNormalize.load("vecnormalize.pkl", env)
env.training = False          # eval mode (no stats update)
env.norm_reward = False       # return raw rewards

model = PPO.load("ppo_energy", env=env)

# Keep a global observation for stepping
obs = env.reset()

@app.route("/")
def root():
    # Serve index.html from the same folder
    return send_from_directory(".", "index.html")

@app.route("/step", methods=["GET"])
def step():
    global obs

    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)

    if bool(done[0]):
        obs = env.reset()

    # unwrap observation (shape (1,5))
    normalized = obs[0]

    # UNNORMALIZE using stored stats
    mean = env.obs_rms.mean
    var = env.obs_rms.var
    real_obs = normalized * np.sqrt(var) + mean

    load, solar, wind, soc, time_of_day = real_obs.tolist()

    return jsonify({
        "load": float(load),
        "solar": float(solar),
        "wind": float(wind),
        "battery_soc": float(soc),
        "time_of_day": float(time_of_day),
        "action": float(action[0][0]),
        "reward": float(reward[0])
    })

if __name__ == "__main__":
    # pip install flask flask-cors
    app.run(debug=True)
