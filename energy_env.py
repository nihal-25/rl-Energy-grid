import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class EnergyEnv(gym.Env):
    def __init__(self, data_path="germany_energy.csv", battery_capacity=5.0, max_dispatch=2.0):
        super(EnergyEnv, self).__init__()
        self.data = pd.read_csv(data_path, index_col=0, parse_dates=True)
        self.load = self.data["load"].values
        self.solar = self.data["solar"].values
        self.wind = self.data["wind"].values
        self.n_steps = len(self.data)

        self.battery_capacity = battery_capacity
        self.max_dispatch = max_dispatch
        self.soc = None
        self.t = None

        # action: battery dispatch in [-max_dispatch, +max_dispatch]
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        # obs: [load, solar, wind, soc, time_of_day]
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(5,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self.soc = self.battery_capacity / 2.0
        return self._get_obs(), {}

    def step(self, action):
        dispatch = float(action[0]) * self.max_dispatch
        # Net renewable
        net_gen = self.solar[self.t] + self.wind[self.t]
        demand = self.load[self.t]

        # Apply battery
        self.soc = np.clip(self.soc + dispatch, 0, self.battery_capacity)
        supplied = net_gen
        if dispatch > 0:  # discharge
            supplied += min(dispatch, self.soc)
        else:  # charging
            supplied += dispatch  # negative means absorbing

        # Net balance
        balance = supplied - demand

        # Reward
        penalty_unserved = -10 * max(0, demand - supplied)
        penalty_over = -5 * max(0, supplied - demand)
        penalty_cycle = -0.1 * abs(dispatch)
        reward = penalty_unserved + penalty_over + penalty_cycle

        self.t += 1
        done = self.t >= self.n_steps - 1
        return self._get_obs(), reward, done, False, {}

    def _get_obs(self):
        time_of_day = (self.t % 24) / 24.0
        return np.array([
            self.load[self.t],
            self.solar[self.t],
            self.wind[self.t],
            self.soc,
            time_of_day
        ], dtype=np.float32)
