import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class EnergyEnv(gym.Env):
    def __init__(self, data_path="germany_energy.csv", battery_capacity=20.0, max_dispatch=10.0):
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

        # observation: [load, solar, wind, state of charge, time of day]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([100, 50, 50, self.battery_capacity, 1], dtype=np.float32)
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self.soc = self.battery_capacity / 2.0
        return self._get_obs(), {}

    def step(self, action):
        dispatch = float(action[0]) * self.max_dispatch

        net_gen = self.solar[self.t] + self.wind[self.t]
        demand = self.load[self.t]

        # ---------- Battery Charge / Discharge Logic ----------
        if dispatch > 0:
            # Discharging battery to grid
            discharge = min(dispatch, self.soc)
            self.soc -= discharge
            supplied = net_gen + discharge
        else:
            # Charging battery (from grid or renewables)
            charge = min(-dispatch, self.battery_capacity - self.soc)
            self.soc += charge
            supplied = net_gen - charge
        # -------------------------------------------------------

        # Supply-Demand Balance
        balance = supplied - demand

        # ---------- Reward Function ----------
        relative_mismatch = abs(balance) / max(demand, 1e-3)

        reward_stability = -5 * relative_mismatch
        blackout_penalty = -20 * max(0, demand - supplied) / max(demand, 1e-3)
        cycle_penalty = -1.0 * (abs(dispatch) / self.max_dispatch)
        soc_penalty = -2.0 * (abs(self.soc / self.battery_capacity - 0.5))

        reward = reward_stability + blackout_penalty + cycle_penalty + soc_penalty
        # -------------------------------------

        self.t += 1
        terminated = self.t >= self.n_steps - 1
        truncated = False

        return self._get_obs(), reward, terminated, truncated, {}

    def _get_obs(self):
        time_of_day = (self.t % 24) / 24.0
        return np.array([
            self.load[self.t],
            self.solar[self.t],
            self.wind[self.t],
            self.soc,
            time_of_day
        ], dtype=np.float32)
