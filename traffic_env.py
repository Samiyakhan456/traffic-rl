"""
traffic_env.py
--------------
A 4-way intersection traffic signal environment modelled after
Indian urban conditions (mixed vehicle types, asymmetric flow).

State  : queue lengths on all 4 lanes + current phase + time in phase
Action : 0 = keep current phase, 1 = switch to next phase
Reward : negative total queue length (minimise waiting vehicles)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


# ── Vehicle mix weights (Indian roads: more 2-wheelers and autos) ──
VEHICLE_TYPES = {
    "two_wheeler": 0.45,
    "auto":        0.20,
    "car":         0.25,
    "bus_truck":   0.10,
}

# Arrival rates per lane per step (vehicles/step) — peak hour
ARRIVAL_RATES = {
    "north": 0.6,
    "south": 0.5,
    "east":  0.7,   # main road — heavier
    "west":  0.7,
}

LANES       = ["north", "south", "east", "west"]
PHASES      = [("north", "south"), ("east", "west")]   # which lanes get green
MIN_GREEN   = 5    # steps before a switch is allowed
MAX_GREEN   = 30   # steps before a forced switch
MAX_QUEUE   = 30   # cap queue display (not hard cap)
MAX_STEPS   = 500  # episode length


class TrafficEnv(gym.Env):
    """
    Gymnasium-compatible single-intersection traffic environment.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, arrival_rates=None, render_mode=None):
        super().__init__()

        self.arrival_rates = arrival_rates or ARRIVAL_RATES
        self.render_mode   = render_mode

        # Observation: [q_N, q_S, q_E, q_W, phase_idx, steps_in_phase]
        self.observation_space = spaces.Box(
            low   = np.zeros(6, dtype=np.float32),
            high  = np.array([MAX_QUEUE]*4 + [1, MAX_GREEN], dtype=np.float32),
            dtype = np.float32
        )

        # Action: 0 = keep phase, 1 = switch
        self.action_space = spaces.Discrete(2)

        self.reset()

    # ── Internal helpers ──────────────────────────────────────────────

    def _arrivals(self):
        """Stochastic Poisson arrivals on each lane."""
        return {lane: np.random.poisson(rate)
                for lane, rate in self.arrival_rates.items()}

    def _departures(self, green_lanes):
        """
        Vehicles that clear the intersection.
        Green lanes clear up to 2 vehicles/step; red lanes clear 0.
        """
        dep = {}
        for lane in LANES:
            if lane in green_lanes:
                dep[lane] = min(self.queues[lane], 2)
            else:
                dep[lane] = 0
        return dep

    def _get_obs(self):
        return np.array(
            [min(self.queues[l], MAX_QUEUE) for l in LANES]
            + [float(self.phase_idx), float(self.steps_in_phase)],
            dtype=np.float32
        )

    def _total_wait(self):
        return sum(self.queues.values())

    # ── Gymnasium API ─────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.queues         = {lane: 0 for lane in LANES}
        self.phase_idx      = 0
        self.steps_in_phase = 0
        self.step_count     = 0
        self.total_vehicles_cleared = 0
        return self._get_obs(), {}

    def step(self, action):
        # ── Enforce minimum green time ──
        can_switch = self.steps_in_phase >= MIN_GREEN
        force_switch = self.steps_in_phase >= MAX_GREEN

        if force_switch or (action == 1 and can_switch):
            self.phase_idx      = 1 - self.phase_idx
            self.steps_in_phase = 0
        else:
            self.steps_in_phase += 1

        green_lanes = PHASES[self.phase_idx]

        # ── Update queues ──
        arrivals   = self._arrivals()
        departures = self._departures(green_lanes)

        for lane in LANES:
            self.queues[lane] = max(0, self.queues[lane]
                                    + arrivals[lane]
                                    - departures[lane])
            self.total_vehicles_cleared += departures[lane]

        # ── Reward: penalise total queue, bonus for clearing heavy lanes ──
        queue_penalty  = -self._total_wait()
        heavy_bonus    = sum(departures[l] for l in green_lanes) * 0.5
        reward = queue_penalty + heavy_bonus

        self.step_count += 1
        terminated = False
        truncated  = self.step_count >= MAX_STEPS

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, truncated, {
            "queues":    dict(self.queues),
            "green":     green_lanes,
            "cleared":   self.total_vehicles_cleared,
        }

    def render(self):
        green = PHASES[self.phase_idx]
        print(f"Step {self.step_count:>4} | "
              f"N:{self.queues['north']:>3} "
              f"S:{self.queues['south']:>3} "
              f"E:{self.queues['east']:>3} "
              f"W:{self.queues['west']:>3} | "
              f"GREEN: {green[0].upper()}-{green[1].upper()} | "
              f"Wait total: {self._total_wait():>4}")
