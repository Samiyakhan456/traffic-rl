"""
q_agent.py
----------
Tabular Q-learning agent for traffic signal control.

State discretisation:
  - Each lane queue bucketed into 3 levels: low / medium / high
  - Phase index: 0 or 1
  - Steps in phase bucket: early / mid / late

This keeps the state space small enough for tabular Q-learning
while capturing the information the agent actually needs.
"""

import numpy as np
import pickle


# ── Discretisation ────────────────────────────────────────────────────

def discretise_queue(q):
    """0=low(<5), 1=medium(5-14), 2=high(>=15)"""
    if q < 5:   return 0
    if q < 15:  return 1
    return 2

def discretise_time(t):
    """0=early(<8), 1=mid(8-19), 2=late(>=20)"""
    if t < 8:   return 0
    if t < 20:  return 1
    return 2

def obs_to_state(obs):
    """
    Convert raw observation array to a discrete hashable state tuple.
    obs = [q_N, q_S, q_E, q_W, phase, steps_in_phase]
    """
    q_n, q_s, q_e, q_w, phase, t = obs
    return (
        discretise_queue(q_n),
        discretise_queue(q_s),
        discretise_queue(q_e),
        discretise_queue(q_w),
        int(phase),
        discretise_time(t),
    )
    # State space size: 3^4 * 2 * 3 = 486 states


# ── Q-Learning Agent ──────────────────────────────────────────────────

class QLearningAgent:
    def __init__(
        self,
        n_actions   = 2,
        alpha       = 0.1,    # learning rate
        gamma       = 0.95,   # discount factor
        epsilon     = 1.0,    # initial exploration rate
        epsilon_min = 0.05,
        epsilon_decay = 0.995,
    ):
        self.n_actions     = n_actions
        self.alpha         = alpha
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table       = {}   # state -> [Q(s,0), Q(s,1)]

    def _get_q(self, state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.n_actions)
        return self.q_table[state]

    def select_action(self, obs):
        state = obs_to_state(obs)
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)   # explore
        return int(np.argmax(self._get_q(state)))      # exploit

    def update(self, obs, action, reward, next_obs, done):
        state      = obs_to_state(obs)
        next_state = obs_to_state(next_obs)

        q_current  = self._get_q(state)[action]
        q_next_max = 0 if done else np.max(self._get_q(next_state))

        # Bellman update
        td_target  = reward + self.gamma * q_next_max
        td_error   = td_target - q_current
        self._get_q(state)[action] += self.alpha * td_error

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path="q_table.pkl"):
        with open(path, "wb") as f:
            pickle.dump({"q_table": self.q_table, "epsilon": self.epsilon}, f)
        print(f"Agent saved → {path}")

    def load(self, path="q_table.pkl"):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.q_table = data["q_table"]
        self.epsilon = data["epsilon"]
        print(f"Agent loaded ← {path}  |  States known: {len(self.q_table)}")
