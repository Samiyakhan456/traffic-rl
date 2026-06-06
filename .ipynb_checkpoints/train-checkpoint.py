"""
train.py
--------
Train the Q-learning agent and compare against fixed-time baseline.
Run: python train.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from traffic_env import TrafficEnv
from q_agent    import QLearningAgent
from baseline   import fixed_time_action, FIXED_CYCLE


# ── Config ────────────────────────────────────────────────────────────

TRAIN_EPISODES = 800
EVAL_EPISODES  = 100
EVAL_EVERY     = 50    # evaluate agent every N training episodes
SEED           = 42
np.random.seed(SEED)


# ── Helpers ───────────────────────────────────────────────────────────

def run_episode(env, policy_fn, render=False):
    """
    Run one episode with a given policy function.
    policy_fn(obs) -> action int
    Returns (total_reward, avg_queue, vehicles_cleared)
    """
    obs, _ = env.reset()
    total_reward   = 0
    queue_log      = []
    done           = False

    while not done:
        action = policy_fn(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        queue_log.append(sum(info["queues"].values()))

    return total_reward, np.mean(queue_log), info["cleared"]


def evaluate(env, agent, n=EVAL_EPISODES):
    agent.epsilon_backup = agent.epsilon
    agent.epsilon = 0.0   # greedy eval
    rewards, queues, cleared = [], [], []
    for _ in range(n):
        r, q, c = run_episode(env, agent.select_action)
        rewards.append(r); queues.append(q); cleared.append(c)
    agent.epsilon = agent.epsilon_backup
    return np.mean(rewards), np.mean(queues), np.mean(cleared)


# ── Baseline measurement ──────────────────────────────────────────────

print("=" * 55)
print("  MEASURING FIXED-TIME BASELINE")
print("=" * 55)

env      = TrafficEnv()
baseline_rewards, baseline_queues, baseline_cleared = [], [], []

for ep in range(EVAL_EPISODES):
    obs, _ = env.reset()
    total_reward, queue_log, done = 0, [], False
    while not done:
        # Fixed-time controller uses steps_in_phase from observation
        action = fixed_time_action(step_in_phase=int(obs[5]))
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        queue_log.append(sum(info["queues"].values()))
    baseline_rewards.append(total_reward)
    baseline_queues.append(np.mean(queue_log))
    baseline_cleared.append(info["cleared"])

baseline_avg_queue   = np.mean(baseline_queues)
baseline_avg_reward  = np.mean(baseline_rewards)
baseline_avg_cleared = np.mean(baseline_cleared)

print(f"  Avg queue length : {baseline_avg_queue:.2f} vehicles")
print(f"  Avg episode reward: {baseline_avg_reward:.1f}")
print(f"  Avg vehicles cleared: {baseline_avg_cleared:.0f}")


# ── Training ─────────────────────────────────────────────────────────

print("\n" + "=" * 55)
print("  TRAINING Q-LEARNING AGENT")
print("=" * 55)

agent = QLearningAgent()
env   = TrafficEnv()

train_rewards  = []
eval_rewards   = []
eval_queues    = []
eval_steps     = []

for ep in range(1, TRAIN_EPISODES + 1):
    obs, _ = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = agent.select_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        agent.update(obs, action, reward, next_obs, done)
        obs = next_obs
        total_reward += reward

    agent.decay_epsilon()
    train_rewards.append(total_reward)

    if ep % EVAL_EVERY == 0:
        avg_r, avg_q, avg_c = evaluate(env, agent)
        eval_rewards.append(avg_r)
        eval_queues.append(avg_q)
        eval_steps.append(ep)
        print(f"  Ep {ep:>4} | ε={agent.epsilon:.3f} | "
              f"Avg queue: {avg_q:.2f} | "
              f"Reward: {avg_r:.1f} | "
              f"Q-table states: {len(agent.q_table)}")

agent.save("q_table.pkl")


# ── Final evaluation ──────────────────────────────────────────────────

print("\n" + "=" * 55)
print("  FINAL RESULTS")
print("=" * 55)

final_r, final_q, final_c = evaluate(env, agent, n=EVAL_EPISODES)

improvement = (baseline_avg_queue - final_q) / baseline_avg_queue * 100

print(f"\n  Fixed-time baseline avg queue : {baseline_avg_queue:.2f}")
print(f"  Q-learning agent  avg queue   : {final_q:.2f}")
print(f"\n  ✅ Queue reduction: {improvement:.1f}%")
print(f"  Vehicles cleared (baseline)  : {baseline_avg_cleared:.0f}")
print(f"  Vehicles cleared (RL agent)  : {final_c:.0f}")


# ── Plots ─────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("Traffic Signal RL — Q-Learning vs Fixed-Time Baseline", fontsize=13, fontweight="bold")

# 1. Training reward curve
ax = axes[0]
window = 30
smoothed = np.convolve(train_rewards, np.ones(window)/window, mode="valid")
ax.plot(smoothed, color="#2563EB", linewidth=1.5, label="Training reward (smoothed)")
ax.axhline(baseline_avg_reward, color="#DC2626", linestyle="--", linewidth=1.5, label=f"Baseline ({baseline_avg_reward:.0f})")
ax.set_title("Training Reward Curve")
ax.set_xlabel("Episode"); ax.set_ylabel("Episode Reward")
ax.legend(); ax.grid(alpha=0.3)

# 2. Eval queue length over training
ax = axes[1]
ax.plot(eval_steps, eval_queues, color="#2563EB", marker="o", linewidth=2, label="RL agent")
ax.axhline(baseline_avg_queue, color="#DC2626", linestyle="--", linewidth=1.5, label=f"Baseline ({baseline_avg_queue:.2f})")
ax.set_title("Avg Queue Length During Training")
ax.set_xlabel("Training Episode"); ax.set_ylabel("Avg Queue Length (vehicles)")
ax.legend(); ax.grid(alpha=0.3)

# 3. Final comparison bar chart
ax = axes[2]
labels  = ["Fixed-Time\nBaseline", "Q-Learning\nAgent"]
values  = [baseline_avg_queue, final_q]
colors  = ["#DC2626", "#16A34A"]
bars    = ax.bar(labels, values, color=colors, width=0.4, edgecolor="white")
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            f"{val:.2f}", ha="center", va="bottom", fontweight="bold")
ax.set_title(f"Avg Queue: {improvement:.1f}% Reduction")
ax.set_ylabel("Avg Queue Length (vehicles)")
ax.set_ylim(0, max(values) * 1.2)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("results.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n  Plot saved → results.png")
