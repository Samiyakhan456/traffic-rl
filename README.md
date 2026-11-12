# Adaptive Traffic Signal Control using Reinforcement Learning
### Q-Learning Agent vs Fixed-Time Baseline | Indian Urban Intersection

[![Open Live Demo](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://traffic-rl-dashboard.streamlit.app)

**Live Demo:** https://traffic-rl-dashboard.streamlit.app


---

## Problem
Most Indian urban intersections use **fixed-time signal controllers** — the green phase lasts a predetermined number of seconds regardless of actual traffic conditions. During peak hours on asymmetric roads (one direction gets 70% of the volume), this causes unnecessary queuing and delay.

## Approach
A **tabular Q-learning agent** that observes real-time queue lengths on all 4 lanes and learns an adaptive switching policy — holding a phase longer when queues are heavy, switching earlier when traffic is light.

### Environment
- Custom `gymnasium`-compatible simulation of a 4-way intersection
- Stochastic Poisson vehicle arrivals modelling Indian traffic mix (45% two-wheelers, 20% autos, 25% cars, 10% heavy vehicles)
- Asymmetric arrival rates: East-West corridor carries higher volume (main road)
- Minimum green time (5 steps) and maximum green time (30 steps) enforced

### State Space
| Feature | Values | Levels |
|---|---|---|
| Queue length per lane (×4) | 0-30+ | 3 bins each |
| Current phase | 0 or 1 | 2 |
| Steps in current phase | 0-30 | 3 bins |
| **Total states** | | **486** |

### Reward Function
`reward = -(total queue length) + 0.5 × (vehicles cleared on green lanes)`

Penalises waiting vehicles while rewarding throughput.

### Q-Learning Hyperparameters
| Parameter | Value |
|---|---|
| Learning rate α | 0.1 |
| Discount factor γ | 0.95 |
| Initial ε (exploration) | 1.0 |
| ε decay | 0.995 |
| Min ε | 0.05 |
| Training episodes | 800 |

## Results
After 800 training episodes, the Q-learning agent reduces average queue length by ~X% compared to the fixed-time baseline with 15-second phase cycles.

See `results.png` for full training curve and final comparison.

## Project Structure
```
traffic_rl/
├── traffic_env.py   # Gymnasium environment (simulation core)
├── q_agent.py       # Q-learning agent with ε-greedy exploration
├── baseline.py      # Fixed-time controller
├── train.py         # Training loop + evaluation + plots
├── q_table.pkl      # Saved Q-table (after training)
└── results.png      # Training curves and final comparison
```

## Run It
```bash
pip install gymnasium numpy matplotlib
python train.py
```

## Next Steps (in progress)
- [ ] Replace tabular Q-learning with DQN for continuous state space
- [ ] Add emergency vehicle detection (priority override)
- [ ] Multi-intersection corridor control
- [ ] Real Navi Mumbai traffic flow data integration
