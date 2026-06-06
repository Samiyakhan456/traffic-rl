"""
baseline.py
-----------
Fixed-time signal controller — the benchmark your RL agent must beat.
Switches phase every FIXED_CYCLE steps regardless of queue state.
This is what most Indian intersections actually use.
"""

FIXED_CYCLE = 15   # seconds/steps per phase — typical Indian timing


def fixed_time_action(step_in_phase, **kwargs):
    """Returns 1 (switch) every FIXED_CYCLE steps, else 0 (keep)."""
    return 1 if step_in_phase >= FIXED_CYCLE else 0
