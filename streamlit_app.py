
import streamlit as st
import numpy as np
import pandas as pd
import time

from traffic_env import TrafficEnv, LANES, PHASES
from q_agent import QLearningAgent
from baseline import fixed_time_action

st.set_page_config(
    page_title="Adaptive Traffic Signal Control",
    page_icon="🚦",
    layout="wide"
)

st.title("🚦 Adaptive Traffic Signal Control")
st.caption("Comparing Q-learning with a fixed-time traffic signal controller")

# ---------------- LOAD TRAINED AGENT ----------------

@st.cache_resource
def load_agent():
    agent = QLearningAgent()
    agent.load("q_table.pkl")
    agent.epsilon = 0
    return agent

try:
    agent = load_agent()
    model_loaded = True
except Exception as e:
    agent = None
    model_loaded = False
    st.error(f"Could not load Q-table: {e}")

# ---------------- SIDEBAR ----------------

st.sidebar.header("Simulation Settings")

controller = st.sidebar.selectbox(
    "Choose traffic controller",
    ["Q-learning", "Fixed-time"]
)

simulation_steps = st.sidebar.slider(
    "Simulation steps",
    min_value=50,
    max_value=500,
    value=200,
    step=50
)

run_simulation = st.sidebar.button(
    "▶ Run Simulation",
    type="primary"
)

# ---------------- INTERSECTION DISPLAY ----------------

def display_intersection(queues, phase):
    north_south_green = phase == 0

    north_color = "🟢" if north_south_green else "🔴"
    south_color = "🟢" if north_south_green else "🔴"
    east_color = "🟢" if not north_south_green else "🔴"
    west_color = "🟢" if not north_south_green else "🔴"

    st.markdown(
        f"""
        <div style="background:#30343b; padding:22px;
                    border-radius:12px; text-align:center;
                    color:white;">
            <div style="font-size:18px;">
                {north_color} NORTH
                <br>🚗 🚙 🛵
                <br><b>Queue: {queues['north']}</b>
            </div>
            <div style="font-size:28px; margin:12px;">
                ⬇️
            </div>
            <div style="font-size:20px; padding:20px;
                        background:#555b65; border-radius:8px;">
                🚗  INTERSECTION  🚙
            </div>
            <div style="font-size:28px; margin:12px;">
                ⬆️
            </div>
            <div style="font-size:18px;">
                {south_color} SOUTH
                <br>🚗 🚙 🛵
                <br><b>Queue: {queues['south']}</b>
            </div>
            <hr style="border-color:#777;">
            <div>
                {west_color} WEST: {queues['west']} vehicles
                &nbsp;&nbsp;&nbsp;
                {east_color} EAST: {queues['east']} vehicles
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ---------------- SIMULATION ----------------

if run_simulation:

    if controller == "Q-learning" and not model_loaded:
        st.error("The trained Q-table could not be loaded.")
        st.stop()

    env = TrafficEnv()
    obs, _ = env.reset(seed=42)

    total_queue = 0
    total_cleared = 0
    queue_history = []
    step_history = []

    progress = st.progress(0)
    status = st.empty()
    intersection_area = st.empty()

    for step in range(simulation_steps):

        if controller == "Q-learning":
            action = agent.select_action(obs)
        else:
            action = fixed_time_action(env.steps_in_phase)

        obs, reward, terminated, truncated, info = env.step(action)

        queues = info["queues"]
        current_queue = sum(queues.values())

        total_queue += current_queue
        total_cleared = info["cleared"]

        queue_history.append(current_queue)
        step_history.append(step + 1)

        with intersection_area.container():
            display_intersection(queues, env.phase_idx)

        status.write(
            f"Step: {step + 1} / {simulation_steps} "
            f"| Controller: {controller}"
        )

        progress.progress((step + 1) / simulation_steps)

        if truncated or terminated:
            break

    # ---------------- RESULTS ----------------

    average_queue = total_queue / len(queue_history)

    st.success("Simulation completed!")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Average Queue Length",
        f"{average_queue:.2f}"
    )

    col2.metric(
        "Vehicles Cleared",
        total_cleared
    )

    col3.metric(
        "Simulation Steps",
        len(queue_history)
    )

    st.subheader("Traffic Queue Over Time")

    chart_data = pd.DataFrame({
        "Step": step_history,
        "Total Queue": queue_history
    }).set_index("Step")

    st.line_chart(chart_data)

    st.subheader("Final Traffic Distribution")

    final_queues = pd.DataFrame({
        "Direction": list(queues.keys()),
        "Vehicles Waiting": list(queues.values())
    })

    st.bar_chart(
        final_queues.set_index("Direction")
    )

else:
    st.info(
        "Choose a controller from the sidebar and click "
        "'Run Simulation' to begin."
    )

    st.subheader("How the controllers work")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🧠 Q-learning")
        st.write(
            "Uses the trained Q-table to decide whether "
            "to keep or switch the current signal phase."
        )

    with col2:
        st.markdown("### ⏱️ Fixed-time")
        st.write(
            "Switches the signal according to a fixed "
            "15-step cycle, independent of traffic queues."
        )
