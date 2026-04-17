import streamlit as st
import pandas as pd
import pydeck as pdk

# -----------------------------------
# Page setup
# -----------------------------------
st.set_page_config(page_title="SF Food Truck MDP Optimizer", layout="wide")
st.title("SF Food Truck Routing with MDP")
st.caption("Optimal next-step routing based on a full Markov Decision Process formulation and value iteration.")

# -----------------------------------
# Constants
# -----------------------------------
LOCATIONS = [
    "financial_district",
    "soma",
    "mission",
    "fishermans_wharf",
    "civic_center"
]

TIME_ORDER = ["morning", "lunch", "afternoon", "evening", "night"]

BASE_REVENUE = {
    "financial_district": 14,
    "soma": 13,
    "mission": 11,
    "fishermans_wharf": 15,
    "civic_center": 11
}

TRAVEL_TIMES = {
    ("financial_district", "financial_district"): 0,
    ("financial_district", "soma"): 5,
    ("financial_district", "mission"): 12,
    ("financial_district", "fishermans_wharf"): 15,
    ("financial_district", "civic_center"): 8,

    ("soma", "financial_district"): 7,
    ("soma", "soma"): 0,
    ("soma", "mission"): 8,
    ("soma", "fishermans_wharf"): 18,
    ("soma", "civic_center"): 6,

    ("mission", "financial_district"): 14,
    ("mission", "soma"): 9,
    ("mission", "mission"): 0,
    ("mission", "fishermans_wharf"): 20,
    ("mission", "civic_center"): 10,

    ("fishermans_wharf", "financial_district"): 12,
    ("fishermans_wharf", "soma"): 16,
    ("fishermans_wharf", "mission"): 18,
    ("fishermans_wharf", "fishermans_wharf"): 0,
    ("fishermans_wharf", "civic_center"): 14,

    ("civic_center", "financial_district"): 9,
    ("civic_center", "soma"): 5,
    ("civic_center", "mission"): 8,
    ("civic_center", "fishermans_wharf"): 15,
    ("civic_center", "civic_center"): 0
}

CONGESTION = {
    "morning": 1.3,
    "lunch": 1.1,
    "afternoon": 1.0,
    "evening": 1.4,
    "night": 0.7
}

LOCATION_COORDS = {
    "financial_district": {"lat": 37.7946, "lon": -122.3999},
    "soma": {"lat": 37.7786, "lon": -122.4059},
    "mission": {"lat": 37.7599, "lon": -122.4148},
    "fishermans_wharf": {"lat": 37.8080, "lon": -122.4177},
    "civic_center": {"lat": 37.7793, "lon": -122.4192},
}

TERMINAL_VALUE = 0.0

# -----------------------------------
# Load data
# -----------------------------------
@st.cache_data
def load_data():
    demand_df = pd.read_csv("data/demand_matrix_bart.csv")
    comp_df = pd.read_csv("data/competition_matrix_schedule.csv")
    return demand_df, comp_df

demand_df, comp_df = load_data()

# -----------------------------------
# Helper functions
# -----------------------------------
def format_location_name(name: str) -> str:
    return name.replace("_", " ").title()

def get_next_time_slot(current_time: str):
    idx = TIME_ORDER.index(current_time)
    if idx == len(TIME_ORDER) - 1:
        return None
    return TIME_ORDER[idx + 1]

def get_demand(location: str, time_slot: str) -> float:
    row = demand_df[demand_df["location"] == location]
    if row.empty:
        return 0.0
    return float(row.iloc[0][time_slot])

def get_competition(location: str, time_slot: str) -> float:
    row = comp_df[comp_df["location"] == location]
    if row.empty:
        return 0.0
    return float(row.iloc[0][time_slot])

def get_travel_time(from_loc: str, to_loc: str) -> float:
    return TRAVEL_TIMES.get((from_loc, to_loc), 999)

def get_success_probability(from_loc: str, to_loc: str, current_time: str) -> float:
    """
    P_success = clamp(1 - travel_min * congestion / 60, 0.3, 0.95)
    """
    if from_loc == to_loc:
        return 1.0
    travel_minutes = get_travel_time(from_loc, to_loc)
    effective_time = travel_minutes * CONGESTION[current_time]
    p = 1.0 - effective_time / 60.0
    return max(0.3, min(0.95, p))

def compute_revenue(location: str, time_slot: str, k: float) -> float:
    """
    Revenue = base_revenue * demand / (1 + competition * k)
    """
    demand = get_demand(location, time_slot)
    competition = get_competition(location, time_slot)
    base = BASE_REVENUE[location]
    return base * demand / (1 + competition * k)

def get_actions(state_location: str):
    """
    We model actions as choosing the next destination, including staying in place.
    So there are 5 actions per state in this project.
    """
    return LOCATIONS.copy()

def expected_immediate_reward(current_loc: str, action_dest: str, current_time: str, k: float) -> float:
    """
    Reward definition based on slide:
    Revenue = base_revenue × demand / (1 + competition × k)

    If stay:
      deterministic, time advances
      reward = revenue_next

    If move:
      success -> Revenue - travel_cost
      delayed -> Revenue * 0.5 - travel_cost * 1.2
      expected reward = p * reward_success + (1-p) * reward_delayed
    """
    next_time = get_next_time_slot(current_time)
    if next_time is None:
        return 0.0

    revenue_next = compute_revenue(action_dest, next_time, k)
    travel_minutes = get_travel_time(current_loc, action_dest)
    travel_cost = travel_minutes * 0.5

    if current_loc == action_dest:
        return revenue_next

    p_success = get_success_probability(current_loc, action_dest, current_time)

    reward_success = revenue_next - travel_cost
    reward_delayed = revenue_next * 0.5 - travel_cost * 1.2

    return p_success * reward_success + (1 - p_success) * reward_delayed

def next_state(current_loc: str, action_dest: str, current_time: str):
    """
    State only includes (location, time_slot).
    Both on-time and delayed outcomes advance to the next time slot.
    The next location is the intended destination.
    """
    next_time = get_next_time_slot(current_time)
    if next_time is None:
        return None
    return (action_dest, next_time)

# -----------------------------------
# MDP: Value Iteration
# -----------------------------------
@st.cache_data
def value_iteration(gamma: float, k: float, theta: float = 1e-6, max_iterations: int = 500):
    """
    Standard Bellman optimality update:
    V(s) = max_a [ E[R(s,a,s')] + gamma * V(s') ]
    """
    states = [(loc, t) for loc in LOCATIONS for t in TIME_ORDER]
    V = {s: 0.0 for s in states}
    policy = {}

    for _ in range(max_iterations):
        delta = 0.0
        new_V = V.copy()

        for loc, t in states:
            next_t = get_next_time_slot(t)

            # Night is the final decision step; after that -> terminal
            if next_t is None:
                new_V[(loc, t)] = TERMINAL_VALUE
                policy[(loc, t)] = None
                continue

            best_action = None
            best_value = float("-inf")

            for action_dest in get_actions(loc):
                immediate = expected_immediate_reward(loc, action_dest, t, k)
                s_prime = next_state(loc, action_dest, t)

                if s_prime is None:
                    total = immediate
                else:
                    total = immediate + gamma * V[s_prime]

                if total > best_value:
                    best_value = total
                    best_action = action_dest

            new_V[(loc, t)] = best_value
            policy[(loc, t)] = best_action
            delta = max(delta, abs(new_V[(loc, t)] - V[(loc, t)]))

        V = new_V
        if delta < theta:
            break

    return V, policy

def build_q_table(current_loc: str, current_time: str, V: dict, gamma: float, k: float) -> pd.DataFrame:
    rows = []
    next_t = get_next_time_slot(current_time)

    if next_t is None:
        return pd.DataFrame()

    for action_dest in LOCATIONS:
        immediate = expected_immediate_reward(current_loc, action_dest, current_time, k)
        future = V[(action_dest, next_t)]
        q_value = immediate + gamma * future
        rows.append({
            "action_destination": action_dest,
            "next_time_slot": next_t,
            "immediate_reward": round(immediate, 2),
            "future_value": round(future, 2),
            "q_value": round(q_value, 2),
            "travel_minutes": get_travel_time(current_loc, action_dest),
            "p_success": round(get_success_probability(current_loc, action_dest, current_time), 3)
        })

    return pd.DataFrame(rows).sort_values(by="q_value", ascending=False)

def build_policy_table(policy: dict, V: dict) -> pd.DataFrame:
    rows = []
    for t in TIME_ORDER:
        for loc in LOCATIONS:
            rows.append({
                "time_slot": t.title(),
                "location": format_location_name(loc),
                "optimal_action": "Terminal" if policy[(loc, t)] is None else format_location_name(policy[(loc, t)]),
                "state_value": round(V[(loc, t)], 2)
            })
    return pd.DataFrame(rows)

def rollout_policy(start_loc: str, start_time: str, policy: dict) -> pd.DataFrame:
    """
    Roll out the optimal policy from current state until terminal.
    """
    rows = []
    current_loc = start_loc
    current_time = start_time

    while True:
        action = policy.get((current_loc, current_time))
        next_t = get_next_time_slot(current_time)

        rows.append({
            "current_time": current_time.title(),
            "current_location": format_location_name(current_loc),
            "optimal_next_action": "Terminal" if action is None else format_location_name(action),
            "next_time": "Terminal" if next_t is None else next_t.title()
        })

        if action is None or next_t is None:
            break

        current_loc = action
        current_time = next_t

    return pd.DataFrame(rows)

def build_all_locations_df():
    rows = []
    for loc, coords in LOCATION_COORDS.items():
        rows.append({
            "location": format_location_name(loc),
            "lat": coords["lat"],
            "lon": coords["lon"]
        })
    return pd.DataFrame(rows)

def build_route_df(current_loc: str, best_dest: str) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "route_label": f"{format_location_name(current_loc)} → {format_location_name(best_dest)}",
            "start_lat": LOCATION_COORDS[current_loc]["lat"],
            "start_lon": LOCATION_COORDS[current_loc]["lon"],
            "end_lat": LOCATION_COORDS[best_dest]["lat"],
            "end_lon": LOCATION_COORDS[best_dest]["lon"],
        }
    ])

# -----------------------------------
# Sidebar inputs
# -----------------------------------
st.sidebar.header("MDP Inputs")

current_location = st.sidebar.selectbox("Current Location", LOCATIONS)
current_time = st.sidebar.selectbox("Current Time Slot", TIME_ORDER)

gamma = st.sidebar.slider(
    "Discount Factor (γ)",
    min_value=0.50,
    max_value=0.99,
    value=0.90,
    step=0.01
)

k_value = st.sidebar.slider(
    "Competition Sensitivity (k)",
    min_value=0.05,
    max_value=0.30,
    value=0.15,
    step=0.05
)

st.sidebar.success("Decision Mode: Optimal Policy (MDP)")

# -----------------------------------
# Top visual annotations
# -----------------------------------
st.subheader("MDP Decision Timeline")
t1, t2, t3, t4, t5, t6 = st.columns(6)

with t1:
    st.info("Morning")
with t2:
    st.info("Lunch")
with t3:
    st.info("Afternoon")
with t4:
    st.info("Evening")
with t5:
    st.info("Night")
with t6:
    st.error("Terminal")

st.caption("State = (location, time_slot). After Night, the process transitions to the terminal state.")

st.subheader("Current State Snapshot")
s1, s2, s3, s4 = st.columns(4)

# Solve MDP
V, policy = value_iteration(gamma=gamma, k=k_value)

state_value = V[(current_location, current_time)]
optimal_action = policy[(current_location, current_time)]
next_time = get_next_time_slot(current_time)

with s1:
    st.metric("Current Location", format_location_name(current_location))
with s2:
    st.metric("Current Time", current_time.title())
with s3:
    st.metric("Discount γ", f"{gamma:.2f}")
with s4:
    st.metric("State Value V(s)", f"{state_value:,.2f}")

# -----------------------------------
# Main output
# -----------------------------------
if optimal_action is None or next_time is None:
    st.warning("This is the final decision step. After Night, the system transitions to the terminal state.")
else:
    q_df = build_q_table(current_location, current_time, V, gamma, k_value)
    best_row = q_df.iloc[0]

    st.subheader("Optimal MDP Recommendation")
    r1, r2, r3, r4 = st.columns(4)

    with r1:
        st.metric("Optimal Next Destination", format_location_name(optimal_action))
    with r2:
        st.metric("Q*(s,a)", f"{best_row['q_value']:,.2f}")
    with r3:
        st.metric("Next Time Slot", next_time.title())
    with r4:
        st.metric("Travel Time (min)", int(best_row["travel_minutes"]))

    st.caption(
        f"Optimal policy recommends: **{format_location_name(current_location)} → {format_location_name(optimal_action)}**"
    )

    # ---------- visuals first ----------
    vcol1, vcol2 = st.columns([1.2, 1])

    with vcol1:
        st.subheader("Action Q-Value Visualization")
        chart_df = q_df[["action_destination", "q_value"]].copy()
        chart_df["action_destination"] = chart_df["action_destination"].apply(format_location_name)
        chart_df = chart_df.set_index("action_destination")
        st.bar_chart(chart_df)
        st.caption("Each bar shows the total expected return Q(s,a) = immediate reward + γ × future value.")

    with vcol2:
        st.subheader("Action Breakdown")
        display_q_df = q_df.copy()
        display_q_df["action_destination"] = display_q_df["action_destination"].apply(format_location_name)
        display_q_df["immediate_reward"] = display_q_df["immediate_reward"].map(lambda x: f"{x:,.2f}")
        display_q_df["future_value"] = display_q_df["future_value"].map(lambda x: f"{x:,.2f}")
        display_q_df["q_value"] = display_q_df["q_value"].map(lambda x: f"{x:,.2f}")
        st.dataframe(display_q_df, use_container_width=True, hide_index=True)
        st.caption("The optimal action is the one with the highest Q-value.")

    # ---------- route map ----------
    st.subheader("Optimal Route Visualization")

    all_locations_df = build_all_locations_df()
    route_df = build_route_df(current_location, optimal_action)

    view_state = pdk.ViewState(
        latitude=37.785,
        longitude=-122.410,
        zoom=11,
        pitch=0,
    )

    scatter_layer = pdk.Layer(
        "ScatterplotLayer",
        data=all_locations_df,
        get_position='[lon, lat]',
        get_radius=250,
        get_fill_color='[80, 130, 255, 160]',
        pickable=True,
    )

    line_layer = pdk.Layer(
        "LineLayer",
        data=route_df,
        get_source_position='[start_lon, start_lat]',
        get_target_position='[end_lon, end_lat]',
        get_width=6,
        get_color='[255, 99, 71]',
        pickable=True,
    )

    deck = pdk.Deck(
        layers=[scatter_layer, line_layer],
        initial_view_state=view_state,
        tooltip={"text": "{route_label}"}
    )

    st.pydeck_chart(deck)
    st.caption("Blue points are candidate locations. The red line shows the optimal next action under the MDP policy.")

    # ---------- rollout ----------
    st.subheader("Optimal Policy Rollout (Until Terminal)")
    rollout_df = rollout_policy(current_location, current_time, policy)
    st.dataframe(rollout_df, use_container_width=True, hide_index=True)
    st.caption("This shows how the optimal policy would continue over the rest of the day.")

    # ---------- policy table ----------
    st.subheader("Optimal Policy Table")
    policy_df = build_policy_table(policy, V)
    st.dataframe(policy_df, use_container_width=True, hide_index=True)

    # ---------- final summary ----------
    st.subheader("Final Recommendation")
    st.success(
        f"Under the MDP optimal policy, the food truck should move from "
        f"**{format_location_name(current_location)}** to "
        f"**{format_location_name(optimal_action)}** "
        f"in the next time slot (**{next_time.title()}**)."
    )