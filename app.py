import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk

# -----------------------------------
# Page setup
# -----------------------------------
st.set_page_config(page_title="SF Food Truck MDP", layout="wide")
st.title("SF Food Truck MDP Dashboard")
st.caption(
    "Markov Decision Process model for optimal food truck routing in San Francisco. "
    "Adjust γ and k to see how the optimal policy changes."
)

# -----------------------------------
# Constants from model documentation
# -----------------------------------
LOCATIONS = [
    "financial_district", "soma", "mission",
    "fishermans_wharf", "civic_center"
]

TIME_ORDER = ["morning", "lunch", "afternoon", "evening", "night"]

BASE_REVENUE = {
    "financial_district": 14,
    "soma": 13,
    "mission": 11,
    "fishermans_wharf": 15,
    "civic_center": 11,
}

TRAVEL_TIMES = {
    ("financial_district", "soma"): 5,
    ("financial_district", "mission"): 12,
    ("financial_district", "fishermans_wharf"): 15,
    ("financial_district", "civic_center"): 8,
    ("soma", "financial_district"): 7,
    ("soma", "mission"): 8,
    ("soma", "fishermans_wharf"): 18,
    ("soma", "civic_center"): 6,
    ("mission", "financial_district"): 14,
    ("mission", "soma"): 9,
    ("mission", "fishermans_wharf"): 20,
    ("mission", "civic_center"): 10,
    ("fishermans_wharf", "financial_district"): 12,
    ("fishermans_wharf", "soma"): 16,
    ("fishermans_wharf", "mission"): 18,
    ("fishermans_wharf", "civic_center"): 14,
    ("civic_center", "financial_district"): 9,
    ("civic_center", "soma"): 5,
    ("civic_center", "mission"): 8,
    ("civic_center", "fishermans_wharf"): 15,
}

CONGESTION = {
    "morning": 1.3,
    "lunch": 1.1,
    "afternoon": 1.0,
    "evening": 1.4,
    "night": 0.7,
}

LOCATION_COORDS = {
    "financial_district": {"lat": 37.7946, "lon": -122.3999},
    "soma": {"lat": 37.7786, "lon": -122.4059},
    "mission": {"lat": 37.7599, "lon": -122.4148},
    "fishermans_wharf": {"lat": 37.8080, "lon": -122.4177},
    "civic_center": {"lat": 37.7793, "lon": -122.4192},
}

FUEL_COST_PER_MIN = 0.5

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
def format_location_name(name):
    """Format location ID to display name."""
    return name.replace("_", " ").title()

def get_next_time(t):
    idx = TIME_ORDER.index(t)
    return TIME_ORDER[idx + 1] if idx < len(TIME_ORDER) - 1 else None

def get_demand(loc, t):
    row = demand_df[demand_df["location"] == loc]
    return float(row.iloc[0][t]) if not row.empty else 0

def get_competition(loc, t):
    row = comp_df[comp_df["location"] == loc]
    return float(row.iloc[0][t]) if not row.empty else 0

def compute_revenue(loc, t, k):
    return BASE_REVENUE[loc] * get_demand(loc, t) / (1 + get_competition(loc, t) * k)

def get_travel_time(from_loc, to_loc):
    if from_loc == to_loc:
        return 0
    return TRAVEL_TIMES.get((from_loc, to_loc), 999)

def get_success_probability(from_loc, to_loc, current_time):
    if from_loc == to_loc:
        return 1.0
    travel_minutes = get_travel_time(from_loc, to_loc)
    effective_time = travel_minutes * CONGESTION[current_time]
    p = 1.0 - effective_time / 60
    p = max(0.3, min(0.95, p))
    return p

# -----------------------------------
# MDP Solver — value iteration
# -----------------------------------
def run_value_iteration(gamma, k, theta=1e-6):
    """
    Value iteration solver for the food truck MDP.

    Returns:
        policy     — dict  {(loc, time): "stay" | "move_to_X", ...}
        values     — dict  {(loc, time): float, ...}
        q_table    — dict  {(loc, time): {action: float, ...}, ...}
        iterations — int   number of sweeps until convergence
    """
    policy = {}
    q_table = {}
    values = {(loc, t): 0.0 for loc in LOCATIONS for t in TIME_ORDER}

    def get_actions(loc):
        actions = ["stay"]
        for dest in LOCATIONS:
            if dest != loc:
                actions.append(f"move_to_{dest}")
        return actions

    iterations = 0

    while True:
        delta = 0
        new_values = values.copy()

        for loc in LOCATIONS:
            for t in TIME_ORDER:
                state = (loc, t)
                next_t = get_next_time(t)
                if next_t is None:  # Terminal state at night
                    new_values[state] = 0
                    policy[state] = "stay"
                    q_table[state] = {"stay": 0}
                    continue

                best_q = float("-inf")
                best_action = None
                q_table[state] = {}

                for action in get_actions(loc):
                    if action == "stay":
                        dest = loc
                        travel_time = 0
                    else:
                        dest = action.replace("move_to_", "")
                        travel_time = TRAVEL_TIMES.get((loc, dest), 0)

                    revenue = compute_revenue(dest, next_t, k)
                    travel_cost = travel_time * FUEL_COST_PER_MIN
                    reward = revenue - travel_cost

                    # Bellman update — stay is deterministic, move is stochastic
                    if action == "stay":
                        q_value = reward + gamma * values[(dest, next_t)]
                    else:
                        p = get_success_probability(loc, dest, t)  # Probability of reaching dest
                        q_value = (
                            p * (reward + gamma * values[(dest, next_t)])
                            + (1 - p) * (revenue * 0.5 - travel_cost * 1.2 + gamma * values[(loc, next_t)])
                        )

                    q_table[state][action] = q_value

                    if q_value > best_q:
                        best_q = q_value
                        best_action = action

                new_values[state] = best_q
                policy[state] = best_action

                delta = max(delta, abs(values[state] - best_q))

        values = new_values
        iterations += 1

        if delta < theta:
            break

    return policy, values, q_table, iterations

# -----------------------------------
# Sidebar — Parameters
# -----------------------------------
st.sidebar.header("⚙️ MDP Parameters")

gamma = st.sidebar.slider(
    "Discount Factor (γ)",
    min_value=0.1,
    max_value=0.99,
    value=0.90,
    step=0.01,
    help="Low γ → short-sighted (cares about next slot). High γ → plans across the full day.",
)

k_value = st.sidebar.slider(
    "Competition Sensitivity (k)",
    min_value=0.05,
    max_value=0.30,
    value=0.15,
    step=0.05,
    help="Low k → ignores competition, chases demand. High k → avoids crowded areas.",
)

st.sidebar.divider()

# FIX 1: Moved selectboxes to sidebar (were incorrectly in main body in Doc 6)
st.sidebar.header("🔍 State Inspector")
current_location = st.sidebar.selectbox("Location", LOCATIONS, format_func=format_location_name)
current_time = st.sidebar.selectbox("Time Slot", TIME_ORDER)

# -----------------------------------
# Run MDP
# -----------------------------------
policy, values, q_table, iterations = run_value_iteration(gamma, k_value)

# ===================================================================
# Section 1: Optimal Policy Table
# ===================================================================
st.subheader("Optimal Policy π(s)")
st.caption(
    "Each cell shows the best action for that (location, time) state. "
    "Adjust γ and k in the sidebar to see the policy change."
)

policy_data = []
for loc in LOCATIONS:
    row = {"Location": format_location_name(loc)}
    for t in TIME_ORDER:
        row[t.title()] = policy.get((loc, t), "—")
    policy_data.append(row)

policy_table = pd.DataFrame(policy_data).set_index("Location")
st.dataframe(policy_table, use_container_width=True)

if iterations > 0:
    st.caption(f"Converged in **{iterations}** iterations (θ = 1e-6)")
else:
    st.info("⚠️ Value iteration not yet implemented. Policy table will populate once the solver is connected.")

# ===================================================================
# Section 2: State Value V(s) Heatmap
# ===================================================================
st.subheader("State Values V(s)")
st.caption("Darker color = higher value. Shows expected total reward from each state onward.")

value_data = []
for loc in LOCATIONS:
    row = {"Location": format_location_name(loc)}
    for t in TIME_ORDER:
        # FIX 2: Store as float, not str — color_cells needs numeric values for intensity calculation
        row[t.title()] = round(values.get((loc, t), 0), 2)
    value_data.append(row)

value_df = pd.DataFrame(value_data).set_index("Location")

def color_cells(val):
    try:
        val = float(val)
    except:
        return ""
    if val == 0:
        return "background-color: #f0f0f0; color: #999"
    max_val = value_df.values.max() if value_df.values.max() > 0 else 1
    intensity = val / max_val
    r = int(255 * (1 - intensity))
    g = int(180 + 75 * (1 - intensity))
    b = int(100 + 155 * (1 - intensity))
    return f"background-color: rgb({r},{g},{b}); color: #222; font-weight: bold"

styled_values = value_df.style.map(color_cells)
st.dataframe(styled_values, use_container_width=True)

# ===================================================================
# Section 3: Route Map
# ===================================================================
st.subheader("Route Map")

sel_action = policy.get((current_location, current_time), "—")
sel_dest = None

if sel_action.startswith("move_to_"):
    sel_dest = sel_action.replace("move_to_", "")
elif sel_action == "stay":
    sel_dest = current_location

all_loc_df = pd.DataFrame([
    {"location": format_location_name(loc), "lat": c["lat"], "lon": c["lon"]}
    for loc, c in LOCATION_COORDS.items()
])

layers = [
    pdk.Layer(
        "ScatterplotLayer",
        data=all_loc_df,
        get_position="[lon, lat]",
        get_radius=250,
        get_fill_color="[80, 130, 255, 160]",
        pickable=True,
    )
]

if sel_dest and sel_dest != current_location and sel_dest in LOCATION_COORDS:
    route_df = pd.DataFrame([{
        "route_label": f"{format_location_name(current_location)} → {format_location_name(sel_dest)}",
        "start_lat": LOCATION_COORDS[current_location]["lat"],
        "start_lon": LOCATION_COORDS[current_location]["lon"],
        "end_lat": LOCATION_COORDS[sel_dest]["lat"],
        "end_lon": LOCATION_COORDS[sel_dest]["lon"],
    }])
    layers.append(pdk.Layer(
        "LineLayer",
        data=route_df,
        get_source_position="[start_lon, start_lat]",
        get_target_position="[end_lon, end_lat]",
        get_width=6,
        get_color="[255, 99, 71]",
        pickable=True,
    ))

deck = pdk.Deck(
    layers=layers,
    initial_view_state=pdk.ViewState(
        latitude=37.785, longitude=-122.410, zoom=11.5, pitch=0,
    ),
    tooltip={"text": "{location}"},
)
st.pydeck_chart(deck, use_container_width=True)
st.caption(
    f"Selected state: **{format_location_name(current_location)}** @ **{current_time.title()}** → "
    f"Optimal action: **{sel_action}**"
)

# ===================================================================
# Section 4: Q-Value Breakdown (selected state)
# ===================================================================
st.subheader("Q-Values for Selected State")
st.caption(
    f"All action values at ({format_location_name(current_location)}, {current_time.title()}). "
    f"The highest Q-value determines the optimal action."
)

q_for_state = q_table.get((current_location, current_time), {})
if q_for_state:
    q_rows = []
    for action, q_val in sorted(q_for_state.items(), key=lambda x: -x[1]):
        q_rows.append({"Action": action, "Q(s, a)": round(q_val, 2)})
    q_df = pd.DataFrame(q_rows)
    st.dataframe(q_df, use_container_width=True, hide_index=True)
else:
    st.info("Q-values will appear here once the solver is implemented.")

# ===================================================================
# Section 5: Revenue Breakdown (selected state)
# ===================================================================
st.subheader("Revenue Breakdown")

next_t = get_next_time(current_time)
if next_t is None:
    st.warning("Night is the terminal time slot — no further transitions.")
else:
    rev_rows = []
    for loc in LOCATIONS:
        d = get_demand(loc, next_t)
        c = get_competition(loc, next_t)
        base = BASE_REVENUE[loc]
        denom = 1 + c * k_value
        rev = base * d / denom
        travel = TRAVEL_TIMES.get((current_location, loc), 0) if loc != current_location else 0
        travel_cost = travel * FUEL_COST_PER_MIN

        rev_rows.append({
            "Destination": format_location_name(loc),
            "Base ($)": base,
            "Demand": int(d),
            "Competition": int(c),
            f"1 + comp × {k_value:.2f}": round(denom, 2),
            "Revenue ($)": f"{rev:,.1f}",
            "Travel (min)": travel,
            "Travel Cost ($)": f"{travel_cost:.1f}",
        })

    rev_df = pd.DataFrame(rev_rows)
    st.dataframe(rev_df, use_container_width=True, hide_index=True)
    st.caption(
        f"Revenue at each destination in the **{next_t.title()}** slot.  "
        f"Formula: base × demand / (1 + competition × k)"
    )

# ===================================================================
# Section 6: Input Data — Demand & Competition
# ===================================================================
st.subheader("Input Data")

data_col1, data_col2 = st.columns(2)

with data_col1:
    st.markdown("**Demand Matrix** (BART exits)")
    disp_demand = demand_df[["location"] + TIME_ORDER].copy()
    disp_demand["location"] = disp_demand["location"].apply(format_location_name)
    disp_demand = disp_demand.rename(columns={"location": "Location"})
    disp_demand.columns = ["Location"] + [t.title() for t in TIME_ORDER]
    st.dataframe(disp_demand.set_index("Location"), use_container_width=True)

with data_col2:
    st.markdown("**Competition Matrix** (active trucks)")
    disp_comp = comp_df[["location"] + TIME_ORDER].copy()
    disp_comp["location"] = disp_comp["location"].apply(format_location_name)
    disp_comp = disp_comp.rename(columns={"location": "Location"})
    disp_comp.columns = ["Location"] + [t.title() for t in TIME_ORDER]
    st.dataframe(disp_comp.set_index("Location"), use_container_width=True)