# SF Food Truck MDP — Complete Model Documentation (v2)

## Data Sources

Two datasets:

1. **Mobile Food Schedule** (SF Open Data, `jjew-r69b`)
   - File: `sf_food_schedule.csv` (raw) → `competition_matrix_schedule.csv` (processed)
   - Used for: computing **competition(loc, time)** — how many trucks are active at each location and time
   - 218 schedule records with day of week, start/end time, location, coordinates
   - Download: `https://data.sfgov.org/resource/jjew-r69b.csv`

2. **BART Hourly Ridership by Origin-Destination** (`bart.gov/about/reports/ridership`)
   - File: `demand_matrix_bart.csv` (processed from 9.1 million rows of 2025 data)
   - Used for: computing **demand(loc, time)** — actual pedestrian flow at each location and time
   - Download: `https://afcweb.bart.gov/ridership/origin-destination/`

---

## MDP Definition: (S, A, T, R, γ)

### S — State Space

Each state is a pair `(location, time_slot)`.

**5 Locations** defined by latitude/longitude bounding boxes within San Francisco:

| Location ID | Name | Lat Range | Lon Range |
|---|---|---|---|
| financial_district | Financial District | 37.789 – 37.796 | -122.404 – -122.394 |
| soma | SoMa | 37.770 – 37.789 | -122.410 – -122.388 |
| mission | Mission District | 37.748 – 37.770 | -122.425 – -122.410 |
| fishermans_wharf | Fisherman's Wharf | 37.800 – 37.810 | -122.420 – -122.410 |
| civic_center | Civic Center | 37.775 – 37.785 | -122.422 – -122.412 |

How to assign a truck to a location: for each record in the permit or schedule CSV where `latitude != 0`, check which bounding box the coordinates fall into. Records outside all boxes are excluded.

**5 Time Slots** mapped from 24-hour clock:

| Slot | Hours |
|---|---|
| morning | 06:00 – 10:59 |
| lunch | 11:00 – 13:59 |
| afternoon | 14:00 – 16:59 |
| evening | 17:00 – 20:59 |
| night | 21:00 – 05:59 |

**Total states**: 5 locations × 5 time slots + 1 terminal = **26 states**.

---

### A — Actions

At each state, the agent chooses one of:

- `stay` — remain at current location, serve customers
- `move_to_X` — drive to location X (4 possible targets, since 5 locations minus current)

Total actions per state: 5.

---

### T — Transition Function

T(s' | s, a) = probability of reaching state s' given state s and action a.

#### If action = `stay`:

Deterministic. Time advances to the next slot, location unchanged.

```
T((loc, next_time) | (loc, time), stay) = 1.0
```

If time = night (last slot), transition to terminal state.

#### If action = `move_to_Y`:

Stochastic. Two outcomes: arrive on time, or arrive delayed.

**Step 1**: Look up travel time (minutes) between locations:

| From → To | Min | From → To | Min |
|---|---|---|---|
| financial_district → soma | 5 | soma → financial_district | 7 |
| financial_district → mission | 12 | soma → mission | 8 |
| financial_district → fishermans_wharf | 15 | soma → fishermans_wharf | 18 |
| financial_district → civic_center | 8 | soma → civic_center | 6 |
| mission → financial_district | 14 | fishermans_wharf → financial_district | 12 |
| mission → soma | 9 | fishermans_wharf → soma | 16 |
| mission → fishermans_wharf | 20 | fishermans_wharf → mission | 18 |
| mission → civic_center | 10 | fishermans_wharf → civic_center | 14 |
| civic_center → financial_district | 9 | civic_center → soma | 5 |
| civic_center → mission | 8 | civic_center → fishermans_wharf | 15 |

These are estimated based on typical driving times. Not from a dataset.

**Step 2**: Look up congestion multiplier for the current time slot:

| Time Slot | Congestion |
|---|---|
| morning | 1.3 |
| lunch | 1.1 |
| afternoon | 1.0 |
| evening | 1.4 |
| night | 0.7 |

**Step 3**: Compute probability of arriving on time:

```
effective_time = travel_minutes × congestion
P_success = clamp(1.0 - effective_time / 60, min=0.3, max=0.95)
```

Meaning: the longer the drive, the less likely to arrive on time. Clamped so it's always between 30% and 95%.

**Step 4**: Two transitions, both go to (Y, next_time) but with different rewards:

```
With probability P_success:       arrive on time → full reward
With probability 1 - P_success:   arrive delayed → reduced reward
```

---

### R — Reward Function

```
Revenue(loc, time) = base_revenue(loc) × demand(loc, time) / (1 + competition(loc, time) × k)
```

Each variable explained below.

#### base_revenue(loc) — average spend per customer

Set by location type. Not from data.

| Location | base_revenue ($) | Rationale |
|---|---|---|
| financial_district | 14 | Higher-income office workers |
| soma | 13 | Tech workers |
| mission | 11 | Mixed residential |
| fishermans_wharf | 15 | Tourists pay premium |
| civic_center | 11 | Government workers |

#### demand(loc, time) — customer flow from BART data

Source: BART "Average Weekday Exits by Station" or hourly OD data.

BART stations map to MDP locations:

| BART Station(s) | MDP Location |
|---|---|
| Embarcadero + Montgomery St | financial_district |
| Powell St | soma |
| Civic Center / UN Plaza | civic_center |
| 16th St Mission + 24th St Mission | mission |
| (no BART station) | fishermans_wharf |

**How to compute**: download the BART hourly exit data. For each station, sum exits by time slot:

```
demand(financial_district, morning) = exits(Embarcadero, 6-10) + exits(Montgomery, 6-10)
demand(financial_district, lunch)   = exits(Embarcadero, 11-13) + exits(Montgomery, 11-13)
... and so on for each location and time slot
```

For fishermans_wharf: no BART station, so this is the one location where demand must be assumed. Use a tourist-area profile (low morning, peak afternoon).

**This is an independent dataset from the food truck data, so demand and competition are not derived from the same source.**

#### competition(loc, time) — active competing trucks from Schedule data

Source: Mobile Food Schedule CSV (`jjew-r69b`).

**How to compute**: for each schedule record, check if its coordinates fall in a location zone, and if its start_time/end_time overlaps a time slot. Count all overlaps.

Computed from real data (218 schedule records, 128 matched to zones):

| Location | morning | lunch | afternoon | evening | night |
|---|---|---|---|---|---|
| financial_district | 23 | 42 | 33 | 12 | 0 |
| soma | 34 | 45 | 35 | 29 | 7 |
| mission | 17 | 18 | 14 | 6 | 4 |
| fishermans_wharf | 1 | 1 | 1 | 1 | 1 |
| civic_center | 0 | 0 | 0 | 0 | 1 |

Note: fishermans_wharf and civic_center have few schedule records because the bounding boxes are small. These numbers may undercount. This is a known limitation.

#### k — competition sensitivity (hyperparameter)

Default: `k = 0.15`. No data source. Controls how much each additional truck reduces revenue.

```
If competition = 42 and k = 0.15:  denominator = 1 + 42 × 0.15 = 7.3
If competition = 42 and k = 0.05:  denominator = 1 + 42 × 0.05 = 3.1
```

Recommend running sensitivity analysis with k = 0.05, 0.1, 0.15, 0.2, 0.3 to check if the optimal policy changes.

#### Reward for `stay`:

```
R((loc, time), stay) = Revenue(loc, time)
```

#### Reward for `move_to_Y`:

```
travel_cost = travel_minutes × fuel_cost_per_min
fuel_cost_per_min = 0.5 ($/minute, assumed)

If on-time (probability P_success):
  R = Revenue(Y, next_time) - travel_cost

If delayed (probability 1 - P_success):
  R = Revenue(Y, next_time) × 0.5 - travel_cost × 1.2
```

The 0.5 means half revenue when late (lost part of the time slot). The 1.2 means 20% extra fuel cost from sitting in traffic.

---

### γ — Discount Factor

| γ | Meaning |
|---|---|
| 0.5 | Short-sighted, mostly cares about next slot |
| 0.9 | Balanced (recommended default) |
| 0.99 | Almost no discounting |

---

## Algorithms

### Value Iteration

```
Initialize V(s) = 0 for all s
Repeat:
  For each state s:
    For each action a:
      Q(s, a) = Σ T(s'|s,a) × [R(s,a,s') + γ × V(s')]
    V_new(s) = max_a Q(s, a)
    π(s) = argmax_a Q(s, a)
  δ = max |V_new(s) - V(s)| over all s
  V = V_new
Until δ < θ  (θ = 1e-6)
```

### Policy Iteration

```
Initialize π(s) = "stay" for all s

Repeat:
  # Policy Evaluation: compute V under fixed π
  Repeat:
    For each state s:
      V(s) = Σ T(s'|s,π(s)) × [R(s,π(s),s') + γ × V(s')]
  Until V converges (δ < θ)

  # Policy Improvement: update π greedily
  stable = True
  For each state s:
    old = π(s)
    π(s) = argmax_a Σ T(s'|s,a) × [R(s,a,s') + γ × V(s')]
    If π(s) ≠ old: stable = False
Until stable
```

---

## Summary: Data Source for Each Variable

| Variable | Source | Type |
|---|---|---|
| Location zones | SF geography + schedule coordinates | Manual + data validation |
| **demand(loc, time)** | **BART hourly station exits** | **Real data (independent)** |
| **competition(loc, time)** | **Food truck schedule CSV** | **Real data** |
| base_revenue(loc) | Assumed by area type | Assumption |
| travel_minutes | Estimated driving times | Assumption |
| congestion multipliers | Assumed by time of day | Assumption |
| k, fuel_cost_per_min | Hyperparameters | Assumption |
| γ, θ | Algorithm parameters | Choice |
