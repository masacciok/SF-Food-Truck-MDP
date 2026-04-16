# SF Food Truck Destination Recommender

## Project Overview
This project builds a decision-support dashboard for food trucks in San Francisco.
The system recommends the next destination based on projected demand, competition,
travel cost, and different selection strategies.

## Dashboard Features
- Select current location
- Select current time slot
- Choose a recommendation strategy
- View recommended destination
- Compare top candidate destinations
- Visualize scores across destinations
- Compare results across multiple strategies

## Data Sources
- demand_matrix_bart.csv
- competition_matrix_schedule.csv
- sf_food_schedule.csv
- model assumptions from MODEL_DOCUMENTATION.md

## How to Run
1. Install dependencies:
   pip install streamlit pandas

2. Run the app:
   streamlit run app.py
