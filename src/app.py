import os
import io
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np
import inspect
import json
import plotly
from data_in import (
    # load_meter_data,
    # normalise_meter_data,
    # plot_daily_avg_price_per_month,
    # plot_hourly_price_by_season,
    # plot_hourly_price_se_by_season,
    get_price_data,
    get_price_file,
    export_df,
    plot_volatility_timeseries,
)

from charts import plot_energy_sankey  # boxplot_interval, boxplot_aggpd_season
from synthetic import (
    generate_synthetic_driving,
    show_driving_summary,
    generate_synthetic_pv,
    pv_summary,
    generate_synthetic_consumption,
    cons_summary,
    autocast_params,
    initialize_from_scenario,
    prepare_driving_params,
)

from model import run_model, plot_res, export_df, combine_all_data, Battery, Grid

# from scenario import load_scenario, get_generator_param, get_system_param, get_data_path


# def export_df(df, filename):
# df.to_csv(os.path.join(EXPORT_DIR, filename), index=False)
# print(
#     f"[INFO] export_df called for '{filename}', but file writing is disabled in this environment."
# )
# Optionally, show the first few rows for debugging:
# print(df.head(5))


INPUT_DIR = "data/inputs"
PROCESSED_DIR = "data/processed"


def get_input_path(filename):
    return os.path.join(INPUT_DIR, filename)


def get_processed_path(filename):
    return os.path.join(PROCESSED_DIR, filename)


# Utility to get parameter from session state or fallback to default
def get_param(key, default):
    return st.session_state.get(key, default)


def update_params(group, subgroup, edited_params):
    scenario = st.session_state["scenario"]
    scenario[group][subgroup].update(edited_params)
    st.session_state["scenario"] = scenario


def save_scenario_to_json(filename):
    scenario = st.session_state.get("scenario")
    if scenario is None:
        st.warning("No scenario found in session state.")
        return
    # Ensure directory exists
    file_path = get_input_path(filename)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(scenario, f, indent=2)
    st.success(f"Scenario saved to {file_path}")


used_battery_args = [
    "capacity_kwh",
    "max_charge_kw",
    "max_discharge_kw",
    "cycle_eff_pct",
]
# st.set_page_config(layout="wide") ## This to fill the width
# Add this near the top of your app.py  # Should control width using max_width: 1200px;  But that parameter seems not very effective.
st.set_page_config(layout="wide")  ## This needed if following width control to be used.
# Custom CSS for main container width
st.markdown(
    """
    <style>
    .block-container {
        max-width: 1100px;
        margin-left: auto;
        margin-right: auto;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
# ...existing code...
# Initialise with default scenario into session state
def_scen = "ForRobyn.json"
def_price = "priceVic.csv"
if "scenario" not in st.session_state:
    scenario = initialize_from_scenario(
        st,
        get_input_path(def_scen),
        get_input_path(def_price),
        export_df,
        autocast_params,
        generate_synthetic_driving,
        generate_synthetic_pv,
        generate_synthetic_consumption,
        Battery,
        Grid,
        used_battery_args,
    )
    st.session_state["model_dirty"] = True
else:
    scenario = st.session_state["scenario"]


# --- Sidebar controls ---
st.sidebar.header("Scenario Management")
scenario_json_path = st.sidebar.text_input(
    "Scenario (JSON file)", value=get_input_path(def_scen).replace("\\", "/")
)
# download_filename = st.sidebar.text_input(
#     "Download scenario as...", value="scenario_saved.json"
# )
# download_filename = get_input_path("???.csv").replace("\\", "/")
download_filename = "???.csv"
# Autoload scenario when text input changes
if st.session_state.get("scenario_json_path") != scenario_json_path:
    st.session_state["scenario_json_path"] = scenario_json_path
    try:
        scenario = initialize_from_scenario(
            st,
            scenario_json_path,
            get_input_path(def_price),
            export_df,
            autocast_params,
            generate_synthetic_driving,
            generate_synthetic_pv,
            generate_synthetic_consumption,
            Battery,
            Grid,
            used_battery_args,
        )
        # st.success("Scenario loaded and parameters applied.")
        st.session_state["model_dirty"] = True
    except Exception as e:
        st.error(f"Failed to load scenario: {e}")
# Add download button for Scenario
scenario = st.session_state.get("scenario")
if scenario is not None:
    scenario_json = json.dumps(scenario, indent=2)
    st.sidebar.download_button(
        label="Download Scenario",
        data=scenario_json,
        file_name=download_filename,
        mime="application/json",
    )

# Add sidebar options for main page control
st.sidebar.header("⚡ Main Actions")
main_page_option = st.sidebar.selectbox(
    "Choose an action 👇",
    ["edit parameters", "price data", "project model"],
    key="main_page_option",
)
# ... parameter input code ...
# Only show group/subgroup selectboxes if editing parameters
if main_page_option == "edit parameters":
    group = st.sidebar.selectbox("Parameter group", list(scenario.keys()))
    subgroups = list(scenario[group].keys())
    subgroup = st.sidebar.selectbox("Subgroup", subgroups)
else:
    group = None
    subgroup = None

show_doc = st.sidebar.checkbox("Show Documentation", value=False)
if show_doc:
    with open("V2G.md", "r", encoding="utf-8") as f:
        doc_text = f.read()
    st.markdown(doc_text)

# --- Main body ---
if main_page_option == "edit parameters":
    params = scenario[group][subgroup]
    edited_params = {}

    st.write(f"Editing parameters for: {group} → {subgroup}")
    # In main page, when editing driving parameters
    if group == "generator_params" and subgroup == "driving":
        trips = params.get("trips", [])
        # ... input widgets for trip parameters ...
        # ... update/add/delete/save buttons ...
        # st.write("Current trip parameters:")
        # for i, trip in enumerate(trips):
        #     st.write(f"Trip {i}: {trip}")
        trip_index = st.number_input(
            "Trip Index (base 0)", min_value=0, max_value=max(0, len(trips)), value=0
        )

        driving = st.session_state["scenario"]["generator_params"].get("driving", {})
        if "trips" not in driving:
            driving["trips"] = []

        # UI for editing a single trip set
        # Get current trip values if editing an existing trip
        if trip_index < len(trips):
            trip = trips[trip_index]
        else:
            trip = {}

        cols = st.columns(4)
        with cols[0]:
            probability = st.number_input(
                "Probability",
                min_value=0.0,
                max_value=1.0,
                value=trip.get("probability", 0.8),
            )
            weekday = st.checkbox("Weekday", value=trip.get("weekday", True))
            weekend = st.checkbox("Weekend", value=trip.get("weekend", True))
        with cols[1]:
            distance_mean = st.number_input(
                "Distance Mean (km)", value=trip.get("distance_mean", 50.0)
            )
            distance_se = st.number_input(
                "Distance Std Dev", value=trip.get("distance_std", 0.2)
            )
        with cols[2]:
            time_mean = st.number_input(
                "Time Mean (hour)", value=trip.get("time_mean", 8.0)
            )
            time_se = st.number_input("Time Std Dev", value=trip.get("time_std", 0.2))
        with cols[3]:
            length_mean = st.number_input(
                "Length Mean (hr)", value=trip.get("length_mean", 2.0)
            )
            length_se = st.number_input(
                "Length Std Dev", value=trip.get("length_std", 0.3)
            )

        # # Show all trip sets
        # st.subheader("Current Trip Sets")
        # trips = driving["trips"]
        # for i, trip in enumerate(trips):
        #     st.write(f"**Trip {i}:**", trip)
        cols_1 = st.columns(2)
        with cols_1[0]:
            # Add/Update trip set
            if st.button("Write/Update Trip Set"):
                new_trip = {
                    "probability": probability,
                    "weekday": weekday,
                    "weekend": weekend,
                    "distance_mean": distance_mean,
                    "distance_std": distance_se,
                    "time_mean": time_mean,
                    "time_std": time_se,
                    "length_mean": length_mean,
                    "length_std": length_se,
                }
                if trip_index < len(trips):
                    trips[trip_index] = new_trip
                    st.success(f"Trip {trip_index} updated.")
                else:
                    trips.append(new_trip)
                    st.success(f"Trip {len(trips)-1} added.")
                driving["trips"] = trips
                st.session_state["scenario"]["generator_params"]["driving"] = driving

        with cols_1[1]:
            # Delete trip set
            if st.button("Delete Trip Set"):
                if 0 <= trip_index < len(trips):
                    trips.pop(trip_index)
                    st.success(f"Trip {trip_index} deleted.")
                    driving["trips"] = trips
                    st.session_state["scenario"]["generator_params"][
                        "driving"
                    ] = driving
                else:
                    st.warning("Invalid trip index for deletion.")

        # Optionally, save scenario to file
        # if st.button("Save Scenario"):
        #     with open("data/inputs/V2g_scen1.json", "w") as f:
        #         json.dump(st.session_state["scenario"], f, indent=2)
        #     st.success("Scenario saved to V2g_scen1.json")

    else:
        # Distribute parameter inputs in rows of up to 4 columns
        param_items = list(params.items())
        n_params = len(param_items)
        cols_per_row = 4

        for i in range(0, n_params, cols_per_row):
            cols = st.columns(min(cols_per_row, n_params - i))
            for j, (param, value) in enumerate(param_items[i : i + cols_per_row]):
                with cols[j]:
                    if isinstance(value, (int, float)):
                        new_value = st.number_input(param, value=float(value))
                    else:
                        new_value = st.text_input(param, value=str(value))
                    edited_params[param] = new_value

    if st.button("Save changes & view profile"):
        update_params(group, subgroup, edited_params)
        # Generate and store the relevant DataFrame
        st.session_state["model_dirty"] = True
        if group == "generator_params":
            params = scenario[group][subgroup]
            # if subgroup == "driving":
            #     params = autocast_params(generate_synthetic_driving, params)
            #     df_padded = generate_synthetic_driving(**params)
            #     st.session_state["df_padded"] = df_padded
            #     export_df(df_padded, "df_padded.csv")
            if subgroup == "driving":
                driving_raw = params  # gen_params.get("driving", {})
                driving_params = prepare_driving_params(
                    driving_raw, autocast_params, generate_synthetic_driving
                )
                df_padded = generate_synthetic_driving(**driving_params)
                st.session_state["df_padded"] = df_padded
                export_df(df_padded, "df_padded.csv")
            elif subgroup == "pv":
                params = autocast_params(generate_synthetic_pv, params)
                df_pv = generate_synthetic_pv(**params)
                st.session_state["df_pv"] = df_pv
                export_df(df_pv, "df_pv.csv")
            elif subgroup == "consumption":
                params = autocast_params(generate_synthetic_consumption, params)
                df_cons = generate_synthetic_consumption(**params)
                st.session_state["df_cons"] = df_cons
                export_df(df_cons, "df_cons.csv")

        elif group == "system_params":
            params = scenario[group][subgroup]
            if subgroup == "home_battery":
                # Extract only the required keys for the constructor
                battery_args = {k: params[k] for k in used_battery_args}
                home_battery = Battery(**battery_args)
                st.session_state["home_battery"] = home_battery
            if subgroup == "vehicle_battery":
                # Extract only the required keys for the constructor
                battery_args = {k: params[k] for k in used_battery_args}
                vehicle_battery = Battery(**battery_args)
                st.session_state["vehicle_battery"] = vehicle_battery
            if subgroup == "grid":
                # Extract only the required keys for the constructor
                grid_args = {
                    k: params[k]
                    for k in [
                        "network_cost_import_per_kwh",
                        "network_cost_export_per_kwh",
                        "daily_fee",
                        "max_export_kw",
                    ]
                }
                grid = Grid(**grid_args)
                st.session_state["grid"] = grid
            if subgroup == "global":
                # Assign global parameters to session_state for use in modelling
                for k in ["kwh_per_km", "min_price_threshold", "start_date"]:
                    if k in params:
                        st.session_state[k] = params[k]

    # --- For debugging: show current scenario dict ---
    # st.success("Parameters updated.")
    # st.write("Current scenario:", scenario)

    df = None
    if group == "generator_params":
        if subgroup == "driving" and "df_padded" in st.session_state:
            df = st.session_state["df_padded"]
            show_driving_summary(df, st)
        elif subgroup == "pv" and "df_pv" in st.session_state:
            df = st.session_state["df_pv"]
            pv_summary(df, st)
        elif subgroup == "consumption" and "df_cons" in st.session_state:
            df = st.session_state["df_cons"]
            cons_summary(df, st)

    # --- Season selection and volatility chart ---
    if df is not None and "season" in df.columns:
        st.subheader("Short Time Series: Volatility by Season")
        seasons = sorted(df["season"].unique())
        season = st.selectbox("Select season", seasons, key="season_select")
        value_candidates = [
            c
            for c in df.columns
            if c not in ["date", "hour", "season", "timestamp"]
            and pd.api.types.is_numeric_dtype(df[c])
        ]
        value_cols = value_candidates
        if value_cols and season:
            plot_volatility_timeseries(df, value_cols, season)

    # else:
    # st.info("No data available or season column missing.")
    st.write("Current scenario - not editable:", scenario)

elif main_page_option == "price data":
    st.header("Select Price Data File")
    # Use existing price data if available
    df_price = st.session_state.get("df_price")
    price_file = get_price_file(st)
    # If a new file is selected, load it and update session state
    if price_file and (
        df_price is None or price_file != st.session_state.get("price_file")
    ):
        df_price = get_price_data(st, price_file)
        st.session_state["df_price"] = df_price
        st.session_state["price_file"] = price_file
        st.session_state["model_dirty"] = True

    if df_price is not None:
        export_df(df_price, "df_price.csv")
        if "season" in df_price.columns:
            st.subheader(
                "Randomly selected weekly pricing to show volatility (by season)"
            )
            seasons = sorted(df_price["season"].unique())
            season = st.selectbox("Select season", seasons, key="season_select")
            plot_volatility_timeseries(df_price, ["price"], season)
    # else:
    # st.info("No data available or season column missing.")
    st.write("Current scenario - not editable:", scenario)
elif main_page_option == "project model":
    st.header("Model Results")
    # Only run model if needed
    if (
        st.session_state.get("model_dirty", True)
        or "results_df" not in st.session_state
    ):
        home_battery = st.session_state.get("home_battery")
        vehicle_battery = st.session_state.get("vehicle_battery")
        grid = st.session_state.get("grid")
        kwh_per_km = st.session_state.get("kwh_per_km")
        min_price_threshold = st.session_state.get("min_price_threshold")
        df_all = combine_all_data(st)
        results_df = run_model(
            st, home_battery, vehicle_battery, grid, min_price_threshold, kwh_per_km
        )
        st.session_state["model_dirty"] = False
    else:
        results_df = st.session_state["results_df"]
    export_df(results_df, "results_df.csv")
    col1, col2 = st.columns(2)
    with col1:
        chart_type = st.selectbox(
            "Chart Type",
            ["summary table", "weekly", "single day", "daily_avg"],  # , "sum", "avg"],
            index=0,
            key="chart_type_select",
            help="Choose how to aggregate and display results",
        )
    if st.session_state["chart_type_select"] == "summary table":
        period_options = ["totals", "daily_averages"]
        period_box_name = "Totals or averages"
    elif st.session_state["chart_type_select"] in ["weekly", "single day"]:
        period_options = ["Summer", "Autumn", "Winter", "Spring"]
        period_box_name = "Season"
    else:
        period_options = ["mthly", "season"]
        period_box_name = "Monthly or by season"
    with col2:
        period = st.selectbox(
            period_box_name,
            period_options,
            index=0,
            key="period_select",
            help="Choose aggregation period for sum/avg",
        )

    plot_res(st, results_df, chart_type, period)
# st.write("Current scenario (not editable):", st.session_state["scenario"])
