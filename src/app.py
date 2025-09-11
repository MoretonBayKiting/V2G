import os
import io
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np
import json
import datetime
from data_in import (
    get_price_data,
    export_df,
    plot_volatility_timeseries,
    show_price_summary,
)

# from charts import plot_energy_sankey  # boxplot_interval, boxplot_aggpd_season
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

from model import (
    run_model,
    plot_res,
    export_df,
    combine_all_data,
    get_summary_table,
    Battery,
    Grid,
)
from tariff import (
    tariff_ui,
    generate_synthetic_tariff_price_df,
    consumption_ui,
)

INPUT_DIR = "data/inputs"
PROCESSED_DIR = "data/processed"
CONFIG_PATH = os.path.join(INPUT_DIR, "config.json")


def load_config(config_path=CONFIG_PATH):
    with open(config_path, "r") as f:
        return json.load(f)


config = load_config()


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
    "target_soc_lookahead_hours",
    "export_lookahead_hours",
    "export_good_price_periods",
    "min_export_price",
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
# Initialise with default scenario into session state
# For scenario selection
scenario_list = config["scenarios"]
def_scen = config.get("default_scenario", scenario_list[0]) + ".json"
# For price selection
price_options = config["price_files"]
def_price = config.get("default_price", price_options[0]) + ".json"


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

# Autoload scenario when text input changes
if "scenario" not in st.session_state:
    scenario_json_path = get_input_path(def_scen + ".json")
    price_path = get_input_path(def_price + ".csv")
    scenario = initialize_from_scenario(
        st,
        scenario_json_path,
        price_path,
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
    # st.session_state["model_dirty"] = True
    st.session_state["scenario"] = scenario
    st.session_state["scenario_json_path"] = scenario_json_path
    st.session_state["model_dirty"] = True

if "mode" not in st.session_state:
    st.session_state["mode"] = "edit"
mode = st.session_state["mode"]
# Add sidebar options for main page control
edit_mode = st.sidebar.button("Edit parameters", key="edit_btn")
if edit_mode:
    st.session_state["mode"] = "edit"
price_mode = st.sidebar.button("Select price series", key="price_btn")
if price_mode:
    st.session_state["mode"] = "price"
project_mode = st.sidebar.button("Project model", key="project_btn")
if project_mode:
    st.session_state["mode"] = "project"
mode = st.session_state.get("mode", "edit")  # default to edit

# Only show group/subgroup selectboxes if editing parameters
if "mode" not in st.session_state:
    st.session_state["mode"] = "edit"
mode = st.session_state["mode"]

if mode == "price":
    price_options = config["price_files"]
    # Use the value returned by the selectbox directly
    price_base = st.sidebar.selectbox(
        "Choose a stored price history file or tariff schedule 👇",
        price_options,
        index=price_options.index(
            st.session_state.get("price_selectbox", price_options[0])
        ),
        key="price_selectbox",
    )

#     if price_base == "synthetic_tariff":
#         tariff_periods = st.session_state.get("tariff_periods", [])
#         start_date = st.session_state.get("start_date", "2024-07-01")
#         n_days = st.session_state.get("n_days", 365)
#         df_price = generate_synthetic_tariff_price_df(
#             tariff_periods, start_date=start_date, n_days=n_days
#         )
#         print(f"synthetic: {df_price.columns}")
#         st.session_state["df_price"] = df_price
#         st.session_state["model_dirty"] = True
#         show_price_summary(df_price, st)
#     else:
#         price_path = get_input_path(price_base + ".csv")
#         st.session_state["price_path"] = price_path
#         df_price = get_price_data(price_path)
#         print(f"not synthetic: {df_price.columns}")
#         st.session_state["df_price"] = df_price
#         st.session_state["price_file"] = price_path
#         st.session_state["model_dirty"] = True
#         show_price_summary(df_price, st)


if mode == "edit":
    # scenario_list = ["2Drives", "drive1", "2DrivesTest"]
    # Get the current scenario base name (without .json)
    current_scenario_path = st.session_state.get(
        "scenario_json_path", get_input_path(def_scen + ".json")
    )

    # scenario_list = ["2DrivesTest", "2Drives", "drive1"]
    # Use the last selected scenario from session state, or default
    last_selected_scenario = st.session_state.get(
        "selected_scenario_base", scenario_list[0]
    )
    scenario_base = st.sidebar.selectbox(
        "Choose a stored scenario 👇",
        scenario_list,
        index=(
            scenario_list.index(last_selected_scenario)
            if last_selected_scenario in scenario_list
            else 0
        ),
        key="scenario_selectbox",
    )
    # Store the user's selection in session state
    st.session_state["selected_scenario_base"] = scenario_base
    scenario_json_path = get_input_path(scenario_base + ".json")
    # Use the current price_path if available
    price_path = st.session_state.get("price_path", get_input_path(def_price + ".csv"))
    # Add download button for Scenario
    # download_filename = get_input_path("???.csv").replace("\\", "/")
    download_filename = scenario_base + ".json"
    scenario = st.session_state.get("scenario")
    if scenario is not None:
        scenario_json = json.dumps(scenario, indent=2)
        st.sidebar.download_button(
            label="Download Scenario",
            data=scenario_json,
            file_name=download_filename,
            mime="application/json",
        )
        # Scenario initialization if selection changes
    if st.session_state.get("scenario_json_path") != scenario_json_path:
        st.session_state["scenario_json_path"] = scenario_json_path
        try:
            scenario = initialize_from_scenario(
                st,
                scenario_json_path,
                price_path,
                export_df,
                autocast_params,
                generate_synthetic_driving,
                generate_synthetic_pv,
                generate_synthetic_consumption,
                Battery,
                Grid,
                used_battery_args,
            )
            st.session_state["scenario"] = scenario
            st.session_state["model_dirty"] = True
        except Exception as e:
            st.error(f"Failed to load scenario: {e}")

    group = st.sidebar.selectbox("Parameter group", list(scenario.keys()))
    subgroups = list(scenario[group].keys())
    subgroup = st.sidebar.selectbox("Subgroup", subgroups)
    st.sidebar.header("Scenario Management")
else:
    group = None
    subgroup = None

# Add a vertical spacer to push the checkboxes to the bottom
st.sidebar.markdown("<div style='height:200px;'></div>", unsafe_allow_html=True)
show_doc = st.sidebar.checkbox("Show User Guide", value=False)
if show_doc:
    with open("UG.md", "r", encoding="utf-8") as f:
        doc_text = f.read()
    st.markdown(doc_text)

show_doc = st.sidebar.checkbox("Show Documentation", value=False)
if show_doc:
    with open("V2G.md", "r", encoding="utf-8") as f:
        doc_text = f.read()
    st.markdown(doc_text)

# Use the selected price_path from the sidebar
if mode == "price":
    # price_options = config["price_files"]
    # # Use the value returned by the selectbox directly
    # price_base = st.sidebar.selectbox(
    #     "Choose a stored price history file or tariff schedule 👇",
    #     price_options,
    #     index=price_options.index(
    #         st.session_state.get("price_selectbox", price_options[0])
    #     ),
    #     key="price_selectbox",
    # )
    price_base = st.session_state.get("price_selectbox", "synthetic_tariff")
    if price_base == "synthetic_tariff":
        st.markdown("### Synthetic Tariff Schedule")
        if st.button("Edit Tariff Schedule"):
            # Show the tariff editing UI
            tariff_periods = st.session_state.get(
                "tariff_periods", scenario["synthetic_data_params"].get("tariff", [])
            )
            updated_tariff = tariff_ui(tariff_periods)
            if updated_tariff is not None:
                st.session_state["tariff_periods"] = updated_tariff
                # Regenerate the synthetic tariff price DataFrame
                start_date = st.session_state.get("start_date", "2024-07-01")
                n_days = st.session_state.get("n_days", 365)
                df_price = generate_synthetic_tariff_price_df(
                    updated_tariff, start_date=start_date, n_days=n_days
                )
                st.session_state["df_price"] = df_price
                st.session_state["model_dirty"] = True
                show_price_summary(df_price, st)
        else:
            # Show summary as before
            df_price = st.session_state.get("df_price")
            if df_price is not None:
                show_price_summary(df_price, st)

    else:
        price_path = get_input_path(price_base + ".csv")
        df_price = get_price_data(price_path)
        st.session_state["df_price"] = df_price
        st.session_state["price_file"] = price_path
        st.session_state["model_dirty"] = True
        if df_price is not None:
            show_price_summary(df_price, st)

    if df_price is not None:
        export_df(st.session_state["export_df_flag"], df_price, "df_price.csv")
        if "season" in df_price.columns:
            st.subheader(
                "Randomly selected weekly pricing to show volatility (by season)"
            )
            seasons = sorted(df_price["season"].unique())
            season = st.selectbox("Select season", seasons, key="season_select")
            plot_volatility_timeseries(df_price, ["price"], season)
    st.write("Current scenario - not editable:", scenario)

elif mode == "edit":
    params = scenario[group][subgroup]
    edited_params = {}

    # st.write(f"Editing parameters for: {group} → {subgroup}")
    # In main page, when editing driving parameters
    if group == "synthetic_data_params" and subgroup == "driving":
        st.markdown(
            """
        **Trip Parameters:**  
        - *trip index*: There are up to 4 "trip sets" - parameters for each set determine their frequency, distance, length etc  
        - *probability*: The probability this trip occurs on a given day  
        - *weekday/weekend*: Whether trip can occur on those days  
        - *distance_mean/std*: Mean and standard deviation of trip distance (km)  
        - *time_mean/std*: Mean and std dev of departure time (hour of day)  
        - *length_mean/std*: Mean and std dev of trip duration (hours)
        """
        )

        trips = params.get("trips", [])
        trip_index = st.number_input(
            "Trip Index (base 0).  Choose or specify up to 4 (indexed from 0 to 3)",
            min_value=0,
            max_value=max(0, len(trips)),
            value=0,
        )

        driving = st.session_state["scenario"]["synthetic_data_params"].get(
            "driving", {}
        )
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
                st.session_state["scenario"]["synthetic_data_params"][
                    "driving"
                ] = driving

        with cols_1[1]:
            # Delete trip set
            if st.button("Delete Trip Set"):
                if 0 <= trip_index < len(trips):
                    trips.pop(trip_index)
                    st.success(f"Trip {trip_index} deleted.")
                    driving["trips"] = trips
                    st.session_state["scenario"]["synthetic_data_params"][
                        "driving"
                    ] = driving
                else:
                    st.warning("Invalid trip index for deletion.")

        # Optionally, save scenario to file
        # if st.button("Save Scenario"):
        #     with open("data/inputs/V2g_scen1.json", "w") as f:
        #         json.dump(st.session_state["scenario"], f, indent=2)
        #     st.success("Scenario saved to V2g_scen1.json")

    elif group == "synthetic_data_params" and subgroup == "tariff":
        tariff_periods = tariff_ui(scenario["synthetic_data_params"]["tariff"])

    elif group == "synthetic_data_params" and subgroup == "consumption":
        consumption_periods = consumption_ui(
            scenario["synthetic_data_params"]["consumption"]
        )

    else:
        if group == "synthetic_data_params" and subgroup == "pv":
            st.markdown(
                """
            **PV Generation Parameters:**  
            - *capacity_kw*: Installed PV system size (kW)  
            - *sunny_prob*: Probability a day is sunny  
            - *summer_gen_factor*: Daily kWh per kW in summer  
            - *winter_gen_factor*: Daily kWh per kW in winter  
            - *cloudy_mean_frac/std_frac*: Mean and std dev of output fraction on cloudy days  
            - *n_days*: Number of days to simulate  
            - *seed*: Random seed for reproducibility
            """
            )
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
        if subgroup != "consumption":
            update_params(group, subgroup, edited_params)
        # Generate and store the relevant DataFrame
        st.session_state["model_dirty"] = True
        if group == "synthetic_data_params":
            params = scenario[group][subgroup]

            if subgroup == "driving":
                driving_raw = params  # gen_params.get("driving", {})
                driving_params = prepare_driving_params(
                    driving_raw, autocast_params, generate_synthetic_driving
                )
                df_padded, df_drive_base = generate_synthetic_driving(**driving_params)
                st.session_state["df_padded"] = df_padded
                st.session_state["df_drive_base"] = df_drive_base
                export_df(
                    st.session_state["export_df_flag"], df_padded, "df_padded.csv"
                )
            elif subgroup == "pv":
                params = autocast_params(generate_synthetic_pv, params)
                df_pv = generate_synthetic_pv(**params)
                st.session_state["df_pv"] = df_pv
                export_df(st.session_state["export_df_flag"], df_pv, "df_pv.csv")
            elif subgroup == "consumption":
                # params is now a list of activities
                activities = params
                n_days = st.session_state["scenario"]["synthetic_data_params"].get(
                    "n_days", 365
                )
                start_date = st.session_state.get("start_date", "2024-07-01")
                seed = st.session_state["scenario"]["synthetic_data_params"].get(
                    "seed", None
                )
                df_cons = generate_synthetic_consumption(
                    activities=activities,
                    n_days=n_days,
                    start_date=start_date,
                    seed=seed,
                )
                st.session_state["df_cons"] = df_cons
                export_df(st.session_state["export_df_flag"], df_cons, "df_cons.csv")
            elif subgroup == "consumption":
                # params is now a list of activities
                activities = params
                n_days = st.session_state["scenario"]["synthetic_data_params"].get(
                    "n_days", 365
                )
                start_date = st.session_state.get("start_date", "2024-07-01")
                seed = st.session_state["scenario"]["synthetic_data_params"].get(
                    "seed", None
                )
                df_cons = generate_synthetic_consumption(
                    activities=activities,
                    n_days=n_days,
                    start_date=start_date,
                    seed=seed,
                )
                st.session_state["df_cons"] = df_cons
                export_df(st.session_state["export_df_flag"], df_cons, "df_cons.csv")
            elif subgroup == "consumption":
                # params is now a list of activities
                activities = params
                n_days = st.session_state["scenario"]["synthetic_data_params"].get(
                    "n_days", 365
                )
                start_date = st.session_state.get("start_date", "2024-07-01")
                seed = st.session_state["scenario"]["synthetic_data_params"].get(
                    "seed", None
                )
                df_cons = generate_synthetic_consumption(
                    activities=activities,
                    n_days=n_days,
                    start_date=start_date,
                    seed=seed,
                )
                st.session_state["df_cons"] = df_cons
                export_df(st.session_state["export_df_flag"], df_cons, "df_cons.csv")
            elif subgroup == "consumption":
                # params is now a list of activities
                activities = params
                n_days = st.session_state["scenario"]["synthetic_data_params"].get(
                    "n_days", 365
                )
                start_date = st.session_state.get("start_date", "2024-07-01")
                seed = st.session_state["scenario"]["synthetic_data_params"].get(
                    "seed", None
                )
                df_cons = generate_synthetic_consumption(
                    activities=activities,
                    n_days=n_days,
                    start_date=start_date,
                    seed=seed,
                )
                st.session_state["df_cons"] = df_cons
                export_df(st.session_state["export_df_flag"], df_cons, "df_cons.csv")
            elif subgroup == "tariff":
                # Generate and store synthetic tariff price DataFrame
                tariff_periods = st.session_state["scenario"]["synthetic_data_params"][
                    "tariff"
                ]
                # You may want to get start_date and n_days from scenario/global params
                start_date = st.session_state.get("start_date", "2024-07-01")
                n_days = st.session_state["scenario"]["synthetic_data_params"].get(
                    "n_days", 365
                )
                df_price = generate_synthetic_tariff_price_df(
                    tariff_periods, start_date=start_date, n_days=n_days
                )

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
                        "fit",
                    ]
                }
                grid = Grid(**grid_args)
                st.session_state["grid"] = grid
            if subgroup == "global":
                # Assign global parameters to session_state for use in modelling
                for k in [
                    "kwh_per_km",
                    "start_date",
                    "public_charge_rate",
                    "export_df_flag",
                    # "debug_date",  #
                ]:
                    if k in params:
                        st.session_state[k] = params[k]
                        print(f"Global parameter {k}: {st.session_state[k]}")

    # --- For debugging: show current scenario dict ---
    # st.success("Parameters updated.")
    # st.write("Current scenario:", scenario)

    df = None
    if group == "synthetic_data_params":
        if subgroup == "driving" and "df_padded" in st.session_state:
            df = st.session_state["df_padded"]
            df_drive_base = st.session_state["df_drive_base"]
            show_driving_summary(df, df_drive_base, st)
        elif subgroup == "pv" and "df_pv" in st.session_state:
            df = st.session_state["df_pv"]
            pv_summary(df, st)
        elif subgroup == "consumption" and "df_cons" in st.session_state:
            df = st.session_state["df_cons"]
            cons_summary(df, st)

    if df is not None and "season" in df.columns:
        st.subheader("Short Time Series: Volatility by Season")
        seasons = sorted(df["season"].unique())
        cols_2 = st.columns(3)
        with cols_2[0]:
            season = st.selectbox("Select season", seasons, key="season_select")
        if season != "Any":
            df_season = df[df["season"] == season]
        else:
            df_season = df
        available_dates = sorted(df_season["date"].unique())
        with cols_2[1]:
            if st.button("Resample week"):
                st.session_state["vol_selected_date"] = np.random.choice(
                    available_dates
                )
        # Use session state to persist selected date across reruns
        if (
            "vol_selected_date" not in st.session_state
            or st.session_state.get("season_last") != season
        ):
            st.session_state["vol_selected_date"] = available_dates[0]
            st.session_state["season_last"] = season
        with cols_2[2]:
            selected_date = st.selectbox(
                "Select date",
                available_dates,
                index=(
                    available_dates.index(st.session_state["vol_selected_date"])
                    if st.session_state["vol_selected_date"] in available_dates
                    else 0
                ),
                key="vol_selected_date_select",
            )
            st.session_state["vol_selected_date"] = selected_date
        value_candidates = [
            c
            for c in df.columns
            if c not in ["date", "hour", "season", "timestamp"]
            and pd.api.types.is_numeric_dtype(df[c])
        ]
        value_cols = value_candidates
        if value_cols and season:
            plot_volatility_timeseries(df_season, value_cols, season, selected_date)
        else:
            st.info("No data available or season column missing.")

    st.write("Current scenario - not editable:", scenario)

elif mode == "project":
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
        results_df = run_model(st, home_battery, vehicle_battery, grid, kwh_per_km)
        st.session_state["model_dirty"] = False
    else:
        results_df = st.session_state["results_df"]
    export_df(st.session_state["export_df_flag"], results_df, "results_df.csv")
    col1, col2, col3, col4 = st.columns(4)
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
        help_text = "Choose either annual totals or daily averages"
    elif st.session_state["chart_type_select"] in ["weekly", "single day"]:
        period_options = ["Any", "Summer", "Autumn", "Winter", "Spring"]
        period_box_name = "Season"
        help_text = "Choose a season or Any. Use Any for freedom to choose dates."
    else:
        period_options = ["mthly", "season"]
        period_box_name = "Monthly or by season"
        help_text = "Who knew help text was required here?"
    with col2:
        period = st.selectbox(
            period_box_name,
            period_options,
            index=0,
            key="period_select",
            help=help_text,
        )
    with col3:
        if chart_type in ["single day", "weekly"]:
            available_dates = sorted(results_df["date"].unique())
            # Use session state to persist index
            if "selected_date_idx" not in st.session_state:
                st.session_state["selected_date_idx"] = 0
            selected_date = st.selectbox(
                "Select date. (Weekly uses nearest Sunday)",
                available_dates,
                index=st.session_state["selected_date_idx"],
                key="single_day_date_select",
                help="If no chart displays, make sure your date is in the right season.",
            )
            st.session_state["selected_date_idx"] = available_dates.index(selected_date)
        else:
            selected_date = None
    inc = 1
    # ...existing code...
    with col4:
        if chart_type in ["single day", "weekly"]:
            if chart_type == "weekly":
                inc = 7
            prev_clicked = st.button(
                "Previous", key="prev_btn", help="move to previous day"
            )
            next_clicked = st.button("Next", key="next_btn", help="move to next day")
            # Only update index if button was clicked
            if prev_clicked and st.session_state["selected_date_idx"] > inc - 1:
                st.session_state["selected_date_idx"] -= inc
            if (
                next_clicked
                and st.session_state["selected_date_idx"] < len(available_dates) - inc
            ):
                st.session_state["selected_date_idx"] += inc
            # Ensure selectbox reflects the updated index
            selected_date = available_dates[st.session_state["selected_date_idx"]]
            st.session_state["selected_date_idx"] = available_dates.index(selected_date)

    plot_res(st, st.session_state["results_df"], chart_type, period, selected_date)

    st.subheader("Save Scenario & Results")
    description = st.text_input("Short description for this run (optional):")
    if st.button("Save Results"):
        # Gather data
        scenario = st.session_state.get("scenario", {})
        price_file = st.session_state.get(
            "price_file", st.session_state.get("price_selectbox", "")
        )
        # Get summary table (reuse your summary table logic)
        results_df = st.session_state.get("results_df")
        summary_table = None
        if results_df is not None:
            public_charge_rate = st.session_state.get("public_charge_rate", 0)
            summary_df = get_summary_table(
                results_df, period="totals", public_charge_rate=public_charge_rate
            )
            summary_table = summary_df["Total"].to_dict()
        # Compose output
        output = {
            "description": description,
            "timestamp": datetime.datetime.now().isoformat(),
            "scenario": scenario,
            "price_file": price_file,
            "summary_table": summary_table,
        }
        # Save to file
        # save_dir = "data/processed/archived_results"
        # os.makedirs(save_dir, exist_ok=True)
        filename = f"results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        # filepath = os.path.join(save_dir, filename)
        # with open(filepath, "w") as f:
        #     json.dump(output, f, indent=2)
        # st.success(f"Results saved to {filepath}")

        json_str = json.dumps(output, indent=2)
        st.download_button(
            label="Download Results JSON",
            data=json_str,
            file_name=filename,
            mime="application/json",
        )
