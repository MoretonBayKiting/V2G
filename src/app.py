import os
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np
import inspect
import json
import plotly
from data_in import (
    load_meter_data,
    normalise_meter_data,
    plot_daily_avg_price_per_month,
    plot_hourly_price_by_season,
    plot_hourly_price_se_by_season,
    get_price_data,
    get_price_file,
    combine_all_data,
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
)

from model import run_model, plot_res, Battery, Grid

# from scenario import load_scenario, get_generator_param, get_system_param, get_data_path


def export_df(df, filename):
    df.to_csv(os.path.join(EXPORT_DIR, filename), index=False)


# Utility to get parameter from session state or fallback to default
def get_param(key, default):
    return st.session_state.get(key, default)


def update_params(group, subgroup, edited_params):
    scenario = st.session_state["scenario"]
    scenario[group][subgroup].update(edited_params)
    st.session_state["scenario"] = scenario


DEFAULT_DATA = r"C:\Energy\V2G\data\V2g_scen1.json"
# os.makedirs(DEFAULT_DATA, exist_ok=True)
EXPORT_DIR = r"C:\Energy\V2G\data\synthetic"
os.makedirs(EXPORT_DIR, exist_ok=True)
DEFAULT_PRICE_PATH = r"C:\Energy\V2G\data\NEM\price_all_1h.csv"
used_battery_args = [
    "capacity_kwh",
    "max_charge_kw",
    "max_discharge_kw",
    "cycle_eff_pct",
]
# Initialise with default scenario into session state
if "scenario" not in st.session_state:
    scenario = initialize_from_scenario(
        st,
        DEFAULT_DATA,
        DEFAULT_PRICE_PATH,
        export_df,
        autocast_params,
        generate_synthetic_driving,
        generate_synthetic_pv,
        generate_synthetic_consumption,
        Battery,
        Grid,
        used_battery_args,
    )
else:
    scenario = st.session_state["scenario"]


# --- Sidebar controls ---
st.sidebar.header("Scenario Management")
scenario_json_path = st.sidebar.text_input(
    "Scenario JSON file path", value=DEFAULT_DATA
)
if (
    st.sidebar.button("Load Scenario")
    or st.session_state.get("scenario_json_path") != scenario_json_path
):
    st.session_state["scenario_json_path"] = scenario_json_path
    try:
        scenario = initialize_from_scenario(
            st,
            scenario_json_path,
            DEFAULT_PRICE_PATH,
            export_df,
            autocast_params,
            generate_synthetic_driving,
            generate_synthetic_pv,
            generate_synthetic_consumption,
            Battery,
            Grid,
            used_battery_args,
        )
        st.success("Scenario loaded and parameters applied.")
    except Exception as e:
        st.error(f"Failed to load scenario: {e}")

# Add sidebar options for main page control
st.sidebar.header("⚡ Main Actions")
main_page_option = st.sidebar.selectbox(
    "Choose an action 👇",
    ["edit parameters", "price data", "combine data", "project model"],
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

# --- Main body ---
if main_page_option == "edit parameters":
    params = scenario[group][subgroup]
    edited_params = {}

    st.write(f"Editing parameters for: {group} → {subgroup}")
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
        if group == "generator_params":
            params = scenario[group][subgroup]
            if subgroup == "driving":
                params = autocast_params(generate_synthetic_driving, params)
                df_padded = generate_synthetic_driving(**params)
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
    # st.write("Current scenario:", scenario)        # ... system_params code ...
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
        # value_cols = st.multiselect(
        #     "Select up to 3 series to plot",
        #     value_candidates,
        #     default=value_candidates[:1],
        #     max_selections=3,
        #     key="volatility_series_select",
        # )
        if value_cols and season:
            plot_volatility_timeseries(df, value_cols, season)
    else:
        st.info("No data available or season column missing.")

elif main_page_option == "price data":
    st.header("Select Price Data File")
    price_file = get_price_file(st)
    df_price = get_price_data(st, price_file)
    if df_price is not None:
        export_df(df_price, "df_price.csv")
        if "season" in df_price.columns:
            st.subheader("Short Time Series: Volatility by Season")
            seasons = sorted(df_price["season"].unique())
            season = st.selectbox("Select season", seasons, key="season_select")
            plot_volatility_timeseries(df_price, ["price"], season)
    else:
        st.info("No data available or season column missing.")
elif main_page_option == "combine data":
    st.header("Combine Data")
    df_all = combine_all_data(st)
    export_df(df_all, "df_all.csv")
    # ...code to combine data, show results, etc...
elif main_page_option == "project model":
    st.header("Model Results")
    results_df = run_model(st)
    export_df(st.session_state["results_df"], "results_df.csv")
    col1, col2 = st.columns(2)
    with col1:
        chart_type = st.selectbox(
            "Chart Type",
            ["summary table", "weekly", "sum", "avg", "daily_avg"],
            index=0,
            key="chart_type_select",
            help="Choose how to aggregate and display results",
        )
    if st.session_state["chart_type_select"] == "summary table":
        period_options = ["totals", "daily_averages"]
    else:
        period_options = ["mthly", "season"]
    with col2:
        period = st.selectbox(
            "Period",
            period_options,
            index=0,
            key="period_select",
            help="Choose aggregation period for sum/avg",
        )

    plot_res(st, results_df, chart_type, period)
