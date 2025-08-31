# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly
import altair as alt
from data_in import combine_all_data, plot_volatility_timeseries, export_df
from charts import plot_energy_sankey


class Battery:
    def __init__(
        self,
        capacity_kwh,
        max_charge_kw,
        max_discharge_kw,
        cycle_eff_pct,
        name="Battery",
        soh_init=1.0,
        cal_deg_linear=1e-4,  # calendar degradation rate (linear per hour)
        cal_deg_sqrt=1e-4,  # calendar degradation rate (sqrt per sqrt(hour))
        cyc_deg_exp_drive=2.0,  # cycling exponent for driving
        cyc_deg_exp_charge=1.5,  # cycling exponent for charging
    ):
        self.capacity_kwh = capacity_kwh
        self.max_charge_kw = max_charge_kw
        self.max_discharge_kw = max_discharge_kw
        self.cycle_eff_pct = cycle_eff_pct
        self.name = name
        # self.soc_kwh = 0  # State of charge
        self.soc_kwh = capacity_kwh / 2
        self.soh = soh_init  # State of health (fraction, 1.0 = new)
        self.cal_deg_linear = cal_deg_linear
        self.cal_deg_sqrt = cal_deg_sqrt
        self.cyc_deg_exp_drive = cyc_deg_exp_drive
        self.cyc_deg_exp_charge = cyc_deg_exp_charge
        self.age_hours = 0  # total hours in service

    def calendar_degradation(self, t_hours):
        # Calendar degradation: linear + sqrt term
        return self.cal_deg_linear * t_hours + self.cal_deg_sqrt * (t_hours**0.5)

    def cycling_degradation(self, power_kw, mode="charge"):
        if mode == "charge":
            exp = self.cyc_deg_exp_charge
        elif mode == "drive":
            exp = self.cyc_deg_exp_drive
        else:  # "discharge"
            exp = self.cyc_deg_exp_drive
        return (abs(power_kw) / self.capacity_kwh) ** exp

    def __repr__(self):
        return (
            f"{self.name}(capacity={self.capacity_kwh} kWh, "
            f"max_charge={self.max_charge_kw} kW, "
            f"max_discharge={self.max_discharge_kw} kW, "
            f"soc={self.soc_kwh} kWh, soh={self.soh:.4f})"
        )


class Grid:
    def __init__(
        self,
        network_cost_import_per_kwh,
        network_cost_export_per_kwh,
        daily_fee,
        max_export_kw=7,
    ):
        self.network_cost_import_per_kwh = network_cost_import_per_kwh
        self.network_cost_export_per_kwh = network_cost_export_per_kwh
        self.daily_fee = daily_fee
        self.max_export_kw = max_export_kw

    def __repr__(self):
        return (
            f"Grid(import_cost={self.network_cost_import_per_kwh}, "
            f"export_cost={self.network_cost_export_per_kwh}, "
            f"daily_fee={self.daily_fee}, "
            f"max_export_kw={self.max_export_kw})"
        )


import time


def vectorized_target_soc(
    df_all,
    load_components,
    supply_components,
    max_charge_rate=0,
    min_price_threshold=0.0,
    grid=None,
    lookahead_hours=72,
    plugged_in_key="plugged_in",
):
    start = time.time()
    n = len(df_all)
    load = np.zeros(n)
    for comp in load_components:
        load += df_all.get(comp, 0)
    supply = np.zeros(n)
    for comp in supply_components:
        if comp == "pv_kwh":
            supply += df_all.get("pv_kwh", 0)
        elif comp == "cheap_grid":
            prices = (
                df_all["price"] / 1000 + grid.network_cost_import_per_kwh
                if grid is not None
                else df_all.get("price", 0)
            )
            cheap_grid_mask = prices < min_price_threshold
            supply += cheap_grid_mask.astype(float) * max_charge_rate

    # Only allow supply when plugged in
    if plugged_in_key in df_all.columns:
        plugged_in_mask = df_all[plugged_in_key].astype(float).values
        supply = supply * plugged_in_mask

    pad = np.zeros(lookahead_hours - 1)
    load_padded = np.concatenate([load, pad])
    supply_padded = np.concatenate([supply, pad])

    windows_load = np.lib.stride_tricks.sliding_window_view(
        load_padded, lookahead_hours
    )
    windows_supply = np.lib.stride_tricks.sliding_window_view(
        supply_padded, lookahead_hours
    )

    # For each window, sum load and subtract supply in the first period only
    sum_load = np.sum(windows_load, axis=1)
    supply_first = windows_supply[:, 0]

    target_soc = np.maximum(windows_load[:, 0], sum_load - supply_first)

    elapsed = time.time() - start
    print(
        f"[LOG] vectorized_target_soc ({load_components}, {supply_components}) took {elapsed:.3f} seconds"
    )
    return target_soc[:n]


def precompute_static_columns(
    df_all,
    kwh_per_km,
    min_price_threshold,
    grid,
    home_battery,
    vehicle_battery,
    lookahead_hours=72,
):
    start = time.time()
    df_all = df_all.fillna(0)
    df_all["price_kwh"] = df_all["price"] / 1000
    df_all["effective_import_price"] = (
        df_all["price_kwh"] + grid.network_cost_import_per_kwh
    )
    df_all["effective_export_price"] = (
        df_all["price_kwh"] - grid.network_cost_export_per_kwh
    )
    df_all["pv_to_consumption"] = np.minimum(
        df_all["pv_kwh"], df_all["consumption_kwh"]
    )
    df_all["remaining_consumption"] = (
        df_all["consumption_kwh"] - df_all["pv_to_consumption"]
    )
    df_all["vehicle_consumption"] = df_all["distance_km"] * kwh_per_km

    print("[LOG] Starting vectorized target SoC calculations...")
    target_soc_total = vectorized_target_soc(
        df_all,
        load_components=["consumption_kwh", "vehicle_consumption"],
        supply_components=["pv_kwh", "cheap_grid"],
        max_charge_rate=home_battery.max_charge_kw + vehicle_battery.max_charge_kw,
        min_price_threshold=min_price_threshold,
        grid=grid,
        lookahead_hours=lookahead_hours,
    )
    target_soc_vehicle = vectorized_target_soc(
        df_all,
        load_components=["vehicle_consumption"],
        supply_components=["pv_kwh"],
        max_charge_rate=vehicle_battery.max_charge_kw,
        lookahead_hours=lookahead_hours,
    )
    target_soc_home = vectorized_target_soc(
        df_all,
        load_components=["consumption_kwh"],
        supply_components=["pv_kwh", "cheap_grid"],
        max_charge_rate=home_battery.max_charge_kw,
        min_price_threshold=min_price_threshold,
        grid=grid,
        lookahead_hours=lookahead_hours,
    )

    df_all["target_soc_total"] = target_soc_total
    df_all["target_soc_vehicle"] = target_soc_vehicle
    df_all["target_soc_home"] = target_soc_home
    elapsed = time.time() - start
    print(f"[LOG] precompute_static_columns took {elapsed:.3f} seconds")
    return df_all


def run_energy_flow_model(df_all, home_battery, vehicle_battery, grid):

    results = []
    # Use itertuples for faster iteration
    for idx, row in enumerate(df_all.itertuples(index=False)):
        # Stateful variables
        # Discharge home battery
        home_batt_discharge = min(
            home_battery.soc_kwh,
            home_battery.max_discharge_kw,
            row.remaining_consumption,
        )
        home_battery.soc_kwh -= home_batt_discharge
        remaining_consumption = row.remaining_consumption - home_batt_discharge

        # Driving consumption
        driving_discharge = min(vehicle_battery.soc_kwh, row.vehicle_consumption)
        vehicle_battery.soc_kwh -= driving_discharge
        vehicle_battery.soh -= vehicle_battery.cycling_degradation(
            driving_discharge, mode="drive"
        )

        unmet_vehicle_consumption = row.vehicle_consumption - driving_discharge
        # Discharge vehicle battery to supply home consumption if plugged in
        veh_batt_discharge = 0
        if row.plugged_in:
            veh_batt_discharge = min(
                vehicle_battery.soc_kwh,
                vehicle_battery.max_discharge_kw,
                remaining_consumption,
            )
            vehicle_battery.soc_kwh -= veh_batt_discharge
            remaining_consumption -= veh_batt_discharge

        # Vehicle battery export to grid (if plugged in)
        vehicle_export = 0
        if row.plugged_in and row.effective_export_price > 0:
            # Only export if SoC after export >= target
            available_for_export = max(
                vehicle_battery.soc_kwh - row.target_soc_vehicle, 0
            )
            vehicle_export = min(
                available_for_export,
                vehicle_battery.max_discharge_kw,
                grid.max_export_kw,
            )
            vehicle_battery.soc_kwh -= vehicle_export

        # Home battery export to grid
        home_export = 0
        if row.effective_export_price > 0:
            available_for_export = max(home_battery.soc_kwh - row.target_soc_home, 0)
            home_export = min(
                available_for_export,
                home_battery.max_discharge_kw,
                grid.max_export_kw - vehicle_export,  # Don't exceed grid limit
            )
            home_battery.soc_kwh -= home_export

        # Grid import for unmet consumption
        grid_import = max(remaining_consumption, 0)
        grid_import_cost = grid_import * row.effective_import_price

        # Excess PV after consumption
        excess_pv = row.pv_kwh - row.pv_to_consumption

        veh_batt_charge = 0
        if row.plugged_in:
            # needed_charge = max(vehicle_battery.capacity_kwh - vehicle_battery.soc_kwh, 0)
            veh_batt_charge = min(
                vehicle_battery.max_charge_kw,
                vehicle_battery.capacity_kwh - vehicle_battery.soc_kwh,
                excess_pv,
                # needed_charge,
            )
            veh_batt_charge = veh_batt_charge * vehicle_battery.cycle_eff_pct / 100
            veh_batt_loss = veh_batt_charge * (1 - vehicle_battery.cycle_eff_pct / 100)
            vehicle_battery.soc_kwh += veh_batt_charge
            excess_pv -= veh_batt_charge

        # Home battery charging
        home_batt_charge = min(
            home_battery.max_charge_kw,
            home_battery.capacity_kwh - home_battery.soc_kwh,
            excess_pv,
        )
        home_batt_charge = home_batt_charge * home_battery.cycle_eff_pct / 100
        home_batt_loss = home_batt_charge * (1 - home_battery.cycle_eff_pct / 100)
        home_battery.soc_kwh += home_batt_charge
        excess_pv -= home_batt_charge

        # Grid export or curtail
        grid_export = 0
        curtailment = 0
        if excess_pv > 0:
            if row.effective_export_price > 0:
                grid_export = min(excess_pv, grid.max_export_kw)
                excess_pv -= grid_export
            curtailment = excess_pv

        # Battery degradation
        home_battery.soh -= (
            home_battery.calendar_degradation(1)
            + home_battery.cycling_degradation(home_batt_charge, mode="charge")
            + home_battery.cycling_degradation(home_batt_discharge, mode="discharge")
        )
        vehicle_battery.soh -= (
            vehicle_battery.calendar_degradation(1)
            + vehicle_battery.cycling_degradation(veh_batt_charge, mode="charge")
            + vehicle_battery.cycling_degradation(veh_batt_discharge, mode="discharge")
        )

        # Record results
        results.append(
            {
                "date": row.date,
                "hour": row.hour,
                "veh_batt_loss": veh_batt_loss,
                "home_batt_loss": home_batt_loss,
                "grid_import": grid_import,
                "grid_import_cost": grid_import_cost,
                "grid_export": grid_export,
                "home_batt_charge": home_batt_charge,
                "home_batt_discharge": home_batt_discharge,
                "veh_batt_charge": veh_batt_charge,
                "veh_batt_discharge": veh_batt_discharge,
                "curtailment": curtailment,
                "home_batt_soc": home_battery.soc_kwh,
                "veh_batt_soc": vehicle_battery.soc_kwh,
                "home_batt_soh": home_battery.soh,
                "veh_batt_soh": vehicle_battery.soh,
                "driving_discharge": driving_discharge,
                "unmet_vehicle_consumption": unmet_vehicle_consumption,
                "home_export": home_export,
                "vehicle_export": vehicle_export,
                "export_earnings": grid_export
                * (row.price_kwh - grid.network_cost_export_per_kwh),
            }
        )

    return pd.DataFrame(results).merge(df_all, on=["date", "hour"], how="left")


# # %%
# results_df = run_energy_flow_model(
#     df_all,
#     home_battery,
#     vehicle_battery,
#     grid,
#     kwh_per_km=0.18,
#     min_price_threshold=0.05,
# )
# print(results_df.head())

# # %%


def run_model(st, home_battery, vehicle_battery, grid, min_price_threshold, kwh_per_km):
    df_all = st.session_state.get("df_all")
    required_keys = [
        "home_battery",
        "vehicle_battery",
        "grid",
        "kwh_per_km",
        "min_price_threshold",
    ]
    missing = [k for k in required_keys if k not in st.session_state]
    if df_all is None:
        df_all = combine_all_data(st)

    if missing:
        st.warning(f"Missing model parameters: {', '.join(missing)}")
        return

    df_all = precompute_static_columns(
        df_all,
        kwh_per_km,
        min_price_threshold,
        grid,
        home_battery,
        vehicle_battery,
        lookahead_hours=72,
    )
    export_df(df_all, "df_all.csv")
    results_df = run_energy_flow_model(
        df_all,
        st.session_state["home_battery"],
        st.session_state["vehicle_battery"],
        st.session_state["grid"],
    )

    results_df["total_network_cost"] = (
        results_df["grid_import"] * st.session_state["grid"].network_cost_import_per_kwh
    )
    st.session_state["results_df"] = results_df

    if results_df is None:
        st.warning("No model results found. Please run the model first.")
        print("[DEBUG] results_df is missing or None in session_state.")
    else:
        # Summarize totals for relevant columns
        exclude_cols = [
            "timestamp",
            "season",
            "home_batt_soc",
            "veh_batt_soc",
            "home_batt_soh",
            "veh_batt_soh",
            "target_soc_vehicle",
            "target_soc_home",
        ]
        numeric_cols = [
            col
            for col in results_df.columns
            if pd.api.types.is_numeric_dtype(results_df[col])
            and col not in exclude_cols
        ]
        totals = results_df[numeric_cols].sum().to_frame(name="Total (kWh/$)")
        totals.index.name = "Metric"
        return results_df


def plot_res(st, df, chart_type, period="mthly"):
    if df is not None and "season" in df.columns:
        value_candidates = [
            c
            for c in df.columns
            if c
            not in ["date", "hour", "season", "timestamp", "is_sunny", "plugged_in"]
            and pd.api.types.is_numeric_dtype(df[c])
        ]
        if chart_type == "summary table":
            exclude_cols = [
                "timestamp",
                "season",
                "home_batt_soc",
                "veh_batt_soc",
                "home_batt_soh",
                "veh_batt_soh",
                "is_sunny",
                "plugged_in",
                "price",
                "hour",
                "effective_import_price",
                "remaining_consumption",
                "price_kwh",
                "month",
                "target_soc_total",
                "target_soc_vehicle",
                "target_soc_home",
            ]
            numeric_cols = [
                col
                for col in df.columns
                if pd.api.types.is_numeric_dtype(df[col]) and col not in exclude_cols
            ]

            totals = df[numeric_cols].sum().to_frame(name="Total")
            if period == "daily_averages":
                totals = (df[numeric_cols].mean() * 24).to_frame(name="Total")
            totals.index.name = "Metric"

            totals_sorted = totals.sort_values(by="Total", ascending=False)

            # Split into two by size
            mid = len(totals_sorted) // 2
            left_table = totals_sorted.iloc[:mid]
            right_table = totals_sorted.iloc[mid:]

            # Format both tables
            left_formatted = left_table.applymap(lambda x: f"{int(round(x)):,}")
            right_formatted = right_table.applymap(lambda x: f"{int(round(x)):,}")
            if period == "daily_averages":
                left_formatted = left_table.applymap(lambda x: f"{x:,.2f}")
                right_formatted = right_table.applymap(
                    lambda x: f"{x:,.2f}"
                )  # for table in [left_formatted, right_formatted]:
            #     table.index.name = ""
            #     table.columns = [""]

            # left_formatted.index.name = ""
            # left_formatted.columns = [""]
            # right_formatted.index.name = ""
            # right_formatted.columns = [""]

            left_html = left_formatted.to_html(
                classes="styled-table", escape=False, header=False
            )
            left_html = left_html.replace("<td>", '<td style="text-align: right;">')
            right_html = right_formatted.to_html(
                classes="styled-table", escape=False, header=False
            )
            right_html = right_html.replace("<td>", '<td style="text-align: right;">')

            # Show side-by-side in Streamlit
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(left_html, unsafe_allow_html=True)
            with col2:
                st.markdown(right_html, unsafe_allow_html=True)

            # sankey_fig = plot_energy_sankey(totals["Total"])

            sankey_fig, total_flow, double_counted_sum, net_flow = plot_energy_sankey(
                totals["Total"]
            )
            st.plotly_chart(sankey_fig, use_container_width=True)
            st.write(f"Sum of flows: {total_flow:.2f} kWh")
            st.write(f"Double counted: {double_counted_sum:.2f} kWh")
            st.write(f"Net: {net_flow:.2f} kWh")

        else:
            default_names = [
                "unmet_vehicle_consumption",
                "vehicle_consumption",
                "driving_discharge",
            ]
            default_indices = [
                i for i, c in enumerate(value_candidates) if c in default_names
            ]
            default_values = [value_candidates[i] for i in default_indices]
            value_cols = st.multiselect(
                "Select series ",
                value_candidates,
                default=default_values,
                # default=value_candidates[:1],
                # max_selections=3,
                key="volatility_series_select",
            )
            if chart_type == "weekly":
                # seasons = sorted(df["season"].unique())
                # season = st.selectbox("Select season", seasons, key="season_select")
                season = period  # period is passed to plot_res.  It is season for the weekly case.
                if value_cols and season:
                    plot_volatility_timeseries(df, value_cols, season)
            elif chart_type in ["sum", "avg", "daily_avg"]:
                if period == "mthly":
                    df["month"] = pd.to_datetime(df["date"]).dt.month
                    group_col = "month"
                else:
                    group_col = "season"
                if chart_type == "daily_avg":
                    sum_df = df.groupby(group_col)[value_cols].sum()
                    day_counts = df.groupby(group_col)["date"].nunique()
                    plot_df = sum_df.div(day_counts, axis=0)
                else:
                    agg_func = "sum" if chart_type == "sum" else "mean"
                    plot_df = df.groupby(group_col)[value_cols].agg(agg_func)
                # Unified seasonal plotting logic
                if group_col == "season":
                    season_order = ["Summer", "Autumn", "Winter", "Spring"]
                    plot_df = plot_df.reindex(season_order)
                    plot_df = plot_df.reset_index()
                    melted = plot_df.melt(
                        id_vars="season",
                        value_vars=value_cols,
                        var_name="variable",
                        value_name="value",
                    )
                    chart = (
                        alt.Chart(melted)
                        .mark_line(point=True)
                        .encode(
                            x=alt.X("season:N", sort=season_order, title="Season"),
                            y=alt.Y("value:Q", title="Value"),
                            color=alt.Color("variable:N", title="Series"),
                        )
                    )
                    st.subheader(
                        f"{chart_type.replace('_', ' ').capitalize()} by Season"
                    )
                    st.altair_chart(chart, use_container_width=True)
                else:
                    st.subheader(
                        f"{chart_type.replace('_', ' ').capitalize()} by Month"
                    )
                    st.line_chart(plot_df)

    else:
        st.info("No data available or season column missing.")
