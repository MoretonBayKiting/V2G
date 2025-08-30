# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly
from data_in import combine_all_data, plot_volatility_timeseries
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
        self.soc_kwh = 0  # State of charge
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


# Example parameter setup
# home_battery = Battery(
#     capacity_kwh=13.5,
#     max_charge_kw=5,
#     max_discharge_kw=5,
#     name="Home Battery",
#     soh_init=1.0,
#     cal_deg_linear=2e-4,
#     cal_deg_sqrt=1e-4,
#     cyc_deg_exp_drive=2.0,
#     cyc_deg_exp_charge=1.5,
# )
# vehicle_battery = Battery(
#     capacity_kwh=60,
#     max_charge_kw=7,
#     max_discharge_kw=7,
#     name="Vehicle Battery",
#     soh_init=1.0,
#     cal_deg_linear=1e-4,
#     cal_deg_sqrt=2e-4,
#     cyc_deg_exp_drive=2.2,
#     cyc_deg_exp_charge=1.3,
# )
# grid = Grid(
#     network_cost_import_per_kwh=0.15,
#     network_cost_export_per_kwh=0.05,
#     daily_fee=1.0,
#     max_export_kw=7,
# )


def calculate_vehicle_target_soc(
    df_all,
    current_time_idx,
    kwh_per_km,
    max_charge_rate,
    min_price_threshold,
    grid,
    lookahead_hours=72,
):
    future = df_all.iloc[current_time_idx + 1 : current_time_idx + 1 + lookahead_hours]
    total_trip_kwh = future["distance_km"].sum() * kwh_per_km

    # Vectorized calculation of cheap import capacity
    prices = future["price"] / 1000 + grid.network_cost_import_per_kwh
    cheap_hours = (prices < min_price_threshold).sum()
    cheap_import_capacity = cheap_hours * max_charge_rate

    target_soc = max(total_trip_kwh - cheap_import_capacity, 0)
    return target_soc


def run_energy_flow_model(
    df_all, home_battery, vehicle_battery, grid, kwh_per_km, min_price_threshold
):
    # Precompute static columns
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

        # Discharge vehicle battery if plugged in
        veh_batt_discharge = 0
        if row.plugged_in:
            veh_batt_discharge = min(
                vehicle_battery.soc_kwh,
                vehicle_battery.max_discharge_kw,
                remaining_consumption,
            )
            vehicle_battery.soc_kwh -= veh_batt_discharge
            remaining_consumption -= veh_batt_discharge

        # Grid import for unmet consumption
        grid_import = max(remaining_consumption, 0)
        grid_import_cost = grid_import * row.effective_import_price

        # Excess PV after consumption
        excess_pv = row.pv_kwh - row.pv_to_consumption

        # Vehicle battery charging target
        vehicle_target_soc = calculate_vehicle_target_soc(
            df_all,
            idx,
            kwh_per_km,
            vehicle_battery.max_charge_kw,
            min_price_threshold,
            grid,
        )
        veh_batt_charge = 0
        if row.plugged_in:
            needed_charge = max(vehicle_target_soc - vehicle_battery.soc_kwh, 0)
            veh_batt_charge = min(
                vehicle_battery.max_charge_kw,
                vehicle_battery.capacity_kwh - vehicle_battery.soc_kwh,
                excess_pv,
                needed_charge,
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
                "vehicle_target_soc": vehicle_target_soc,
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


def run_model(st):
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

    results_df = run_energy_flow_model(
        df_all,
        st.session_state["home_battery"],
        st.session_state["vehicle_battery"],
        st.session_state["grid"],
        kwh_per_km=st.session_state["kwh_per_km"],
        min_price_threshold=st.session_state["min_price_threshold"],
    )

    results_df["total_network_cost"] = (
        results_df["grid_import"] * st.session_state["grid"].network_cost_import_per_kwh
    )
    st.session_state["results_df"] = results_df
    # Average Battery SoC
    # results_df = st.session_state.get("results_df")
    if results_df is None:
        st.warning("No model results found. Please run the model first.")
        print("[DEBUG] results_df is missing or None in session_state.")
    else:
        # Summarize totals for all relevant columns
        exclude_cols = [
            "timestamp",
            "season",
            "home_batt_soc",
            "veh_batt_soc",
            "home_batt_soh",
            "veh_batt_soh",
            "vehicle_target_soc",
        ]
        numeric_cols = [
            col
            for col in results_df.columns
            if pd.api.types.is_numeric_dtype(results_df[col])
            and col not in exclude_cols
        ]
        totals = results_df[numeric_cols].sum().to_frame(name="Total (kWh/$)")
        totals.index.name = "Metric"

        # st.header("Total Energy and Cost Metrics")
        # st.table(totals)
        return results_df

        # st.header("Battery SoC Over Time")
        # fig, ax = plt.subplots()
        # ax.plot(
        #     results_df["timestamp"],
        #     results_df["home_batt_soc"],
        #     label="Home Battery SoC",
        # )
        # ax.plot(
        #     results_df["timestamp"],
        #     results_df["veh_batt_soc"],
        #     label="Vehicle Battery SoC",
        # )
        # ax.set_xlabel("Time")
        # ax.set_ylabel("State of Charge (kWh)")
        # ax.legend()
        # st.pyplot(fig)


def plot_res(st, df, chart_type, period="mthly"):
    if df is not None and "season" in df.columns:
        value_candidates = [
            c
            for c in df.columns
            if c
            not in ["date", "hour", "season", "timestamp", "is_sunny", "plugged_in"]
            and pd.api.types.is_numeric_dtype(df[c])
        ]
        # value_cols = st.multiselect(
        #     "Select up to 3 series to plot",
        #     value_candidates,
        #     default=value_candidates[:1],
        #     max_selections=3,
        #     key="volatility_series_select",
        # )
        if chart_type == "summary table":
            exclude_cols = [
                "timestamp",
                "season",
                "home_batt_soc",
                "veh_batt_soc",
                "home_batt_soh",
                "veh_batt_soh",
                "vehicle_target_soc",
                "is_sunny",
                "plugged_in",
                "price",
                "hour",
                "effective_import_price",
                "remaining_consumption",
                "price_kwh",
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

            sankey_fig = plot_energy_sankey(totals["Total"])
            st.plotly_chart(sankey_fig, use_container_width=True)

        else:
            value_cols = st.multiselect(
                "Select up to 3 series to plot",
                value_candidates,
                default=value_candidates[:1],
                max_selections=3,
                key="volatility_series_select",
            )
            if chart_type == "weekly":
                seasons = sorted(df["season"].unique())
                season = st.selectbox("Select season", seasons, key="season_select")
                if value_cols and season:
                    plot_volatility_timeseries(df, value_cols, season)
            elif chart_type in ["sum", "avg"]:
                if period == "mthly":
                    df["month"] = pd.to_datetime(df["date"]).dt.month
                    group_col = "month"
                else:
                    group_col = "season"
                agg_func = "sum" if chart_type == "sum" else "mean"
                agg_df = df.groupby(group_col)[value_cols].agg(agg_func)
                st.subheader(f"{agg_func.capitalize()} by {group_col.capitalize()}")
                st.line_chart(agg_df)
            elif chart_type == "daily_avg":
                if period == "mthly":
                    df["month"] = pd.to_datetime(df["date"]).dt.month
                    group_col = "month"
                else:
                    group_col = "season"
                sum_df = df.groupby(group_col)[value_cols].sum()
                day_counts = df.groupby(group_col)["date"].nunique()
                daily_avg_df = sum_df.div(day_counts, axis=0)
                st.subheader(f"Daily Average by {group_col.capitalize()}")
                st.line_chart(daily_avg_df)
    else:
        st.info("No data available or season column missing.")
