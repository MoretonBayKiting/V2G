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
        min_export_price=10.0,
        soh_init=1.0,
        target_soc_lookahead_hours=72,  # Period to look ahead to assess required SoC
        export_lookahead_hours=24,  # Period to look ahead for export opportunities
        export_good_price_periods=6,  # Number of good price hours in look ahead period in which to export
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
        self.min_export_price = min_export_price
        self.target_soc_lookahead_hours = target_soc_lookahead_hours
        self.export_lookahead_hours = export_lookahead_hours
        self.export_good_price_periods = export_good_price_periods
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
        if self.capacity_kwh > 0:
            if mode == "charge":
                exp = self.cyc_deg_exp_charge
            elif mode == "drive":
                exp = self.cyc_deg_exp_drive
            else:  # "discharge"
                exp = self.cyc_deg_exp_drive
            return (abs(power_kw) / self.capacity_kwh) ** exp
        else:
            return 0

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
        fit=0.0,
    ):
        self.network_cost_import_per_kwh = network_cost_import_per_kwh
        self.network_cost_export_per_kwh = network_cost_export_per_kwh
        self.daily_fee = daily_fee
        self.max_export_kw = max_export_kw
        self.fit = fit

    def __repr__(self):
        return (
            f"Grid(import_cost={self.network_cost_import_per_kwh}, "
            f"export_cost={self.network_cost_export_per_kwh}, "
            f"daily_fee={self.daily_fee}, "
            f"max_export_kw={self.max_export_kw})"
        )


import time


def vectorized_export_opportunity(
    df, price_col="price_kwh", lookahead_hours=24, top_n=1
):
    """
    Returns a boolean mask for each hour: True if price is among top_n in the lookahead window.
    """
    prices = df[price_col].values
    n = len(prices)
    lookahead_hours = int(lookahead_hours)
    top_n = int(top_n)
    pad = np.zeros(int(lookahead_hours) - 1)
    prices_padded = np.concatenate([prices, pad])
    windows = np.lib.stride_tricks.sliding_window_view(prices_padded, lookahead_hours)
    export_mask = np.zeros(n, dtype=bool)
    for i in range(n):
        # Find indices of top_n prices in the window
        top_indices = np.argpartition(windows[i], -top_n)[-top_n:]
        # If current hour is among top_n, allow export
        if 0 in top_indices:
            export_mask[i] = True
    return export_mask


def rolling_partial_sums(arr, window):
    """
    For each index, returns the cumulative sums of the next 'window' values.
    Example: arr = [1,2,3,4], window=3
    Output: [[1,3,6], [2,5,9], [3,7,0], [4,0,0]]
    (last rows padded with zeros)
    """
    n = len(arr)
    pad = np.zeros(window - 1)
    arr_padded = np.concatenate([arr, pad])
    windows = np.lib.stride_tricks.sliding_window_view(arr_padded, window)
    partial_sums = np.cumsum(windows, axis=1)
    return partial_sums  # shape: (n, window)


import numpy as np


def rolling_partial_dots(df, fields, window):
    """
    For each index, returns the cumulative sums of the element-wise product of the given fields
    over the next 'window' values.
    Example: fields = ["consumption_kwh", "effective_export_price"]
    Output: shape (n, window)
    """
    window = int(window)
    arrs = [df[f].values for f in fields]
    # Element-wise product
    prod = np.prod(arrs, axis=0)
    n = len(prod)
    pad = np.zeros(window - 1)
    prod_padded = np.concatenate([prod, pad])
    windows = np.lib.stride_tricks.sliding_window_view(prod_padded, window)
    partial_sums = np.cumsum(windows, axis=1)
    return partial_sums  # shape: (n, window)


def target_soc(
    df_all,
    weighted_components,  # e.g. [("p", ["consumption_kwh", "effective_import_price"]), ("n", ["pv_kwh", "effective_import_price"]), ...]
    components,  # e.g. [("p", ["consumption_kwh"]), ("n", ["pv_kwh"])]
    lookahead_hours=24,
    debug_date=None,
):
    """
    For each time step, computes the required SoC over the lookahead window,
    using rolling partial sums of the element-wise product of fields.
    weighted_components: list of tuples ("p"/"n", [fields...]) for price-weighted calculation
    components: list of tuples ("p"/"n", [fields...]) for unweighted calculation
    The period over which the battery should be charged is that for which the cost  (partial sum of price weighted energy) is highest. idx_max identifies that period.
    The energy required is that which can be supplied in that period - use idx_max to find that energy from the unweighted partial sum.
    Returns: target_soc (np.ndarray, shape [n])
    """

    if debug_date is None or debug_date == "None":
        df = df_all
    else:
        df = df_all[df_all["date"].astype(str) == debug_date]
        print(f"weighted_components:  {weighted_components}")
        print(f"components:  {components}")
    n = len(df)
    print(f"n (length of dataframe in target_soc): {n}")
    print(f"lookahead_hours : {lookahead_hours}")
    # Compute weighted partial sums
    if weighted_components is not None:
        weighted_arrays = []
        for sign, fields in weighted_components:
            arr = rolling_partial_dots(df, fields, lookahead_hours)
            # print(f"fields: {fields}")
            # print(f"arr: {arr}")
            if sign == "n":
                arr = -arr
            weighted_arrays.append(arr)
            # print(f"weighted_arrays: {weighted_arrays}")
        # Sum all weighted arrays to get the diff
        diff = np.sum(weighted_arrays, axis=0)  # shape (n, lookahead_hours)
        idx_max = np.argmax(diff, axis=1)  # shape (n,)

    # Compute unweighted partial sums
    unweighted_arrays = []
    for sign, fields in components:
        arr = rolling_partial_dots(df, fields, lookahead_hours)
        # print(f"fields: {fields}")
        # print(f"arr: {arr}")
        if sign == "n":
            arr = -arr
        unweighted_arrays.append(arr)
        # print(f"umweighted_arrays: {unweighted_arrays}")
    temp = np.sum(unweighted_arrays, axis=0)  # shape (n, lookahead_hours)
    if weighted_components is None:
        idx_max = np.argmax(temp, axis=1)  # shape (n,)
    # For each time step, select value at idx_max
    # target_soc = temp[np.arange(n), idx_max]
    target_soc = np.maximum(temp[np.arange(n), idx_max], 0)
    if debug_date is None or debug_date == "None":
        return target_soc, None, None, None
    else:
        return target_soc, idx_max, weighted_arrays, unweighted_arrays


def precompute_static_columns(
    st,
    df_all,
    kwh_per_km,
    grid,
    home_battery,
    vehicle_battery,
):
    start = time.time()
    df_all = df_all.fillna(0)
    df_all = df_all.sort_values(["date", "hour"]).reset_index(drop=True)
    df_all["price_kwh"] = df_all["price"] / 1000
    df_all["effective_import_price"] = (
        df_all["price_kwh"] + grid.network_cost_import_per_kwh
    )
    if st.session_state.get("price_selectbox") == "synthetic_tariff":
        df_all["effective_export_price"] = grid.fit
    else:
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

    df_all["vehicle_export_allowed"] = vectorized_export_opportunity(
        df_all,
        price_col="price_kwh",
        lookahead_hours=vehicle_battery.export_lookahead_hours,
        top_n=vehicle_battery.export_good_price_periods,
    )
    df_all["home_export_allowed"] = vectorized_export_opportunity(
        df_all,
        price_col="price_kwh",
        lookahead_hours=home_battery.export_lookahead_hours,
        top_n=home_battery.export_good_price_periods,
    )

    df_all["public_charge_rate"] = st.session_state["public_charge_rate"]
    df_all["allow_charge"] = np.floor(
        df_all["plugged_in"]
    )  # Don't allow charging in part periods
    df_all["not_plugged_in"] = 1 - df_all["plugged_in"]
    df_all["no_charging"] = np.ceil(
        df_all["not_plugged_in"]
    )  # Count all consumption in a partly plugged in period
    # df_all["export_price_sign"] = (df_all["effective_export_price"] > 0).astype(int)
    df_all["positive_export_price"] = np.maximum(df_all["effective_export_price"], 0)

    if st.session_state.get("debug_date") is not None:
        print(f"debug date: {st.session_state["debug_date"]}")
    target_soc_vehicle, idx_max_veh, weighted_arrays, unweighted_arrays = target_soc(
        df_all,
        [
            # ("p", ["vehicle_consumption", "public_charge_rate", "no_charging"]),
            ("p", ["vehicle_consumption", "public_charge_rate"]),
            ("n", ["pv_kwh", "effective_export_price", "allow_charge"]),
        ],
        [
            ("p", ["vehicle_consumption", "no_charging"]),
            # ("p", ["vehicle_consumption"]),
            ("n", ["pv_kwh", "positive_export_price", "allow_charge"]),
        ],
        lookahead_hours=vehicle_battery.target_soc_lookahead_hours,
        debug_date=st.session_state.get("debug_date"),
    )
    target_soc_vehicle = np.clip(target_soc_vehicle, 0, vehicle_battery.capacity_kwh)
    if (
        st.session_state.get("debug_date") is None
        or st.session_state.get("debug_date") == "None"
    ):
        print(f"debug_date is either null or None")
    else:
        print(f"weighted_arrays:  {weighted_arrays}")
        print(f"unweighted_arrays:  {unweighted_arrays}")
        print(f"idx_max:  {idx_max_veh}")
        print(f"target_soc_vehicle:  {target_soc_vehicle}")
    target_soc_home, idx_max_home, weighted_arrays, unweighted_arrays = target_soc(
        df_all,
        [
            ("p", ["consumption_kwh", "effective_import_price"]),
            ("n", ["pv_kwh", "effective_export_price"]),
        ],
        [("p", ["consumption_kwh"]), ("n", ["pv_kwh"])],
        lookahead_hours=home_battery.target_soc_lookahead_hours,
        debug_date=st.session_state.get("debug_date"),
    )
    target_soc_home = np.clip(target_soc_home, 0, home_battery.capacity_kwh)
    # if weighted_arrays is not None:
    #     st.write(weighted_arrays)
    # if unweighted_arrays is not None:
    #     st.write(unweighted_arrays)
    df_all["target_soc_vehicle"] = target_soc_vehicle
    df_all["target_soc_home"] = target_soc_home
    # df_all["idx_max_home"] = idx_max_home
    # df_all["idx_max_veh"] = idx_max_veh

    return df_all


def run_energy_flow_model(st, df_all, home_battery, vehicle_battery, grid):

    results = []
    # Use itertuples for faster iteration
    for idx, row in enumerate(df_all.itertuples(index=False)):
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

        public_charge = row.vehicle_consumption - driving_discharge
        # Discharge vehicle battery to supply home consumption if plugged in
        veh_batt_discharge = 0
        # Only supply load if SoC after export >= target or present price exceeds public charge rate
        # 20250910: Previously had this conditional on veh discharge to supply household consumption.  Definitely not a good one.
        # if st.session_state["public_charge_rate"] < row.effective_export_price:
        available_for_discharge = max(
            vehicle_battery.soc_kwh - row.target_soc_vehicle, 0
        )
        # else:
        # available_for_discharge = 0
        if row.plugged_in > 0:
            veh_batt_discharge = min(
                vehicle_battery.soc_kwh,
                vehicle_battery.max_discharge_kw * row.plugged_in,
                remaining_consumption,
                available_for_discharge,
            )
            vehicle_battery.soc_kwh -= veh_batt_discharge
            remaining_consumption -= veh_batt_discharge

        # Vehicle battery export to grid (if plugged in)
        vehicle_export = 0
        if (
            row.plugged_in
            and row.effective_export_price > vehicle_battery.min_export_price
            and row.vehicle_export_allowed
        ):
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
        if (
            row.effective_export_price > home_battery.min_export_price
            and row.home_export_allowed
        ):
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

        # ...existing code...

        # --- Vehicle battery charging: PV first, then grid if needed ---
        veh_batt_charge = 0
        veh_batt_loss = 0
        veh_batt_charge_grid = 0
        veh_batt_loss_grid = 0
        if row.plugged_in > 0:
            # Required charge to reach target SoC
            required_charge = max(row.target_soc_vehicle - vehicle_battery.soc_kwh, 0)
            # PV available for charging
            pv_charge = min(
                vehicle_battery.max_charge_kw * row.plugged_in,
                vehicle_battery.capacity_kwh - vehicle_battery.soc_kwh,
                excess_pv,
                required_charge,
            )
            veh_batt_charge = pv_charge * vehicle_battery.cycle_eff_pct / 100
            veh_batt_loss = pv_charge * (1 - vehicle_battery.cycle_eff_pct / 100)
            vehicle_battery.soc_kwh += veh_batt_charge
            excess_pv -= pv_charge
            required_charge -= pv_charge

            # If PV was insufficient, use grid import for the remainder
            if required_charge > 0 and veh_batt_discharge == 0:
                grid_charge = min(
                    vehicle_battery.max_charge_kw * row.plugged_in,
                    vehicle_battery.capacity_kwh - vehicle_battery.soc_kwh,
                    required_charge,
                )
                veh_batt_charge_grid = grid_charge * vehicle_battery.cycle_eff_pct / 100
                veh_batt_loss_grid = grid_charge * (
                    1 - vehicle_battery.cycle_eff_pct / 100
                )
                vehicle_battery.soc_kwh += veh_batt_charge_grid
                grid_import += grid_charge
                veh_batt_charge += veh_batt_charge_grid
                veh_batt_loss += veh_batt_loss_grid

        # --- Home battery charging: PV first, then grid if needed ---
        home_batt_charge = 0
        home_batt_loss = 0
        home_batt_charge_grid = 0
        home_batt_loss_grid = 0
        # Required charge to reach target SoC
        required_charge_home = max(row.target_soc_home - home_battery.soc_kwh, 0)
        # PV available for charging
        pv_charge_home = min(
            home_battery.max_charge_kw,
            home_battery.capacity_kwh - home_battery.soc_kwh,
            excess_pv,
            required_charge_home,
        )
        home_batt_charge = pv_charge_home * home_battery.cycle_eff_pct / 100
        home_batt_loss = pv_charge_home * (1 - home_battery.cycle_eff_pct / 100)
        home_battery.soc_kwh += home_batt_charge
        excess_pv -= pv_charge_home
        required_charge_home -= pv_charge_home

        # If PV was insufficient, use grid import for the remainder
        if required_charge_home > 0:
            grid_charge_home = min(
                home_battery.max_charge_kw,
                home_battery.capacity_kwh - home_battery.soc_kwh,
                required_charge_home,
            )
            home_batt_charge_grid = grid_charge_home * home_battery.cycle_eff_pct / 100
            home_batt_loss_grid = grid_charge_home * (
                1 - home_battery.cycle_eff_pct / 100
            )
            home_battery.soc_kwh += home_batt_charge_grid
            grid_import += grid_charge_home
            home_batt_charge += home_batt_charge_grid
            home_batt_loss += home_batt_loss_grid

        extra_home_charge = min(
            home_battery.max_charge_kw,
            home_battery.capacity_kwh - home_battery.soc_kwh,
            excess_pv,
        )
        if extra_home_charge > 0:
            home_batt_charge_extra = (
                extra_home_charge * home_battery.cycle_eff_pct / 100
            )
            home_batt_loss_extra = extra_home_charge * (
                1 - home_battery.cycle_eff_pct / 100
            )
            home_battery.soc_kwh += home_batt_charge_extra
            excess_pv -= extra_home_charge
            home_batt_charge += home_batt_charge_extra
            home_batt_loss += home_batt_loss_extra

        # Then vehicle battery (if plugged in)
        extra_vehicle_charge = 0
        veh_batt_charge_extra = 0
        veh_batt_charge_grid = 0
        if row.plugged_in > 0:
            extra_vehicle_charge = min(
                vehicle_battery.max_charge_kw * row.plugged_in,
                vehicle_battery.capacity_kwh - vehicle_battery.soc_kwh,
                excess_pv,
            )
            if extra_vehicle_charge > 0:
                veh_batt_charge_extra = (
                    extra_vehicle_charge * vehicle_battery.cycle_eff_pct / 100
                )
                veh_batt_loss_extra = extra_vehicle_charge * (
                    1 - vehicle_battery.cycle_eff_pct / 100
                )
                vehicle_battery.soc_kwh += veh_batt_charge_extra
                excess_pv -= extra_vehicle_charge
                veh_batt_charge += veh_batt_charge_extra
                veh_batt_loss += veh_batt_loss_extra

        # Grid export or curtail
        pv_export = 0
        curtailment = 0
        if excess_pv > 0:
            if row.effective_export_price > 0:
                pv_export = min(excess_pv, grid.max_export_kw)
                excess_pv -= pv_export
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
                "pv_export": pv_export,
                "home_batt_charge": home_batt_charge,
                "home_batt_discharge": home_batt_discharge,
                "veh_batt_charge": veh_batt_charge,
                "veh_batt_charge_extra": veh_batt_charge_extra,
                "veh_batt_charge_grid": veh_batt_charge_grid,
                "veh_batt_discharge": veh_batt_discharge,
                "curtailment": curtailment,
                "home_batt_soc": home_battery.soc_kwh,
                "veh_batt_soc": vehicle_battery.soc_kwh,
                "home_batt_soh": home_battery.soh,
                "veh_batt_soh": vehicle_battery.soh,
                "driving_discharge": driving_discharge,
                "public_charge": public_charge,
                "home_export": home_export,
                "vehicle_export": vehicle_export,
                "pv_earnings": pv_export * row.effective_export_price,
                "veh_earnings": vehicle_export * row.effective_export_price,
                "home_earnings": home_export * row.effective_export_price,
                "curtailment_op_cost": curtailment * row.effective_export_price,
            }
        )

    return pd.DataFrame(results).merge(df_all, on=["date", "hour"], how="left")


def run_model(st, home_battery, vehicle_battery, grid, kwh_per_km):
    df_all = st.session_state.get("df_all")
    required_keys = [
        "home_battery",
        "vehicle_battery",
        "grid",
        "kwh_per_km",
    ]
    missing = [k for k in required_keys if k not in st.session_state]
    if df_all is None:
        df_all = combine_all_data(st)

    if missing:
        st.warning(f"Missing model parameters: {', '.join(missing)}")
        return
    export_df(st.session_state["export_df_flag"], df_all, "df_all_b4_precompute.csv")
    df_all = precompute_static_columns(
        st,
        df_all,
        kwh_per_km,
        grid,
        home_battery,
        vehicle_battery,
    )
    export_df(st.session_state["export_df_flag"], df_all, "df_all.csv")
    results_df = run_energy_flow_model(
        st,
        df_all,
        st.session_state["home_battery"],
        st.session_state["vehicle_battery"],
        st.session_state["grid"],
    )
    export_df(st.session_state["export_df_flag"], results_df, "results_df1.csv")

    results_df["network_variable_cost"] = (
        -results_df["grid_import"]
        * st.session_state["grid"].network_cost_import_per_kwh
    )
    results_df["network_fixed_cost"] = -st.session_state["grid"].daily_fee / 24
    results_df["grid_energy_cost"] = (
        -results_df["grid_import_cost"] - results_df["network_variable_cost"]
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
            "vehicle_export_allowed",
            "home_export_allowed",
            # "grid_import_cost",
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


def plot_res(st, df, chart_type, period="mthly", selected_date="2025-01-01"):
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
                "effective_export_price",
                "remaining_consumption",
                "price_kwh",
                "month",
                "target_soc_total",
                "target_soc_vehicle",
                "target_soc_home",
                "vehicle_export_allowed",
                "home_export_allowed",
            ]
            numeric_cols = [
                col
                for col in df.columns
                if pd.api.types.is_numeric_dtype(df[col]) and col not in exclude_cols
            ]

            if period == "daily_averages":
                totals = (df[numeric_cols].mean() * 24).to_frame(name="Total")
            else:
                totals = df[numeric_cols].sum().to_frame(name="Total")
            totals.index.name = "Metric"

            # Add these after totals is created and before sorting/formatting
            def safe_divide(totals, num, denom):
                num_val = totals.loc[num, "Total"] if num in totals.index else np.nan
                denom_val = (
                    totals.loc[denom, "Total"] if denom in totals.index else np.nan
                )
                return (
                    100 * (num_val / denom_val)
                    if denom_val not in [0, np.nan]
                    else np.nan
                )

            totals.loc["veh_export_rate", "Total"] = safe_divide(
                totals, "veh_earnings", "vehicle_export"
            )
            totals.loc["home_export_rate", "Total"] = safe_divide(
                totals, "home_earnings", "home_export"
            )
            totals.loc["grid_import_rate", "Total"] = safe_divide(
                totals, "grid_import_cost", "grid_import"
            )
            totals.loc["pv_export_rate", "Total"] = safe_divide(
                totals, "pv_earnings", "pv_export"
            )
            totals.loc["curtailment_rate", "Total"] = safe_divide(
                totals, "curtailment_op_cost", "curtailment"
            )
            totals.loc["grid_energy_rate", "Total"] = -safe_divide(
                totals, "grid_energy_cost", "grid_import"
            )
            public_charge_rate = st.session_state["public_charge_rate"]
            totals.loc["public_charge_cost", "Total"] = (
                -totals.loc["public_charge", "Total"] * public_charge_rate
            )
            totals.loc["public_charge_rate", "Total"] = (
                public_charge_rate * 100
            )  # Need to sort out why this factor of 100 is needed?
            metric_units = {
                # kWh metrics
                "pv_kwh": ("kWh", 0),
                "grid_import": ("kWh", 0),
                "public_charge": ("kWh", 0),
                "consumption_kwh": ("kWh", 0),
                "vehicle_consumption": ("kWh", 0),
                "curtailment": ("kWh", 0),
                "vehicle_export": ("kWh", 0),
                "home_export": ("kWh", 0),
                "pv_export": ("kWh", 0),
                "home_batt_loss": ("kWh", 0),
                "veh_batt_loss": ("kWh", 0),
                # $ metrics
                "home_earnings": ("$", 1),
                "veh_earnings": ("$", 1),
                "pv_earnings": ("$", 1),
                "network_variable_cost": ("$", 1),
                "network_fixed_cost": ("$", 1),
                "grid_energy_cost": ("$", 1),
                "public_charge_cost": ("$", 1),
                # "network_variable_cost": ("$", -1), #Previously had these as -1.  Changed to + and altered their definitions above so that they are negative.
                # "network_fixed_cost": ("$", -1),
                # "grid_energy_cost": ("$", -1),
                # "public_charge_cost": ("$", -1),
                "grid_import_cost": ("$", 0),
                "curtailment_op_cost": ("$", 0),
                # c metrics
                "grid_import_rate": ("c", 0),
                "home_export_rate": ("c", 0),
                "veh_export_rate": ("c", 0),
                "grid_energy_rate": ("c", 0),
                "pv_export_rate": ("c", 0),
                "curtailment_rate": ("c", 0),
                "public_charge_rate": ("c", 0),
            }
            # Use metric_units keys for ordering
            ordered_metrics = list(metric_units.keys())

            tables = {"kWh": [], "$": [], "c": []}
            for metric in ordered_metrics:
                unit, _ = metric_units[metric]
                if metric in totals.index:
                    tables[unit].append(metric)

            # Display tables side-by-side
            col_kwh, col_dollar, col_cents = st.columns(3)
            for col, unit in zip([col_kwh, col_dollar, col_cents], ["kWh", "$", "c"]):
                metrics = tables[unit]
                if metrics:
                    sub_totals = totals.loc[metrics].copy()
                    sub_totals.index.name = f"Metric ({unit})"

                    # Determine sign for each metric
                    signs = [metric_units[m][1] for m in metrics]
                    contributing = [m for m, s in zip(metrics, signs) if s != 0]
                    non_contributing = [m for m, s in zip(metrics, signs) if s == 0]

                    if unit == "$":
                        net_earnings = sum(
                            totals.loc[m, "Total"] * metric_units[m][1]
                            for m in contributing
                        )
                        # Insert Net Earnings after last contributing metric
                        ordered_metrics = (
                            contributing + ["Net Earnings"] + non_contributing
                        )
                        # Build a new DataFrame in the correct order
                        sub_totals = sub_totals.reindex(contributing + non_contributing)
                        # Insert Net Earnings at the correct position
                        sub_totals = pd.concat(
                            [
                                sub_totals.loc[contributing],
                                pd.DataFrame(
                                    {"Total": [net_earnings]}, index=["Net Earnings"]
                                ),
                                sub_totals.loc[non_contributing],
                            ]
                        )

                    # Formatting function for negatives as (140)
                    def fmt(x):
                        if pd.isna(x):
                            return ""
                        x = round(x)
                        if x < 0:
                            return f"({abs(x):,})"
                        else:
                            return f"{x:,}"

                    # Apply formatting and right-align
                    formatted = sub_totals.applymap(fmt)
                    formatted = formatted.style.set_properties(
                        **{"text-align": "right"}
                    )

                    col.subheader(f"{unit} ")
                    col.markdown(
                        formatted.to_html(
                            classes="styled-table", escape=False, header=False
                        ),
                        unsafe_allow_html=True,
                    )
                    # if unit == "$":
                    # Optionally, bold the Net Earnings row
                    # col.markdown(
                    #     f"<div style='text-align:right; font-weight:bold;'>Net Earnings: {fmt(net_earnings)}</div>",
                    #     unsafe_allow_html=True,
                    # )

            sankey_fig, total_flow, double_counted_sum, net_flow = plot_energy_sankey(
                totals["Total"]
            )

            source = ["pv_kwh", "grid_import", "public_charge"]
            sink = [
                "consumption_kwh",
                "vehicle_consumption",
                "curtailment",
                "pv_export",
                "vehicle_export",
                "home_export",
                "home_batt_loss",
                "veh_batt_loss",
            ]
            total_source = sum(
                totals.loc[s, "Total"] for s in source if s in totals.index
            )
            total_sink = sum(totals.loc[s, "Total"] for s in sink if s in totals.index)
            st.plotly_chart(sankey_fig, use_container_width=True)
            st.write(f"Sum of flows: {total_flow:.2f} kWh")
            st.write(f"Double counted: {double_counted_sum:.2f} kWh")
            st.write(f"Net: {net_flow:.2f} kWh")
            st.write(f"Source: {total_source:.2f} kWh")
            st.write(f"Sink: {total_sink:.2f} kWh")

            if st.session_state["export_df_flag"]:
                st.write(totals)

        else:
            # default_names = [
            #     "pv_kwh",
            #     "grid_import",
            #     "veh_batt_soc",
            #     "home_batt_soc",
            #     "target_soc_home",
            #     "target_soc_vehicle",
            # ]
            default_names = [
                "pv_kwh",
                "veh_batt_soc",
                "vehicle_consumption",
                "veh_batt_discharge",
                "veh_batt_charge",
                "target_soc_vehicle",
                "vehicle_export",
            ]
            value_cols = st.multiselect(
                "Select series.  Many series can be selected - but about 6 is as many as works well ",
                value_candidates,
                default=default_names,
                # default=value_candidates[:1],
                # max_selections=3,
                key="volatility_series_select",
            )
            if chart_type in ["weekly", "single day"]:
                season = period  # period is the season string
                # value_candidates = [
                #     c
                #     for c in df.columns
                #     if c not in ["date", "hour", "season", "timestamp"]
                #     and pd.api.types.is_numeric_dtype(df[c])
                # ]
                # value_cols = value_candidates
                if value_cols and season:
                    if chart_type == "weekly":
                        # selected_date is the week start date (Sunday)
                        plot_volatility_timeseries(
                            df,
                            value_cols,
                            season,
                            week_start=selected_date,
                            chart_type="weekly",
                        )
                    elif chart_type == "single day":
                        # selected_date is the day to plot
                        plot_volatility_timeseries(
                            df,
                            value_cols,
                            season,
                            week_start=selected_date,
                            chart_type="single day",
                        )
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
