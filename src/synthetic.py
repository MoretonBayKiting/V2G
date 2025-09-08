# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import inspect
from data_in import get_season


def autocast_params(func, params):
    """
    Casts params dict values to the types specified in func's signature.
    Only works for basic types (int, float, str, bool).
    """
    sig = inspect.signature(func)
    casted = {}
    for k, v in params.items():
        if k in sig.parameters:
            param = sig.parameters[k]
            # Get default value type if annotation is missing
            typ = (
                param.annotation
                if param.annotation != inspect._empty
                else type(param.default)
            )
            try:
                # Only cast if type is int, float, str, bool
                if typ in [int, float, str, bool]:
                    casted[k] = typ(v)
                else:
                    casted[k] = v
            except Exception:
                casted[k] = v  # fallback: leave as is
        else:
            casted[k] = v
    return casted


def prepare_driving_params(driving_raw, autocast_params, generate_synthetic_driving):
    """
    Prepare parameters for generate_synthetic_driving from scenario dict.
    - driving_raw: dict from scenario["synthetic_data_params"]["driving"]
    - autocast_params: function to cast types
    - generate_synthetic_driving: function reference
    Returns: dict of parameters for generate_synthetic_driving
    """
    trips = driving_raw.get("trips", [])
    driving_params = {k: v for k, v in driving_raw.items() if k != "trips"}
    driving_params["trip_sets"] = trips
    driving_params = autocast_params(generate_synthetic_driving, driving_params)
    return driving_params


def generate_synthetic_driving(
    n_days=365,
    trip_sets=None,
    start_date="2024-07-01",
    seed=42,
):
    """
    Generate synthetic driving data using multiple trip parameter sets.
    trip_sets: list of dicts, each with keys:
        - probability (float, 0-1)
        - weekday (bool)
        - weekend (bool)
        - distance_mean (float)
        - distance_logstd (float)
        - time_mean (float, hour of day)
        - time_std (float)
        - length_mean (float, trip duration in hours)
        - length_logstd (float)
    """
    seed = int(seed)
    np.random.seed(seed)
    hours_per_day = 24
    if trip_sets is None:
        # Fallback to a default single trip set if not provided
        trip_sets = [
            {
                "probability": 0.05,
                "weekday": True,
                "weekend": True,
                "distance_mean": 400,
                "distance_logstd": 0.0,
                "time_mean": 8,
                "time_std": 0,
                "length_mean": 36,
                "length_logstd": 0.0,
            }
        ]

    rows = []
    for day in range(n_days):
        date = pd.Timestamp(start_date) + pd.Timedelta(days=day)
        is_weekend = date.dayofweek >= 5
        trip_hours_today = []
        for trip_set in trip_sets:
            # Only consider trip sets valid for today (weekday/weekend)
            if (is_weekend and trip_set.get("weekend", False)) or (
                not is_weekend and trip_set.get("weekday", False)
            ):
                # Determine if this trip occurs today
                if np.random.rand() < trip_set.get("probability", 0):
                    # Draw trip time, distance, and period
                    trip_hour = int(
                        np.clip(
                            np.random.normal(
                                trip_set.get("time_mean", 13),
                                trip_set.get("time_std", 1),
                            ),
                            0,
                            23,
                        )
                    )
                    # Had planned to use log normal distributions - but likley too confusing for users.
                    # variance_real = trip_set.get("distance_std", 30) ** 2
                    # mean_real = trip_set.get("distance_mean", 30)
                    # mu = np.log(mean_real**2 / np.sqrt(variance_real + mean_real**2))
                    # sigma = np.sqrt(np.log(1 + variance_real / mean_real**2))
                    trip_distance = np.clip(
                        np.random.normal(
                            loc=trip_set.get("distance_mean", 30),
                            scale=trip_set.get("distance_std", 30),
                        ),
                        # np.random.lognormal(mean=mu, sigma=sigma),
                        # np.random.lognormal(
                        #     np.log(trip_set.get("distance_mean", 30)),
                        #     trip_set.get("distance_std", 0.4),
                        # ),
                        2,
                        500,
                    )
                    trip_period = np.clip(
                        np.random.normal(
                            loc=trip_set.get("length_mean", 2),
                            scale=trip_set.get("length_std", 0.5),
                        ),
                        0.5,
                        60,
                    )
                    trip_hours_today.append(trip_hour)
                    rows.append(
                        {
                            "date": date.date(),
                            "hour": trip_hour,
                            "distance_km": round(trip_distance, 1),
                            "trip_period_hr": round(trip_period, 2),
                            "plugged_in": 0,
                        }
                    )
    # Fill in plugged_in=1 for all other hours
    for h in range(hours_per_day):
        if h not in trip_hours_today:
            rows.append(
                {
                    "date": date.date(),
                    "hour": h,
                    "distance_km": 0,
                    "trip_period_hr": 0,
                    "plugged_in": 1,
                }
            )
    df_synth = pd.DataFrame(rows)
    # filtered_synth = filter_overlapping_trips(
    #     df_synth, n_days=365, start_date="2024-07-01"
    # )
    # df_expanded = expand_trips(filtered_synth[filtered_synth["distance_km"] > 0])
    df_expanded = expand_trips(df_synth[df_synth["distance_km"] > 0])
    print(f"df_expanded.shape: {df_expanded.shape}")
    df_expanded = df_expanded.groupby(["date", "hour"], as_index=False).agg(
        {
            "distance_km": "sum",
            "plugged_in": "min",  # 0 if any trip is away, else 1
            # Add other fields as needed, e.g.:
            # "vehicle_consumption": "sum",
        }
    )
    print(f"df_expanded.shape after agg: {df_expanded.shape}")
    print(f"df_expanded.columns after agg: {df_expanded.columns}")
    df_padded = pad_expanded(df_expanded, start_date=start_date, n_days=n_days)
    print(f"df_padded.shape: {df_padded.shape}")
    df_padded["season"] = df_padded["date"].apply(get_season)
    df_padded = df_padded.sort_values(["date", "hour"]).reset_index(drop=True)
    return df_padded, df_synth


# def filter_overlapping_trips(df_synth, n_days=365, start_date="2024-07-01"):
#     """Remove overlapping trips from df_synth, keeping the longest trip in case of overlap."""
#     df = df_synth.copy()
#     # Add a unique index for reference
#     df["trip_idx"] = df.index
#     # Calculate absolute start and end hour for each trip
#     base_time = pd.Timestamp(start_date)
#     df["start_dt"] = pd.to_datetime(df["date"]) + pd.to_timedelta(df["hour"], unit="h")
#     df["start_hour_abs"] = (
#         (df["start_dt"] - base_time).dt.total_seconds() // 3600
#     ).astype(int)
#     df["end_hour_abs"] = (df["start_hour_abs"] + df["trip_period_hr"]).astype(int)

#     # Sort by start time, then by distance_km descending (prefer longer trips)
#     df = df.sort_values(
#         ["start_hour_abs", "distance_km"], ascending=[True, False]
#     ).reset_index(drop=True)

#     claimed_hours = set()
#     keep_idxs = []
#     for _, row in df.iterrows():
#         trip_hours = set(range(int(row["start_hour_abs"]), int(row["end_hour_abs"])))
#         if claimed_hours.isdisjoint(trip_hours):
#             keep_idxs.append(row["trip_idx"])
#             claimed_hours.update(trip_hours)
#         # else: skip this trip (it overlaps with a previous one)

#     # Return only non-overlapping trips
#     filtered = df_synth.loc[keep_idxs].reset_index(drop=True)
#     return filtered


#  --- Function to expand trips over multiple hours ---
def expand_trips(df_trip):
    """Expand each trip over its duration, supporting multi-day trips."""
    expanded_rows = []
    for _, row in df_trip.iterrows():
        start_dt = pd.Timestamp(row["date"]) + pd.Timedelta(hours=int(row["hour"]))
        period = row["trip_period_hr"]
        whole_hours = int(period)
        remainder = period - whole_hours
        distance_per_hour = row["distance_km"] / period if period > 0 else 0

        # Expand over whole hours (can cross days)
        for i in range(whole_hours):
            dt = start_dt + pd.Timedelta(hours=i)
            expanded_rows.append(
                {
                    "date": dt.date(),
                    "hour": dt.hour,
                    "distance_km": round(distance_per_hour, 2),
                    "plugged_in": 0,
                }
            )
        # Partial last hour (if any)
        if remainder > 0:
            dt = start_dt + pd.Timedelta(hours=whole_hours)
            expanded_rows.append(
                {
                    "date": dt.date(),
                    "hour": dt.hour,
                    "distance_km": round(distance_per_hour * remainder, 2),
                    "plugged_in": round(1 - remainder, 2),
                }
            )
    return pd.DataFrame(expanded_rows)


def pad_expanded(df_expanded, start_date, n_days):
    """Ensure every hour of every day is present in df_expanded, filling missing with plugged_in=1."""
    all_hours = pd.DataFrame(
        [
            {
                "date": (pd.Timestamp(start_date) + pd.Timedelta(days=day)).date(),
                "hour": hour,
            }
            for day in range(n_days)
            for hour in range(24)
        ]
    )
    df_padded = pd.merge(all_hours, df_expanded, on=["date", "hour"], how="left")
    df_padded["plugged_in"] = df_padded["plugged_in"].fillna(1)
    df_padded["distance_km"] = df_padded["distance_km"].fillna(0)
    return df_padded


def plot_plugged_in_distribution(df_expanded):
    """Return matplotlib figure for mean probability of being plugged in/away by hour of day."""
    import matplotlib.pyplot as plt

    mean_plugged = df_expanded.groupby("hour")["plugged_in"].mean()
    mean_away = 1 - mean_plugged

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(mean_plugged.index, mean_plugged.values, label="Plugged In", marker="o")
    ax.plot(mean_away.index, mean_away.values, label="Away from Home", marker="o")
    ax.set_title("Probability of Vehicle Plugged In / Away by Hour of Day")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Probability")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    return fig


def show_driving_summary(df_padded, df_drive_base, st):
    """
    Display summary, charts, and download widgets for synthetic driving data in Streamlit.
    """
    if df_padded is None or df_padded.empty or "distance_km" not in df_padded.columns:
        st.error(
            "No driving data available or 'distance_km' column missing in df_padded."
        )
        return
    if df_drive_base is None or df_drive_base.empty:
        st.error("df_drive_base empty or missing.")
        return
    else:
        if "distance_km" not in df_drive_base.columns:
            st.error("df_drive_base empty or missing.")
            return

    # st.write("Synthetic Driving Data:", df_padded.head(5))
    total_km = df_padded["distance_km"].sum()
    st.write(
        f"Total km travelled in the year: {total_km:.1f} km ({round(total_km/1000, 1)}k)"
    )

    fig = plot_plugged_in_distribution(df_padded)
    st.pyplot(fig)

    csv = df_padded.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download padded Driving CSV", csv, "padded_driving.csv", "text/csv"
    )

    fig, ax = plt.subplots()
    df_trip = df_drive_base[df_drive_base["distance_km"] > 0]
    ax.hist(df_trip["distance_km"], bins=30, color="skyblue", edgecolor="black")
    ax.set_title("Trip Distances - (return to home)")
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)


# %%
def generate_synthetic_pv(
    n_days=365,
    capacity_kw=10,
    summer_gen_factor=6,  # kWh/kW for a full sun summer day
    winter_gen_factor=4,  # kWh/kW for a full sun winter day
    sunny_prob=0.6,
    cloudy_mean_frac=0.6,
    cloudy_std_frac=0.2,
    seed=42,
    start_date="2024-07-01",
):
    """
    Generate synthetic hourly PV generation (kWh) for a year.
    Returns a DataFrame with columns: date, hour, pv_kwh, is_sunny
    """
    np.random.seed(seed)
    days = [pd.Timestamp(start_date) + pd.Timedelta(days=i) for i in range(n_days)]
    daylight_hours = list(range(8, 16))
    n_hours = len(daylight_hours)
    theta = np.linspace(-np.pi / 2, np.pi / 2, n_hours)
    hour_factors = np.cos(theta)  # peak at noon, zero at ends
    hour_factors = hour_factors / hour_factors.sum()  # normalize to sum=1

    rows = []
    for day in days:
        day_of_year = day.dayofyear
        seasonal = (
            winter_gen_factor
            + (summer_gen_factor - winter_gen_factor)
            * (1 + np.cos(2 * np.pi * (day_of_year - 15) / 365))
            / 2
        )
        if np.random.rand() < sunny_prob:
            gen_total = capacity_kw * seasonal
            is_sunny = 1
        else:
            frac = np.clip(
                np.random.normal(cloudy_mean_frac, cloudy_std_frac), 0.2, 1.0
            )
            gen_total = capacity_kw * seasonal * frac
            is_sunny = 0
        # Distribute gen_total over daylight hours
        for i, hour in enumerate(daylight_hours):
            pv_kwh = gen_total * hour_factors[i]
            rows.append(
                {
                    "date": day.date(),
                    "hour": hour,
                    "pv_kwh": round(pv_kwh, 3),
                    "is_sunny": is_sunny,
                }
            )
        # Night hours: zero generation
        for hour in range(24):
            if hour not in daylight_hours:
                rows.append(
                    {
                        "date": day.date(),
                        "hour": hour,
                        "pv_kwh": 0.0,
                        "is_sunny": is_sunny,
                    }
                )
    df_pv = pd.DataFrame(rows)
    df_pv["season"] = df_pv["date"].apply(get_season)
    df_pv = df_pv.sort_values(["date", "hour"]).reset_index(drop=True)
    return df_pv


def plot_pv_seasonal_avg(df_pv):
    """Return fig for average daily PV generation by month (aggregated from hourly)."""
    import matplotlib.pyplot as plt

    # Aggregate hourly to daily
    df_daily = df_pv.groupby("date", as_index=False)["pv_kwh"].sum()
    df_daily["month"] = pd.to_datetime(df_daily["date"]).dt.month
    monthly_avg = df_daily.groupby("month")["pv_kwh"].mean()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(monthly_avg.index, monthly_avg.values, marker="o")
    ax.set_title("Average Daily PV Generation by Month")
    ax.set_xlabel("Month")
    ax.set_ylabel("Average Daily Generation (kWh)")
    ax.set_xticks(range(1, 13))
    ax.grid(True)
    plt.tight_layout()
    return fig


def plot_pv_histogram(df_pv):
    """Return fig for histogram of daily PV generation over the year (aggregated from hourly)."""
    import matplotlib.pyplot as plt

    # Aggregate hourly to daily
    df_daily = df_pv.groupby("date", as_index=False)["pv_kwh"].sum()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df_daily["pv_kwh"], bins=30, color="gold", edgecolor="black")
    ax.set_title("Histogram of Daily PV Generation")
    ax.set_xlabel("Daily Generation (kWh)")
    ax.set_ylabel("Frequency")
    plt.tight_layout()
    return fig


def pv_summary(df_pv, st):
    st.session_state["df_pv"] = df_pv
    # st.write("Synthetic PV Data :", df_pv.head(5))
    st.write(f"Total PV generation: {df_pv['pv_kwh'].sum():.1f} kWh")
    st.write(f"Average daily PV generation: {df_pv['pv_kwh'].mean():.2f} kWh")

    # Charts
    fig_seasonal = plot_pv_seasonal_avg(df_pv)
    st.pyplot(fig_seasonal)
    fig_hist = plot_pv_histogram(df_pv)
    st.pyplot(fig_hist)

    # Download
    csv = df_pv.to_csv(index=False).encode("utf-8")
    st.download_button("Download Synthetic PV CSV", csv, "synthetic_pv.csv", "text/csv")


def generate_synthetic_consumption(
    n_days=365,
    base_avg=0.3,  # kWh per hour (fridges etc)
    base_std=0.2,  # standard deviation for base load
    evening_peak_kwh=2.0,  # kWh per hour for evening peak
    evening_peak_std=0.5,  # std for evening peak
    morning_peak_kwh=2.0,  # kWh for morning peak
    morning_peak_std=0.5,  # std for morning peak
    seed=42,
    start_date="2024-07-01",
):
    """
    Generate synthetic hourly household consumption for a year.
    Returns a DataFrame with columns: date, hour, consumption_kwh
    """
    np.random.seed(seed)
    rows = []
    for day in range(n_days):
        date = pd.Timestamp(start_date) + pd.Timedelta(days=day)
        for hour in range(24):
            # Base load
            consumption = np.random.normal(base_avg, base_std)
            # Evening peak: 6pm and 7pm (18, 19)
            if hour in [18, 19]:
                consumption += np.random.normal(evening_peak_kwh, evening_peak_std)
            # Morning peak: 7am (7)
            if hour == 7:
                consumption += np.random.normal(morning_peak_kwh, morning_peak_std)
            rows.append(
                {
                    "date": date.date(),
                    "hour": hour,
                    "consumption_kwh": max(consumption, 0),  # no negative consumption
                }
            )
    df_cons = pd.DataFrame(rows)
    df_cons["season"] = df_cons["date"].apply(get_season)
    return df_cons


def plot_consumption_hourly_avg(df_cons):
    """Return fig for average hourly household consumption over the year."""
    import matplotlib.pyplot as plt

    hourly_avg = df_cons.groupby("hour")["consumption_kwh"].mean()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(hourly_avg.index, hourly_avg.values, marker="o")
    ax.set_title("Average Hourly Household Consumption")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Average Consumption (kWh)")
    ax.grid(True)
    plt.tight_layout()
    return fig


def plot_consumption_histogram(df_cons):
    """Return fig for histogram of hourly household consumption."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df_cons["consumption_kwh"], bins=30, color="lightgreen", edgecolor="black")
    ax.set_title("Histogram of Hourly Household Consumption")
    ax.set_xlabel("Consumption (kWh)")
    ax.set_ylabel("Frequency")
    plt.tight_layout()
    return fig


def cons_summary(df_cons, st):
    st.session_state["df_cons"] = df_cons
    # st.write("Synthetic Consumption Data ", df_cons.head(5))
    st.write(f"Total consumption: {df_cons['consumption_kwh'].sum():.1f} kWh")
    st.write(
        f"Average daily consumption: {df_cons.groupby('date')['consumption_kwh'].sum().mean():.2f} kWh"
    )

    # Charts
    fig_hourly = plot_consumption_hourly_avg(df_cons)
    st.pyplot(fig_hourly)
    fig_hist = plot_consumption_histogram(df_cons)
    st.pyplot(fig_hist)

    # Download
    csv = df_cons.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Synthetic Consumption CSV",
        csv,
        "synthetic_consumption.csv",
        "text/csv",
    )


def initialize_from_scenario(
    st,
    scenario_path,
    price_path,
    export_df,
    autocast_params,
    generate_synthetic_driving,
    generate_synthetic_pv,
    generate_synthetic_consumption,
    Battery,
    Grid,
    used_battery_args,
):
    import json
    import pandas as pd

    print(f"[DEBUG] Loading scenario from: {scenario_path}")
    with open(scenario_path, "r") as f:
        scenario = json.load(f)
    print(f"[DEBUG] Scenario keys: {list(scenario.keys())}")
    st.session_state["scenario"] = scenario
    # if "global" in scenario:
    global_params = scenario["system_params"]["global"]
    st.session_state["export_df_flag"] = global_params.get("export_df_flag", False)
    st.session_state["public_charge_rate"] = global_params.get(
        "public_charge_rate", False
    )

    # --- Synthetic data ---
    gen_params = scenario.get("synthetic_data_params", {})
    print(f"[DEBUG] synthetic_data_params keys: {list(gen_params.keys())}")
    pv_params = autocast_params(generate_synthetic_pv, gen_params.get("pv", {}))
    df_pv = generate_synthetic_pv(**pv_params)
    st.session_state["df_pv"] = df_pv
    export_df(st.session_state["export_df_flag"], df_pv, "df_pv.csv")

    driving_raw = gen_params.get("driving", {})
    print(f"[DEBUG] Driving raw: {driving_raw}")
    driving_params = prepare_driving_params(
        driving_raw, autocast_params, generate_synthetic_driving
    )
    print(f"[DEBUG] Driving params for generator: {driving_params}")
    try:
        df_padded, df_drive_base = generate_synthetic_driving(**driving_params)
        st.session_state["df_padded"] = df_padded
        st.session_state["df_drive_base"] = df_drive_base
        export_df(st.session_state["export_df_flag"], df_padded, "df_padded.csv")
        export_df(
            st.session_state["export_df_flag"], df_drive_base, "df_drive_base.csv"
        )
    except Exception as e:
        st.error(f"Error generating synthetic driving data: {e}")
        st.session_state["df_padded"] = pd.DataFrame()
        st.session_state["df_drive_base"] = pd.DataFrame()

    cons_params = autocast_params(
        generate_synthetic_consumption, gen_params.get("consumption", {})
    )
    df_cons = generate_synthetic_consumption(**cons_params)
    st.session_state["df_cons"] = df_cons
    export_df(st.session_state["export_df_flag"], df_cons, "df_cons.csv")

    # --- Price data ---
    if price_path and os.path.exists(price_path):
        df_price = pd.read_csv(price_path)
        df_price = df_price.rename(columns={"interval": "hour"})
        df_price = df_price.rename(columns={"value": "price"})
        df_price["date"] = pd.to_datetime(df_price["timestamp"]).dt.date
        st.session_state["df_price"] = df_price
        export_df(st.session_state["export_df_flag"], df_price, "df_price.csv")

    # --- System params ---
    sys_params = scenario.get("system_params", {})
    home_batt_params = sys_params.get("home_battery", {})
    if home_batt_params:
        battery_args = {
            k: home_batt_params[k] for k in used_battery_args if k in home_batt_params
        }
        st.session_state["home_battery"] = Battery(**battery_args)
    vehicle_batt_params = sys_params.get("vehicle_battery", {})
    if vehicle_batt_params:
        battery_args = {
            k: vehicle_batt_params[k]
            for k in used_battery_args
            if k in vehicle_batt_params
        }
        st.session_state["vehicle_battery"] = Battery(**battery_args)
    grid_params = sys_params.get("grid", {})
    if grid_params:
        grid_args = {
            k: grid_params[k]
            for k in [
                "network_cost_import_per_kwh",
                "network_cost_export_per_kwh",
                "daily_fee",
                "max_export_kw",
            ]
            if k in grid_params
        }
        st.session_state["grid"] = Grid(**grid_args)
    global_params = sys_params.get("global", {})
    for k in ["kwh_per_km", "min_price_threshold", "start_date"]:
        if k in global_params:
            st.session_state[k] = global_params[k]

    print("[DEBUG] Scenario initialization complete.")
    st.session_state["model_dirty"] = True
    return scenario
