import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from data_in import export_df, get_season


def tariff_ui(tariff_periods):
    st.markdown(
        """
        **Tariff Periods:**  
        - Each entry has {start_hour, rate}. Start at 0 and don't go past 23.
        - Periods are assumed ordered by start_hour; the start_hour of one is the end_hour of the previous.
        - Export price is set by system_params.grid.fit.
        """
    )
    # Use a working copy in session_state
    if (
        "tariff_periods_working" not in st.session_state
        or st.session_state["tariff_periods_working"] is None
    ):
        st.session_state["tariff_periods_working"] = (
            list(tariff_periods) if tariff_periods else []
        )

    periods = st.session_state["tariff_periods_working"]

    st.subheader("Edit Tariff Periods")
    edited_periods = []
    for i, period in enumerate(periods):
        cols = st.columns([1, 2, 2, 1])
        with cols[0]:
            st.write(f"Period {i}")
        with cols[1]:
            start_hour = st.number_input(
                f"Start Hour {i}",
                min_value=0,
                max_value=23,
                value=int(period.get("start_hour", 0)),
                key=f"start_hour_{i}",
            )
        with cols[2]:
            rate = st.number_input(
                f"Rate (c/kWh) {i}",
                min_value=0.0,
                value=float(period.get("rate", 0.0)),
                key=f"rate_{i}",
            )
        with cols[3]:
            if st.button(f"Delete", key=f"delete_{i}"):
                periods.pop(i)
                st.session_state["tariff_periods_working"] = periods
                st.rerun()
        edited_periods.append({"start_hour": start_hour, "rate": rate})

    # Add new period
    st.markdown("---")
    cols = st.columns([2, 2, 1])
    with cols[0]:
        new_start_hour = st.number_input(
            "New Start Hour", min_value=0, max_value=23, value=0, key="new_start_hour"
        )
    with cols[1]:
        new_rate = st.number_input(
            "New Rate (c/kWh)", min_value=0.0, value=0.0, key="new_rate"
        )
    with cols[2]:
        if st.button("Add New Tariff Period", key="add_new_period"):
            periods.append({"start_hour": new_start_hour, "rate": new_rate})
            st.session_state["tariff_periods_working"] = periods
            st.rerun()

    # Save all changes
    if st.button("Save Tariff Changes", key="save_tariff_changes"):
        st.session_state["scenario"]["synthetic_data_params"]["tariff"] = (
            edited_periods if edited_periods else periods
        )
        st.session_state["tariff_periods_working"] = (
            edited_periods if edited_periods else periods
        )
        st.success("Tariff periods updated.")
        st.write(st.session_state["scenario"]["synthetic_data_params"]["tariff"])
        # Optionally regenerate price series
        start_date = st.session_state.get("start_date", "2024-07-01")
        n_days = st.session_state["scenario"]["synthetic_data_params"].get(
            "n_days", 365
        )
        df_price = generate_synthetic_tariff_price_df(
            st.session_state["scenario"]["synthetic_data_params"]["tariff"],
            start_date=start_date,
            n_days=n_days,
        )
        st.session_state["df_price"] = df_price
        export_df(st.session_state["export_df_flag"], df_price, "df_price.csv")

    # Show current periods and chart
    if periods:
        hourly_price = np.zeros(24)
        if edited_periods:
            periods = edited_periods
        sorted_periods = sorted(periods, key=lambda p: p["start_hour"])
        for i, period in enumerate(sorted_periods):
            start = period["start_hour"]
            end = (
                sorted_periods[i + 1]["start_hour"]
                if i + 1 < len(sorted_periods)
                else 24
            )
            hourly_price[start:end] = period["rate"]

        fig, ax = plt.subplots(figsize=(7, 2.5))
        ax.step(
            range(25),
            np.append(hourly_price, hourly_price[-1]),
            where="post",
            label="Tariff Rate",
        )
        ax.set_xticks(range(0, 25, 2))
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Rate (c/kWh)")
        ax.set_title("Tariff Profile (24h)")
        ax.grid(True, axis="x", linestyle="--", alpha=0.5)
        st.pyplot(fig)
        st.write("Current Tariff Periods:", sorted_periods)
    else:
        st.info("No tariff periods defined. Add one above.")

    return periods


def tariff_ui_old(tariff_periods):
    st.markdown(
        """
        **Tariff Periods:**  
        - For most use of this model, a price file, with wholesale NEM/SWIS prices, will be used.  But at least for comparison it will be useful
          to consider conventional, including ToU, tariffs.  Use this section to setup a tariff schedule.  Each entry has {start_hour, rate}. 
          The start_hour of one pair is the end_hour of the previous.  There's limited validation of hours.  Start at 0 and don't go past 23. 
          If a non-Amber style tariff is used, the export price is the fit specified with other system_params.grid parameters.
        - Somewhere, probably in the sidebar, you'll have the option to choose to use either the tariff schedule or a pricing file.
    """
    )
    if not tariff_periods:
        tariff_periods = []

    cols = st.columns(3)
    # Get current period values if editing an existing period
    with cols[0]:
        period_index = st.number_input(
            "Tariff Period Index (base 0)",
            min_value=0,
            max_value=max(0, len(tariff_periods)),
            value=0,
        )

        if period_index < len(tariff_periods):
            period = tariff_periods[period_index]
        else:
            period = {}
    with cols[1]:
        start_hour = st.number_input(
            "Start Hour (0-23)",
            min_value=0,
            max_value=23,
            value=int(period.get("start_hour", 0)),
        )
    with cols[2]:
        rate = st.number_input(
            "Rate (c/kWh)", min_value=0.0, value=float(period.get("rate", 0.0))
        )

    cols = st.columns(3)
    with cols[0]:
        if st.button("Write/Update Tariff Period"):
            new_period = {"start_hour": start_hour, "rate": rate}
            if period_index < len(tariff_periods):
                tariff_periods[period_index] = new_period
                st.success(f"Tariff period {period_index} updated.")
            else:
                tariff_periods.append(new_period)
                st.success(f"Tariff period {len(tariff_periods)-1} added.")

    with cols[1]:
        if st.button("Delete Tariff Period"):
            if 0 <= period_index < len(tariff_periods):
                tariff_periods.pop(period_index)
                st.success(f"Tariff period {period_index} deleted.")
            else:
                st.warning("Invalid period index for deletion.")
    with cols[2]:
        if st.button("Save tariff changes & view series"):
            tariff_periods = st.session_state["scenario"]["synthetic_data_params"][
                "tariff"
            ]
            start_date = st.session_state.get("start_date", "2024-07-01")
            n_days = st.session_state["scenario"]["synthetic_data_params"].get(
                "n_days", 365
            )
            df_price = generate_synthetic_tariff_price_df(
                tariff_periods, start_date=start_date, n_days=n_days
            )
            st.session_state["df_price"] = df_price
            export_df(st.session_state["export_df_flag"], df_price, "df_price.csv")

    if tariff_periods:
        # Build hourly price lookup for 24 hours
        hourly_price = np.zeros(24)
        sorted_periods = sorted(tariff_periods, key=lambda p: p["start_hour"])
        for i, period in enumerate(sorted_periods):
            start = period["start_hour"]
            end = (
                sorted_periods[i + 1]["start_hour"]
                if i + 1 < len(sorted_periods)
                else 24
            )
            hourly_price[start:end] = period["rate"]

        fig, ax = plt.subplots(figsize=(7, 2.5))
        ax.step(
            range(25),
            np.append(hourly_price, hourly_price[-1]),
            where="post",
            label="Tariff Rate",
        )
        ax.set_xticks(range(0, 25, 2))
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Rate (c/kWh)")
        ax.set_title("Tariff Profile (24h)")
        ax.grid(True, axis="x", linestyle="--", alpha=0.5)
        st.pyplot(fig)
    # Optionally, show all periods
    # st.write("Current Tariff Periods:", tariff_periods)
    return tariff_periods


def generate_synthetic_tariff_price_df(
    tariff_periods, start_date="2024-07-01", n_days=365, timezone="Australia/Sydney"
):
    """
    Generate a synthetic price DataFrame from a list of tariff periods.
    Each period should be a dict with at least 'start_hour' and 'rate'.
    The periods should cover 0-24h, ordered by start_hour.

    Args:
        tariff_periods (list): List of dicts, each with 'start_hour' (int 0-23) and 'rate' (float).
        start_date (str): Start date in 'YYYY-MM-DD' format.
        n_days (int): Number of days to generate.
        timezone (str): Timezone for datetime index.

    Returns:
        pd.DataFrame: DataFrame with columns ['date', 'hour', 'price'].
    """
    # Build hourly price lookup for 24 hours
    hourly_price = np.zeros(24)
    sorted_periods = sorted(tariff_periods, key=lambda p: p["start_hour"])
    for i, period in enumerate(sorted_periods):
        start = period["start_hour"]
        end = sorted_periods[i + 1]["start_hour"] if i + 1 < len(sorted_periods) else 24
        hourly_price[start:end] = period["rate"]

    # Build date/hour grid
    dates = pd.date_range(start=start_date, periods=n_days, freq="D", tz=timezone)
    df = pd.DataFrame(
        [(d.date(), h, hourly_price[h]) for d in dates for h in range(24)],
        columns=["date", "hour", "price"],
    )
    return convert_synthetic_tariff_df(df)


def convert_synthetic_tariff_df(df):
    """
    Convert a synthetic tariff DataFrame to the standard format expected by plotting functions.
    Ensures columns: timestamp, price, season, hour, date.
    Assumes df has at least 'date', 'hour', and 'price' columns.
    """
    df = df.copy()
    # Create timestamp if missing
    if "timestamp" not in df.columns:
        df["timestamp"] = pd.to_datetime(df["date"].astype(str)) + pd.to_timedelta(
            df["hour"], unit="h"
        )
    # Ensure price column is named 'price'
    if "price" not in df.columns and "value" in df.columns:
        df = df.rename(columns={"value": "price"})
    # Add season if missing
    if "season" not in df.columns:
        df["season"] = pd.to_datetime(df["timestamp"]).apply(get_season)
    # Add hour if missing
    if "hour" not in df.columns:
        df["hour"] = pd.to_datetime(df["timestamp"]).dt.hour
    # Add date if missing
    if "date" not in df.columns:
        df["date"] = pd.to_datetime(df["timestamp"]).dt.date
    return df


def consumption_ui(consumption_activities):
    st.markdown(
        """
    **Consumption Activities:**  
    - *start_hour*: Hour of day activity starts (0-23)  
    - *rate/rate_sd*: Power draw in kW  (and standard deviation)
    - *length/length_sd*: Duration in hours (and standard deviation)
    - *weekday/weekend*: Whether activity occurs on those days  
    """
    )
    activities = consumption_activities if consumption_activities is not None else []
    activity_index = st.number_input(
        "Activity Index (base 0). Choose or specify",
        min_value=0,
        max_value=max(0, len(activities)),
        value=0,
        key="cons_activity_index",
    )

    # Get current activity values if editing an existing activity
    if activity_index < len(activities):
        activity = activities[activity_index]
    else:
        activity = {}

    cols = st.columns(4)
    with cols[0]:
        start_hour = st.number_input(
            "Start Hour",
            min_value=0,
            max_value=23,
            value=int(activity.get("start_hour", 0)),
            key="cons_start_hour",
        )
    with cols[1]:
        rate = st.number_input(
            "Rate (kW)", value=float(activity.get("rate", 1.0)), key="cons_rate"
        )
        rate_sd = st.number_input(
            "Rate SD (kW)",
            value=float(activity.get("rate_sd", 1.0)),
            key="cons_rate_sd",
        )

    with cols[2]:
        length = st.number_input(
            "Length (hours)",
            value=float(activity.get("length", 1.0)),
            key="cons_length",
        )
        length_sd = st.number_input(
            "Length Std Dev",
            value=float(activity.get("length_sd", 0.0)),
            key="cons_length_sd",
        )
    with cols[3]:
        weekday = st.checkbox(
            "Weekday", value=bool(activity.get("weekday", 1)), key="cons_weekday"
        )
        weekend = st.checkbox(
            "Weekend", value=bool(activity.get("weekend", 1)), key="cons_weekend"
        )

    cols2 = st.columns(2)
    with cols2[0]:
        if st.button("Write/Update Activity", key="cons_write_btn"):
            new_activity = {
                "start_hour": int(start_hour),
                "rate": float(rate),
                "rate_sd": float(rate_sd),
                "length": float(length),
                "length_sd": float(length_sd),
                "weekday": int(weekday),
                "weekend": int(weekend),
            }
            if activity_index < len(activities):
                activities[activity_index] = new_activity
                st.success(f"Activity {activity_index} updated.")
            else:
                activities.append(new_activity)
                st.success(f"Activity {len(activities)-1} added.")
    with cols2[1]:
        if st.button("Delete Activity", key="cons_delete_btn"):
            if 0 <= activity_index < len(activities):
                activities.pop(activity_index)
                st.success(f"Activity {activity_index} deleted.")
            else:
                st.warning("Invalid activity index for deletion.")

    # Show all activities
    st.subheader("Current Activities")
    for i, act in enumerate(activities):
        st.write(f"**Activity {i}:** {act}")

    return activities
