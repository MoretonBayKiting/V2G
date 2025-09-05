import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import os


def load_meter_data(file):
    df = pd.read_csv(file)
    return df


def get_season(dt):
    month = dt.month
    if month in [12, 1, 2]:
        return "Summer"
    elif month in [3, 4, 5]:
        return "Autumn"
    elif month in [6, 7, 8]:
        return "Winter"
    else:
        return "Spring"


def normalise_meter_data(df):
    value_vars = df.columns[5:]
    df_long = df.melt(
        id_vars=["NMI", "METER SERIAL NUMBER", "CON/GEN", "DATE", "ESTIMATED?"],
        value_vars=value_vars,
        var_name="Interval",
        value_name="kWh",
    )
    df_long["DATE"] = pd.to_datetime(df_long["DATE"], dayfirst=True)
    df_long["Interval"] = df_long["Interval"].apply(
        lambda x: int(df.columns.get_loc(x) - 5)
    )
    df_long["DateTime"] = df_long["DATE"] + pd.to_timedelta(
        df_long["Interval"] * 30, unit="m"
    )
    df_long["AggPd"] = df_long["Interval"] // 12
    df_long["Season"] = df_long["DATE"].apply(get_season)
    return df_long


def plot_daily_avg_price_per_month(df_price):
    """Plot daily average price per month."""
    df_price["date"] = pd.to_datetime(df_price["timestamp"]).dt.date
    df_price["month"] = pd.to_datetime(df_price["timestamp"]).dt.month
    daily_avg = (
        df_price.groupby(["month", "date"])["price"].mean().groupby("month").mean()
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(daily_avg.index, daily_avg.values, marker="o")
    ax.set_title("Daily Average Price per Month")
    ax.set_xlabel("Month")
    ax.set_ylabel("Average Daily Price")
    ax.set_xticks(range(1, 13))
    ax.grid(True)
    plt.tight_layout()
    return fig


def plot_hourly_price_by_season(df_price):
    """Plot average hourly price for each season (4 series, hour on x axis)."""
    df_price["hour"] = pd.to_datetime(df_price["timestamp"]).dt.hour
    fig, ax = plt.subplots(figsize=(8, 5))
    for season in ["Summer", "Autumn", "Winter", "Spring"]:
        df_season = df_price[df_price["season"] == season]
        hourly_avg = df_season.groupby("hour")["price"].mean()
        ax.plot(hourly_avg.index, hourly_avg.values, marker="o", label=season)
    ax.set_title("Average Hourly Price by Season")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Average Price")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    return fig


def plot_hourly_price_se_by_season(df_price):
    """Plot standard error of hourly price for each season (4 series, hour on x axis)."""
    df_price["hour"] = pd.to_datetime(df_price["timestamp"]).dt.hour
    fig, ax = plt.subplots(figsize=(8, 5))
    for season in ["Summer", "Autumn", "Winter", "Spring"]:
        df_season = df_price[df_price["season"] == season]
        hourly_se = df_season.groupby("hour")["price"].sem()
        ax.plot(hourly_se.index, hourly_se.values, marker="o", label=season)
    ax.set_title("Standard Error of Hourly Price by Season")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Standard Error of Price")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    return fig


def get_price_file(st):
    """
    Show file uploader and return uploaded file object (or None).
    """
    return st.file_uploader("Upload Price Data CSV", type="csv", key="price_file")


def get_price_data(st, price_file):
    """
    Process uploaded price file, update session state, and display charts.
    Returns the DataFrame or None.
    """
    df_price = None
    if price_file:
        df_price = pd.read_csv(price_file)
        df_price = df_price.rename(columns={"interval": "hour"})
        df_price = df_price.rename(columns={"value": "price"})
        df_price["date"] = pd.to_datetime(df_price["timestamp"]).dt.date
        st.session_state["df_price"] = df_price
        st.write("Price Data :", df_price.head(5))
        fig_daily_avg_price = plot_daily_avg_price_per_month(df_price)
        st.pyplot(fig_daily_avg_price)
        fig_hr_seas_price = plot_hourly_price_by_season(df_price)
        st.pyplot(fig_hr_seas_price)
        fig_hr_seas_price_se = plot_hourly_price_se_by_season(df_price)
        st.pyplot(fig_hr_seas_price_se)
    return df_price


def combine_all_data(st, export=True):
    """
    Combines price, PV, consumption, and driving data from session_state.
    Stores result in st.session_state["df_all"] and optionally exports to CSV.
    Returns the combined DataFrame or None if any source is missing.
    """
    required_keys = ["df_price", "df_pv", "df_cons", "df_padded"]
    missing = [k for k in required_keys if k not in st.session_state]
    if missing:
        st.warning(f"Missing required data sources: {', '.join(missing)}")
        print(f"[combine_all_data] Missing keys: {missing}")
        return None

    df_price = st.session_state["df_price"]
    df_pv = st.session_state["df_pv"]
    df_cons = st.session_state["df_cons"]
    df_padded = st.session_state["df_padded"]

    df_price["date"] = pd.to_datetime(df_price["timestamp"]).dt.date
    df_all = (
        df_price.merge(
            df_pv[["date", "hour", "pv_kwh", "is_sunny"]],
            on=["date", "hour"],
            how="left",
        )
        .merge(
            df_cons[["date", "hour", "consumption_kwh"]],
            on=["date", "hour"],
            how="left",
        )
        .merge(
            df_padded[["date", "hour", "plugged_in", "distance_km"]],
            on=["date", "hour"],
            how="left",
        )
    )
    st.session_state["df_all"] = df_all
    return df_all


# Streamlit UI for volatility time series chart
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


def plot_volatility_timeseries(
    df, value_cols, season, week_start=None, chart_type="weekly", day=None
):
    import matplotlib.dates as mdates

    # Filter by season
    if season != "Any":
        df_season = df[df["season"] == season]
    else:
        df_season = df
    unique_dates = sorted(df_season["date"].unique())

    if chart_type == "weekly":
        plot_days = 7
        sundays = [d for d in unique_dates if pd.Timestamp(d).weekday() == 6]
        if len(sundays) == 0:
            st.warning("No Sundays found in this season's data.")
            return
            # Use provided week_start if given, else pick a random Sunday
        if week_start is not None:
            # Ensure week_start is a valid Sunday in the data
            if week_start not in sundays:
                # Find the nearest Sunday
                nearest_sunday = min(
                    sundays,
                    key=lambda s: abs(
                        (pd.Timestamp(s) - pd.Timestamp(week_start)).days
                    ),
                )
                # st.warning(
                #     f"Selected start day {week_start} is not a Sunday in this season. Using nearest Sunday: {nearest_sunday}."
                # )
                week_start = nearest_sunday
        else:
            start_idx = random.randint(0, len(sundays) - 1)
            week_start = sundays[start_idx]
    else:
        plot_days = 1
        if week_start is None:
            week_start = random.choice(unique_dates)

    plot_dates = pd.date_range(week_start, periods=plot_days).date
    df_plot = df_season[df_season["date"].isin(plot_dates)]
    timestamps = pd.to_datetime(
        df_plot["date"].astype(str) + " " + df_plot["hour"].astype(str) + ":00"
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    for col in value_cols:
        ax.plot(timestamps, df_plot[col], label=col)
        # ax.bar(timestamps, df_plot[col], label=col, alpha=0.7)

    if chart_type == "weekly":
        days = pd.to_datetime(plot_dates)
        ax.set_xticks([d for d in days])
        ax.set_xticklabels([d.strftime("%a") for d in days])
        ax.set_title(f"{season} week starting {week_start}")
        ax.set_xlabel("Day of Week")
    else:
        ax.set_xticks(timestamps[::2])  # every 2 hours for clarity
        ax.set_xticklabels([t.strftime("%H:%M") for t in timestamps[::2]])
        ax.set_title(f"{season} day {week_start}")
        ax.set_xlabel("Hour of Day")

    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    st.pyplot(fig)
    cols = ["date", "hour"] + value_cols
    st.write(df_plot[cols])

    # fig, ax = plt.subplots(figsize=(10, 4))
    # bar_width = 0.8 / len(value_cols)  # Adjust bar width for number of series
    # x = np.arange(len(timestamps))

    # for i, col in enumerate(value_cols):
    #     ax.bar(
    #         x + i * bar_width,
    #         df_plot[col].values,
    #         width=bar_width,
    #         label=col,
    #         alpha=0.7,
    #     )

    # ax.set_xticks(x + bar_width * (len(value_cols) - 1) / 2)
    # ax.set_xticklabels(
    #     [
    #         t.strftime("%a") if chart_type == "weekly" else t.strftime("%H:%M")
    #         for t in timestamps
    #     ]
    # )

    # if chart_type == "weekly":
    #     ax.set_title(f"{season} week starting {week_start}")
    #     ax.set_xlabel("Day of Week")
    # else:
    #     ax.set_title(f"{season} day {week_start}")
    #     ax.set_xlabel("Hour of Day")

    # ax.set_ylabel("Value")
    # ax.legend()
    # ax.grid(True)
    # plt.tight_layout()
    # st.pyplot(fig)


def export_df(export_flag, df, filename):
    # Detect if running on Streamlit Cloud
    # on_streamlit_cloud = (
    #     os.environ.get("STREAMLIT_SERVER_HOST") is not None
    #     or os.environ.get("STREAMLIT_CLOUD") is not None
    #     or os.environ.get("STREMLIT_CLOUD") is not None  # typo sometimes present
    #     or "streamlit" in os.environ.get("HOME", "").lower()
    # )
    # if on_streamlit_cloud:
    if export_flag:
        EXPORT_DIR = r"C:\Energy\V2G\data\synthetic"
        df.to_csv(os.path.join(EXPORT_DIR, filename), index=False)
    else:
        print(
            f"[INFO] export_df called for '{filename}', but file writing is disabled in this environment."
        )
