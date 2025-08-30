# %%
import os
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np
from data_in import (
    load_meter_data,
    normalise_meter_data,
    plot_daily_avg_price_per_month,
    plot_hourly_price_by_season,
    plot_hourly_price_se_by_season,
)
from charts import boxplot_interval, boxplot_aggpd_season
from synthetic import (
    generate_synthetic_driving,
    expand_trips,
    plot_plugged_in_distribution,
    pad_expanded,
    generate_synthetic_pv,
    plot_pv_seasonal_avg,
    plot_pv_histogram,
    generate_synthetic_consumption,
    plot_consumption_hourly_avg,
    plot_consumption_histogram,
)


# %%
# Season classification
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


# %%
# Load synthetic csv data
# %% Load synthetic csv data
SYNTHETIC_DIR = r"C:\Energy\V2G\data\synthetic"
NEM_DIR = r"C:\Energy\V2G\data\NEM"
df_names = ["df_drive", "df_padded", "df_pv", "df_cons", "df_all"]
dfs = {}

for name in df_names:
    path = os.path.join(SYNTHETIC_DIR, f"{name}.csv")
    dfs[name] = pd.read_csv(path)
    print(f"{name}: {dfs[name].shape}, columns: {list(dfs[name].columns)}")

path = os.path.join(NEM_DIR, f"price_all_1h.csv")
df_price = pd.read_csv(path)
df_price = df_price.rename(columns={"interval": "hour"})
df_price = df_price.rename(columns={"value": "price"})
df_price["date"] = pd.to_datetime(df_price["timestamp"]).dt.date
print(f"price{df_price.shape}, columns: {list(df_price.columns)}")
# %%
df_all = (
    df_price.merge(dfs["df_pv"], on=["date", "hour"], how="left")
    .merge(dfs["df_cons"], on=["date", "hour"], how="left")
    .merge(
        dfs["df_padded"][["date", "hour", "plugged_in"]],
        on=["date", "hour"],
        how="left",
    )
    .merge(
        dfs["df_drive"][["date", "hour", "distance_km"]],
        on=["date", "hour"],
        how="left",
    )
)

# export_df(df_all, "df_all.csv")
# %%
# modeldata = dfs["df_all"]
# modeldata[modeldata["hour"] == 12]
import os

EXPORT_DIR = r"C:\Energy\V2G\data\synthetic"


def import_df(filename):
    """Import a CSV file as a pandas DataFrame."""
    return pd.read_csv(os.path.join(EXPORT_DIR, filename))


# %%
df = import_df("df_cons.csv")
