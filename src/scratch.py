# %%
import os
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np

# from data_in import (
#     load_meter_data,
#     normalise_meter_data,
#     plot_daily_avg_price_per_month,
#     plot_hourly_price_by_season,
#     plot_hourly_price_se_by_season,
# )
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
# # Season classification
# def get_season(dt):
#     month = dt.month
#     if month in [12, 1, 2]:
#         return "Summer"
#     elif month in [3, 4, 5]:
#         return "Autumn"
#     elif month in [6, 7, 8]:
#         return "Winter"
#     else:
#         return "Spring"


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

# export_df(True,df_all, "df_all.csv")
# %%
# modeldata = dfs["df_all"]
# modeldata[modeldata["hour"] == 12]
import os

EXPORT_DIR = r"C:\Energy\V2G\data\synthetic"


def import_df(filename):
    """Import a CSV file as a pandas DataFrame."""
    return pd.read_csv(os.path.join(EXPORT_DIR, filename))


# %%
import pandas as pd
from datetime import datetime, timedelta

df = import_df("results_df.csv")
vars = [
    "date",
    "hour",
    "pv_kwh",
    "consumption_kwh",
    "target_soc_home",
    "home_batt_soc",
    "effective_import_price",
    "effective_export_price",
]
df_trim = df[vars]
start_date = datetime(2025, 3, 5)
end_date = start_date + timedelta(days=1)
df_trim["date"] = pd.to_datetime(df_trim["date"])
rng = (df_trim["date"] >= start_date) & (df_trim["date"] < end_date)
df_week = df_trim[rng].copy()
print(df_week)
# %%
load = df_week["consumption_kwh"] * df_week["effective_import_price"]
supply = df_week["pv_kwh"] * df_week["effective_export_price"] + df_week[
    "consumption_kwh"
] * (df_week["effective_import_price"] - df_week["effective_export_price"])

lookahead_hours = 3
pad = np.zeros(lookahead_hours - 1)
load_padded = np.concatenate([load, pad])
supply_padded = np.concatenate([supply, pad])

windows_load = np.lib.stride_tricks.sliding_window_view(load_padded, lookahead_hours)
windows_supply = np.lib.stride_tricks.sliding_window_view(
    supply_padded, lookahead_hours
)

# %%
# print(windows_load)

sum_load = np.sum(windows_load, axis=1)
supply_rest = np.sum(windows_supply[:, 1:], axis=1)
targ_soc = np.maximum(windows_load[:, 0], sum_load - supply_rest)
# %%import numpy as np
import matplotlib.pyplot as plt

# Desired real-space mean and std
mean_real = 300
std_real = 100

# Convert to log-space parameters
variance_real = std_real**2
mu = np.log(mean_real**2 / np.sqrt(variance_real + mean_real**2))
sigma = np.sqrt(np.log(1 + variance_real / mean_real**2))

# Generate samples
samples = np.random.lognormal(mean=mu, sigma=sigma, size=10000)

# Plot histogram
plt.figure(figsize=(8, 4))
plt.hist(samples, bins=50, color="skyblue", edgecolor="black", alpha=0.7)
plt.title("Lognormal Distribution (mean=300, std=50)")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.show()

# Print sample mean and std for verification
print(f"Sample mean: {samples.mean():.2f}")
print(f"Sample std: {samples.std():.2f}")

# %%

# %%

# df_all = import_df("df_all.csv")
results_df = import_df("results_df.csv")
# results_df already includes all relevant fields from df_all
# test = results_df.merge(df_all, on=["date", "hour"], how="left", suffixes=("", "_y"))
# Drop all columns from df_all that have the "_y" suffix (i.e., duplicates)
# test = test[[col for col in test.columns if not col.endswith("_y")]]

sample_date = "2024-12-05"  # "2024-11-13"
test1 = results_df[results_df["date"] == sample_date]
# test1 = results_df[
#     (results_df["date"] == sample_date) | (results_df["date"] == "2024-11-13")
# ]
extra_vars = [
    "veh_batt_charge",
    "veh_batt_charge_grid",
    "veh_batt_charge_extra",
    "veh_batt_discharge",
    "vehicle_export",
    "driving_discharge",
]

test2 = test1[
    [
        "date",
        "hour",
        "consumption_kwh",
        "pv_export",
        "pv_to_consumption",
        "pv_kwh",
        "veh_batt_soc",
        "target_soc_vehicle",
        "allow_charge",
        "effective_import_price",
        "vehicle_consumption",
        "public_charge",
        "veh_batt_charge",
        "public_charge_rate",
        "no_charging",
        "positive_export_price",
        "effective_export_price",
        "idx_max_veh",
    ]
]
# test2 = test2[(test2["hour"] > 15) & (test2["hour"] < 20)]
# test2 = test2[(test2["hour"] > 5) & (test2["hour"] < 14)]

DIR = r"C:\Energy\V2G\data"
# test1.to_csv(os.path.join(DIR, "test1.csv"), index=False)
test2.to_csv(os.path.join(DIR, "test2.csv"), index=False)
df = test2


# %%
# Load the sample CSV
df = test2
#  As above but use rolling_partial_dots() (from model)
window = 12
# rng = [5:12]
p_fields = ["vehicle_consumption", "public_charge_rate"]
n_fields = ["pv_kwh", "effective_export_price", "allow_charge"]
p_sum = rolling_partial_dots(df, p_fields, window)
n_sum = rolling_partial_dots(df, n_fields, window)
weighted_arrays = []
weighted_arrays.append(p_sum)
weighted_arrays.append(-n_sum)
diff = np.sum(weighted_arrays, axis=0)
idx_max = np.argmax(diff, axis=1)

p_fields2 = ["vehicle_consumption", "no_charging"]
n_fields2 = ["pv_kwh", "positive_export_price", "allow_charge"]
p_sum = rolling_partial_dots(df, p_fields2, window)
n_sum = rolling_partial_dots(df, n_fields2, window)
unweighted_arrays = []
unweighted_arrays.append(p_sum)
unweighted_arrays.append(-n_sum)
temp = np.sum(unweighted_arrays, axis=0)
np.maximum(temp[np.arange(len(df)), idx_max], 0)
targ_soc = np.maximum(temp[np.arange(len(df)), idx_max], 0)
# targ_soc = temp[np.arange(len(df)), idx_max]

id = 7
print(f"Weighted arrays and diff")
print(weighted_arrays[0][id, :])
print(weighted_arrays[1][id, :])
print(diff[id])
print(f"Unweighted arrays  and temp (diff)")
print(unweighted_arrays[0][id, :])
print(unweighted_arrays[1][id, :])
print(temp[id])
print(f"idx_max and targ_soc")
print(idx_max)
print(targ_soc)
print(f"idx_max[]{id}]:  {idx_max[id]}")
# for i in np.arange(5):
#     print(f"targ_soc[]{id+i}]:  {targ_soc[id+i]}")
i = 6
print(f"targ_soc: {id} to {id+i} : {targ_soc[id:id+i]}")

# %%
# Test target_soc calculations
import pandas as pd
import numpy as np
from model import rolling_partial_dots, target_soc

df_all = import_df("df_all.csv")
lookahead_hours = 5
debug_date = "2024-12-05"
target_soc_vehicle, idx_max_veh, weighted_arrays, unweighted_arrays = target_soc(
    df_all,
    [
        ("p", ["vehicle_consumption", "public_charge_rate"]),
        ("n", ["pv_kwh", "effective_export_price", "allow_charge"]),
    ],
    [
        ("p", ["vehicle_consumption", "no_charging"]),
        ("n", ["pv_kwh", "positive_export_price", "allow_charge"]),
    ],
    lookahead_hours=lookahead_hours,
    debug_date=debug_date,
)
# %%

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
    "network_variable_cost": ("$", -1),
    "network_fixed_cost": ("$", -1),
    "grid_energy_cost": ("$", -1),
    "public_charge_cost": ("$", -1),
    "grid_import_cost": ("$", 0),
    "curtailment_op_cost": ("$", 0),
    # c metrics
    "grid_import_rate": ("c", 0),
    "home_export_rate": ("c", 0),
    "veh_export_rate": ("c", 0),
    "public_charge_rate": ("c", 0),
    "pv_export_rate": ("c", 0),
    "curtailment_rate": ("c", 0),
    "grid_energy_rate": ("c", 0),
}
# %%
ordered_metrics = list(metric_units.keys())
totals = import_df("totals.csv")
tables = {"kWh": [], "$": [], "c": []}
for metric in ordered_metrics:
    unit, _ = metric_units[metric]
    if metric in totals.index:
        tables[unit].append(metric)
# %%
from synthetic import (
    expand_trips,
    prepare_driving_params,
    generate_synthetic_driving,
    pad_expanded,
)

#  tripsets here is from 2Drives.json
trip_sets = [
    {
        "probability": 0.6,
        "weekday": True,
        "weekend": True,
        "distance_mean": 20.0,
        "distance_std": 6.0,
        "time_mean": 8.0,
        "time_std": 0.2,
        "length_mean": 2.0,
        "length_std": 2.0,
    },
    {
        "probability": 0.05,
        "weekday": True,
        "weekend": True,
        "distance_mean": 400.0,
        "distance_std": 0.0,
        "time_mean": 8.0,
        "time_std": 0.0,
        "length_mean": 48.0,
        "length_std": 0.0,
    },
]
df_padded, df_synth = generate_synthetic_driving(trip_sets=trip_sets)
# %%
df_all = import_df("df_all.csv")
results_df = import_df("results_df.csv")
df_padded = import_df("df_padded.csv")
df_allb4 = import_df("df_all_b4_precompute.csv")
df_drive = import_df("df_drive.csv")
df_cons = import_df("df_cons.csv")
df_pv = import_df("df_pv.csv")
df_all["timestamp"] = pd.to_datetime(df_all["timestamp"])
all_filtered = df_all[
    df_all["timestamp"].dt.date == pd.to_datetime("2025-01-16").date()
]

df_allb4["timestamp"] = pd.to_datetime(df_allb4["timestamp"])
df_all_b41 = df_allb4[
    df_allb4["timestamp"].dt.date == pd.to_datetime("2025-01-16").date()
]


def get_test_data(df):
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df["timestamp"].dt.date == pd.to_datetime("2025-01-16").date()


def get_test_data2(df):
    return df[df["date"] == "2025-01-16"]


dataframes = {}
required_keys = [
    "df_price",
    "df_pv",
    "df_cons",
    "df_padded",
]
for k in required_keys + ["df_drive_base"]:  # 20250907 Testing for duplicate records ()
    dataframes[k] = import_df(k + "1.csv")
    print(f"{k}  shape: {dataframes[k].shape}")
    # print(dataframes[k].shape)

df = dataframes["df_drive_base"]
test = df[df["distance_km"] > 240]

results_df = import_df("results_df.csv")
veh_exp_cols = [
    "vehicle_export",
    "effective_export_price",
    "date",
    "hour",
    "target_soc_vehicle",
    "veh_batt_soc",
    "price_kwh",
    "price",
]
veh_exp = results_df[results_df["vehicle_export"] > 0]
veh_exp[veh_exp_cols].sort_values(by="effective_export_price", ascending=False)


import numpy as np
import pandas as pd


def rolling_price_spread_components(df, price_col="effective_import_price", m=72, n=12):
    """
    Returns three Series: spread, highest_means, lowest_means, all aligned to df.index.
    """
    prices = df[price_col].values
    if len(prices) < m:
        nan_series = pd.Series(np.nan, index=df.index)
        return nan_series, nan_series, nan_series
    windows = np.lib.stride_tricks.sliding_window_view(prices, m)
    sorted_windows = np.sort(windows, axis=1)
    highest_means = sorted_windows[:, -n:].mean(axis=1)
    lowest_means = sorted_windows[:, :n].mean(axis=1)
    spread = highest_means - lowest_means
    # Pad to align with df.index
    pad = np.full(len(prices), np.nan)
    pad[m - 1 :] = spread
    spread_series = pd.Series(pad, index=df.index)
    pad_high = np.full(len(prices), np.nan)
    pad_high[m - 1 :] = highest_means
    highest_series = pd.Series(pad_high, index=df.index)
    pad_low = np.full(len(prices), np.nan)
    pad_low[m - 1 :] = lowest_means
    lowest_series = pd.Series(pad_low, index=df.index)
    return spread_series, highest_series, lowest_series


# Usage:
# %%
lookaheard_period = 24
number_of_periods = 6
spread, highest, lowest = rolling_price_spread_components(
    df, price_col="price", m=lookaheard_period, n=number_of_periods
)

# plt.figure(figsize=(12, 4))
# plt.plot(spread, label="Spread (High - Low)")
# plt.plot(highest, label="Mean of n Highest")
# plt.plot(lowest, label="Mean of n Lowest")
# plt.xlabel("Index")
# plt.ylabel("c/kWh")
# plt.title("Rolling Price Spread and Components")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()
# print("Mean spread:", np.nanmean(spread))
# print("Mean high:", np.nanmean(highest))
# print("Mean low:", np.nanmean(lowest))

df = results_df[["month", "price"]]
# Add the results to the DataFrame for grouping
df["spread"] = spread
df["highest"] = highest
df["lowest"] = lowest

# Group by month and calculate means (skip NaN)
monthly_means = df.groupby("month")[["spread", "highest", "lowest", "price"]].mean()
plot_path = (
    f"C:/Energy/V2G/data/processed/Spreads{lookaheard_period}_{number_of_periods}.png"
)
# Plot
plt.figure(figsize=(10, 5))
plt.plot(monthly_means.index, monthly_means["spread"], label="Spread (High - Low)")
plt.plot(
    monthly_means.index,
    monthly_means["highest"],
    label=f"Mean of {number_of_periods} Highest",
)
plt.plot(
    monthly_means.index,
    monthly_means["lowest"],
    label=f"Mean of {number_of_periods} Lowest",
)
plt.plot(monthly_means.index, monthly_means["price"], label="Mean of price")
plt.xlabel("Month")
plt.ylabel("$/MWh")
plt.title(
    f"Monthly Means of Rolling Price Spread and Components. lookaheard_period = {lookaheard_period}; number_of_periods = {number_of_periods}"
)
plt.legend()
plt.grid(True)
plt.tight_layout()
# Add mean spread as a text box in the upper left
mean_spread = np.nanmean(spread)
plt.gca().text(
    0.01,
    0.98,
    f"Mean spread: {mean_spread:.2f} c/MWh",
    transform=plt.gca().transAxes,
    fontsize=11,
    verticalalignment="top",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
)
plt.savefig(plot_path)
plt.show()

# %% Assume df has 'vehicle_export', 'home_export', 'veh_earnings', 'home_earnings', 'grid_import', 'grid_import_cost'
# Vehicle export earnings
veh_earnings = df["veh_earnings"].fillna(0)
veh_earnings = veh_earnings[veh_earnings > 0]
veh_sorted = np.sort(veh_earnings)[::-1]
veh_cum = np.cumsum(veh_sorted)
veh_cum /= veh_cum[-1]

# Home export earnings
home_earnings = df["home_earnings"].fillna(0)
home_earnings = home_earnings[home_earnings > 0]
home_sorted = np.sort(home_earnings)[::-1]
home_cum = np.cumsum(home_sorted)
if len(home_cum) > 0:
    home_cum /= home_cum[-1]

# Import costs (positive values)
import_costs = df["grid_import_cost"].fillna(0)
import_costs = import_costs[import_costs > 0]
import_sorted = np.sort(import_costs)[::-1]
import_cum = np.cumsum(import_sorted)
if len(import_cum) > 0:
    import_cum /= import_cum[-1]

plt.figure(figsize=(8, 5))
plt.plot(
    np.linspace(0, 100, len(veh_cum)), veh_cum * 100, label="Vehicle Export Earnings"
)
if len(home_cum) > 0:
    plt.plot(
        np.linspace(0, 100, len(home_cum)), home_cum * 100, label="Home Export Earnings"
    )
if len(import_cum) > 0:
    plt.plot(
        np.linspace(0, 100, len(import_cum)), import_cum * 100, label="Import Costs"
    )

plt.xlabel("% of periods (sorted by value, highest to lowest)")
plt.ylabel("Cumulative % of total")
plt.title("Concentration of Earnings and Costs by Period")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
# %% Assume df has 'price', 'vehicle_export', 'home_export', 'grid_import'
df = import_df("results_df.csv")
price_col = "price_kwh"
bin_width = 5  # c/kWh
max_price = df[price_col].max()
min_price = df[price_col].min()
print(f"min_price: {min_price:.2f}, max_price: {max_price:.2f}")
# bins = np.arange(0, max_price + bin_width, bin_width)
bins = np.arange(
    np.floor(min_price / bin_width) * bin_width, max_price + bin_width, bin_width
)

# Masks for each flow
veh_mask = df["vehicle_export"] > 0
home_mask = df["home_export"] > 0
import_mask = df["grid_import"] > 0

# Histogram: count of periods in each price bin where flow occurred
# Calculate total periods in each price bin
total_hist, _ = np.histogram(df[price_col], bins=bins)

# For each flow, count periods in bin where flow occurred
veh_hist, _ = np.histogram(df.loc[veh_mask, price_col], bins=bins)
home_hist, _ = np.histogram(df.loc[home_mask, price_col], bins=bins)
import_hist, _ = np.histogram(df.loc[import_mask, price_col], bins=bins)

# Avoid division by zero
with np.errstate(divide="ignore", invalid="ignore"):
    veh_prop = np.where(total_hist > 0, veh_hist / total_hist, np.nan)
    home_prop = np.where(total_hist > 0, home_hist / total_hist, np.nan)
    import_prop = np.where(total_hist > 0, import_hist / total_hist, np.nan)

bin_centers = (bins[:-1] + bins[1:]) / 2

plt.figure(figsize=(10, 5))
plt.step(bin_centers, veh_prop, where="mid", label="Vehicle Export", color="tab:blue")
plt.step(bin_centers, home_prop, where="mid", label="Home Export", color="tab:orange")
plt.step(bin_centers, import_prop, where="mid", label="Import", color="tab:green")
plt.xlabel("Price (c/kWh)")
plt.ylabel("Proportion of periods in bin")
plt.title("Proportion of Periods with Flow by Price Bucket")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
# %%
df_price = import_df("df_price.csv")
df_price.sort_values(by="price", ascending=False)
df = import_df("results_df.csv")
df["price", "price_kwh"].sort_values(by="price", ascending=False)
qld = import_df("Qld.csv")
qld.sort_values(by="value", ascending=False)

# %%
df = import_df("results_df.csv")
keep_fields = [
    "date",
    "hour",
    "veh_batt_charge",
    "veh_batt_discharge",
    "grid_import",
    "effective_import_price",
    "veh_earnings",
]
# test = df[(df["veh_batt_charge"] > 0) & (df["grid_import"] > 0)][keep_fields]
test = df[keep_fields]
test["veh_import"] = test["veh_batt_charge"] - np.maximum(
    0, (test["veh_batt_charge"] - test["grid_import"])
)
test["veh_import_cost"] = -test["veh_import"] * test["effective_import_price"]
# Calculate value components
test["displaced_import_value"] = (
    test["effective_import_price"] * test["veh_batt_discharge"]
)
test["veh_export_value"] = test["veh_earnings"]
# %%

threshold = 0.001
test = test.fillna(0)
value_fields = ["displaced_import_value", "veh_import_cost", "veh_export_value"]
for v in value_fields:
    if v == "veh_import_cost":
        print(f"{v} : {test[(test[v] < -threshold)][v].describe()}")
    else:
        print(f"{v} : {test[(test[v] > threshold)][v].describe()}")

test = test[
    (test["displaced_import_value"] > threshold)
    | (test["veh_export_value"] > threshold)
    | (test["veh_import_cost"] < -threshold)
]
# Net value per period
test["net_v2g_v2h_value"] = (
    test["displaced_import_value"] + test["veh_export_value"] + test["veh_import_cost"]
)

# Sort by net value
test_sorted = test.sort_values("net_v2g_v2h_value", ascending=False).reset_index(
    drop=True
)

print(test_sorted[value_fields].describe())

n_bins = 50
test_sorted["bin"] = pd.qcut(test_sorted.index, n_bins, labels=False)

binned = (
    test_sorted.groupby("bin")
    .agg(
        {
            "displaced_import_value": "sum",
            "veh_export_value": "sum",
            "veh_import_cost": "sum",
        }
    )
    .reset_index()
)
periods_per_bin = test_sorted.groupby("bin").size()
median_periods = int(periods_per_bin.median())


plt.figure(figsize=(12, 6))
plt.bar(
    binned["bin"],
    binned["displaced_import_value"],
    label="Displaced Import",
    color="tab:blue",
)
plt.bar(
    binned["bin"],
    binned["veh_export_value"],
    bottom=binned["displaced_import_value"],
    label="Vehicle Export",
    color="tab:green",
)
plt.bar(
    binned["bin"],
    binned["veh_import_cost"],
    bottom=binned["displaced_import_value"] + binned["veh_export_value"],
    label="Grid Import for Charging (Negative)",
    color="tab:red",
)
plt.xlabel("Period Bin (sorted by net value)")
plt.ylabel("Total Value ($) per bin")
plt.title("Stacked Value Duration Curve for V2G/V2H (Binned)")
plt.legend()
plt.tight_layout()
plt.gca().text(
    0.2,
    0.9,
    f"Each bar shows the total financial flow for about {median_periods} periods.\n"
    f"About {median_periods * n_bins:,.0f} periods are represented.\n"
    f"Periods where the financial flow was less than a threshold of {threshold*100}c are not included.",
    transform=plt.gca().transAxes,
    fontsize=11,
    verticalalignment="top",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
)
plt.show()

# %%
