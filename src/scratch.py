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
