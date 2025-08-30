# Set your API key
# %%
import os
import asyncio

# export OPENELECTRICITY_API_KEY=your-api-key
os.environ["OPENELECTRICITY_API_KEY"] = (
    "oe_3ZMnmVc4Dk7inpUS7jWA9Mpb"  # "your-api-key"  Valid for 30 days from 20250520
)
# Optional: Override API server (defaults to production)
# export OPENELECTRICITY_API_URL=http://localhost:8000/v4
# %%
import requests
import os

# 20250827 Use the API key you have saved to test key/access
api_key = os.environ.get("OPENELECTRICITY_API_KEY", "oe_3ZMnmVc4Dk7inpUS7jWA9Mpb")
headers = {"Authorization": f"Bearer {api_key}"}

response = requests.get("https://api.openelectricity.org.au/v4/me", headers=headers)
print("Status code:", response.status_code)
print("Response:", response.json())
# %%
import os

os.environ["OPENELECTRICITY_API_KEY"] = "oe_3ZMnmVc4Dk7inpUS7jWA9Mpb"
from datetime import datetime, timedelta
from openelectricity import OEClient

# 20250520.  Neither UnitFueltechType nor UnitStatusType are in openelectricity.types
from openelectricity.types import DataMetric, UnitFueltechType, UnitStatusType

# %%
# Calculate date range
end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
start_date = end_date - timedelta(days=7)

# Using context manager (recommended)
with OEClient() as client:
    # Get operating solar and wind facilities
    facilities = client.get_facilities(
        network_id=["NEM"],
        status_id=[UnitStatusType.OPERATING],
        fueltech_id=[UnitFueltechType.SOLAR_UTILITY, UnitFueltechType.WIND],
    )

    # Get network data for NEM
    response = client.get_network_data(
        network_code="NEM",
        metrics=[DataMetric.POWER, DataMetric.ENERGY],
        interval="1d",
        date_start=start_date,
        date_end=end_date,
        secondary_grouping="fueltech_group",
    )

    # Print results
    for series in response.data:
        print(f"\nMetric: {series.metric}")
        print(f"Unit: {series.unit}")

        for result in series.results:
            print(f"\n  {result.name}:")
            print(f"  Fuel Tech Group: {result.columns.fueltech_group}")
            for point in result.data:
                print(f"    {point.timestamp}: {point.value:.2f} {series.unit}")
# %%
# import openelectricity.types
# print(dir(openelectricity.types))
# print(openelectricity.__version__)
# %% 20250827 Get Qld wholesale price data
import os

os.environ["OPENELECTRICITY_API_KEY"] = "oe_3ZMnmVc4Dk7inpUS7jWA9Mpb"
from datetime import datetime, timedelta
import pandas as pd
import asyncio
from openelectricity import AsyncOEClient
from openelectricity.types import DataMetric, MarketMetric

output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "NEM")
os.makedirs(output_dir, exist_ok=True)

mth = 12
start_date = datetime(2024, mth, 1, 0, 0, 0)
# end_date = datetime(2025, 1, 31, 0, 0, 0)
end_date = datetime(2024, mth, 30, 23, 59, 0)
# start_date = end_date - timedelta(days=325)


def month_chunks(start_date, end_date):
    """Yield (month_start, month_end) tuples for each calendar month in the range."""
    current = start_date
    while current < end_date:
        if current.month == 12:
            next_month = current.replace(year=current.year + 1, month=1, day=1)
        else:
            next_month = current.replace(month=current.month + 1, day=1)
        month_end = min(next_month, end_date)
        yield current, month_end
        current = month_end


async def fetch_data(startDt, endDt, price_data):
    async with AsyncOEClient() as client:
        response = await client.get_market(
            network_code="NEM",
            metrics=[MarketMetric.PRICE],
            interval="1h",
            date_start=startDt,
            date_end=endDt,
            primary_grouping="network_region",
        )
        for series in response.data:
            for result in series.results:
                print("Result name:", result.name)
                print("Number of data points:", len(result.data))
                for point in result.data:
                    if "QLD" in result.name or "QLD" in getattr(result, "region", ""):
                        price_data.append(
                            {
                                "timestamp": point.timestamp,
                                "value": point.value,
                                "unit": series.unit,
                                "region": result.name,
                            }
                        )


async def main():
    price_data = []
    for chunk_start, chunk_end in month_chunks(start_date, end_date):
        print(f"Fetching {chunk_start} to {chunk_end}")
        await fetch_data(chunk_start, chunk_end, price_data)
    df_price = pd.DataFrame(price_data)
    df_price.to_csv(os.path.join(output_dir, "price.csv"), index=False)
    print("Saved to data/NEM/price.csv")


await main()
# %% Get Qld price data
from openelectricity import types

# print("Available DataMetric attributes:")
# print(dir(types.DataMetric))

# print("All available attributes in openelectricity.types:")
# for attr in dir(types):
#     print(attr)
print("Available MarketMetric attributes:")
print(dir(types.MarketMetric))

# %%
# Import monthly pricing data read from OpenElecticity
import glob

# Path to monthly price files
price_files = glob.glob(os.path.join(output_dir, "price??_1h.csv"))

dfs = []
for file in sorted(price_files):
    df = pd.read_csv(file, usecols=["timestamp", "value"])
    dfs.append(df)

df_all = pd.concat(dfs, ignore_index=True)
df_all.to_csv(os.path.join(output_dir, "price_all_1h.csv"), index=False)
print(f"Combined {len(price_files)} files into price_all_1h.csv")
# %%
# Add season and integral time.
import pandas as pd


def get_season(dt):
    """Return season for a given datetime (Australian convention)."""
    month = dt.month
    if month in [12, 1, 2]:
        return "Summer"
    elif month in [3, 4, 5]:
        return "Autumn"
    elif month in [6, 7, 8]:
        return "Winter"
    else:
        return "Spring"


df_all = pd.read_csv(os.path.join(output_dir, "price_all_1h.csv"))

# Convert timestamp to datetime
df_all["timestamp"] = pd.to_datetime(df_all["timestamp"])

# Calculate interval (hour of day, 0-23 for hourly data)
df_all["interval"] = df_all["timestamp"].dt.hour

# Add season column
df_all["season"] = df_all["timestamp"].apply(get_season)

# Save updated CSV
df_all.to_csv(os.path.join(output_dir, "price_all_1h.csv"), index=False)
print("Added interval and season columns to price_all_1h.csv")
# %%
