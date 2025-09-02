# V2G - Electricity Analysis

This project analyzes electricity data to explore V2G (Vehicle-to-Grid), home battery, and PV (solar) possibilities.

## Folder Structure

- `data/`: Contains raw and processed data.
  - `inputs/`: Input data files (price, PV, consumption, driving).
  - `processed/`: Processed data files.
- `notebooks/`: Jupyter notebooks for exploratory analysis.
- `src/`: Python scripts for data processing, visualization, and analysis.
- `tests/`: Unit tests for the Python scripts.

## Getting Started - runing locally (not for cloud users)

1. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare input data:**
   Place your price, PV, consumption, and driving data in the appropriate folders (see `data/inputs/`).

3. **Run the Streamlit app:**
   ```bash
   streamlit run src/app.py
   ```
   Or run Jupyter notebooks in `notebooks/` for exploratory analysis.

## Getting Started - runing on cloud

Go to https://r9gkgczox3bz22zvp6wufb.streamlit.app/
You'll need some input files - particularly pricing. Some defaults will load. Use those initially

## Parameters:

- **home_battery / vehicle_battery:**

  \*\*Menu to access: Main Actions (sidebar): [edit parameters] / [system_params] / [home_battery]

  > Battery objects with parameters for capacity (kWh), max charge/discharge rates (kW), cycle efficiency (%), and degradation rates.
  > There are quite a few other hidden parameters that will be exposed later. For example, there are degradation parameters that need a little more work.
  > Although they have been implemented with defaults, projected battery SoH has not been tested/reviewed.

- **grid:**  
  \*\*Menu to access: Main Actions (sidebar): [edit parameters] / [system_params] / [grid]
  Grid object with import/export costs (per kWh), daily fee, and max export rate.
  As with battery, there are more parameters that will progressively be exposed.

- **global:**  
  \*\*Menu to access: Main Actions (sidebar): [edit parameters] / [system_params] / [global]
  - Energy consumption per km for the vehicle. **kwh_per_km:**
  - Ignore any others for now

## Synthetic inputs:

Data for PV generation, driving behaviour and household consumption are, for now, all generated stochastically. Some parameter names are more
meaningful than others. If you don't know what it does, it's probably best not to touch it. Importantly, none of these are correlated with
any real world (eg pricing) data. For example, one might expect household consumption to be correlated with pricing - but such association is
absent. For any reasonable size battery, this is unlikely to be an issue. Subsequent iterations may allow upload of retail metering data.

- **driving:**  
   \*\*Menu to access: Main Actions (sidebar): [edit parameters] / [generator_params] / [driving]
  There are parameters for up to 4 trips per day. They specify, stochastically,

  - the probability of a trip happening;
  - the (return) distance of such trip (and hence, through kwh_per_km, energy consumption)
  - the time (hr) at which the trip occurs and
  - the period the trip takes (return)
  - whether the trip is on a weekday, weekend, or both
    For example someone who commutes 3 weekdays per week, do a couple of short trips on the weekend and travels overnight once a month might have these parameters:

          "probability": 0.42,  (3/7)   "probability": 0.8,        "probability": 0.025,
          "weekday": true,              "weekday": false,          "weekday": true,
          "weekend": false,             "weekend": true,           "weekend": false,
          "distance_mean": 30.0,        "distance_mean": 10.0,     "distance_mean": 400.0,
          "distance_std": 1.0,          "distance_std": 5.0,       "distance_std": 50.0,
          "time_mean": 8.0,             "time_mean": 11.0,         "time_mean": 6.0,
          "time_std": 0.2,              "time_std": 5.0,           "time_std": 0.1,
          "length_mean": 9.0,           "length_mean": 2.0,        "length_mean": 36.0,
          "length_std": 0.5             "length_std": 1.0          "length_std": 1.0

  _plugged_in_ is presently set to 1 if the vehicle is not on a trip and 0 otherwise (with fractions allowing for non-integral (mostly) trip periods). Distributional assumptions for this may be important...for example, users who only plug in when they need a charge. But such users are not the target of this initial roll out of the model.

- **PV:**  
  \*\*Menu to access: Main Actions (sidebar): [edit parameters] / [generator_params] / [pv]
  Capacity is probably the only obvious one.
  summer_gen_factor and winter_gen_factor are the multiples of capacity produced on a sunny day in summer and winter respectively. The pattern
  over the year is harmonic as is the pattern over a day (half a period). Presently solar is spread from 8am to 4pm every day - no seasonality.
  sunny_prob is the probability of a day being sunny (regardless of season). If it's sunny you get the full production. If it's not sunny the
  amount of pv generation is governed by cloudy_mean_frac and cloudy_std_frac (I think std should be se).

- **Household consumption:**  
  \*\*Menu to access: Main Actions (sidebar): [edit parameters] / [generator_params] / [consumption]
  base_avg is hourly energy (eg overnight). There are morning and evening peaks to be configured. Vehicle charging is separate.

- **Paths**  
  Ignore these.

- **Price - csv file**  
   \*\*Menu to access: Main Actions (sidebar): [price data]
  The default, and only, price file is for 12 months ending 30 June 2025 for Qld reference node. Fields are timestamp,value,interval,season - you
  may need to add season if you have your own pricing file. Note that this is hourly data. The whole projection is hourly. The "value" I've used
  is the average 5m price from open electricity.

## Projection

\*\*Menu to access: Main Actions (sidebar): [project model]

- **Presentation:**
  These notes will appear at the top of the page if the checkbox in the sidebar is true. I'd recommend closing it when you've read what you
  need.

  For parameter input, generally parameter control is at the top.  
  For synthetic data sets, if there are parameters, some summaries of the synthetic data series are shown below the parameters.
  The last chart of these (under this heading: "Short Time Series: Volatility by Season") selects a week randomly from the chosen season.

  Projection results appear after they are created (_project model_ from side bar menu)
  The inital summary shows totals for many (not all) of the projected values. Labels are the variable names - usually reaonably meaningful.
  Units are either kWh, $ or c (for _rates_) - you'll have to make an educated guess as to which is which.  
  With _summary table_ selected in the left combo box, the adjacent box can be used to select total (default) or daily averages - the latter
  may be more useful for many figures.

  The Sankey diagram is, I hope, reasonably self explanatory. It seems to render better in the cloud implementation than locally.

  The other 3 options in the upper left combo box on the results page are _single day, weekly and daily_avg_. The right combo can be used to
  select a season. This just determines from which season the randomly selected subset is chosen. The box with the garish red selectors
  can be used to choose which variables appear in the chart. _daily_avg_ is probably not very useful. It is not adjusted for things (eg SoC)
  that shouldn't be averaged. _Daily_ and _weekly_ options are possibly useful for understanding flows - possibly more useful during development
  than for general use.

- **Data Integration:**  
  Combines price, PV, consumption, and driving data into a unified DataFrame. No user requirement to do this.

- **Battery Scheduling:**  
  Uses lookahead logic to determine charging/discharging based on future wholesale prices, PV and consumption. Perfect foresight is assumed.
  The algorithm is not too bad for a home battery. But the vehicle requirement is much trickier and hasn't been considered too much yet.
  Dynamic programming might be considered down the track - but a couple of trials produce plausible results.

- **Energy Flow Model:**  
  Simulates hour-by-hour flows: grid import/export, PV usage, battery charge/discharge, vehicle consumption, and cycling losses.
  Battery degradation is provided for but is very preliminary.

- **Variable names requiring explanation:**

  - **consumption_kwh:** Household consumption
  - **unmet_vehicle_consumption:** This arises when battery SoC is not sufficient for travel. This could be called _public_veh_charge_
  - **driving_discharge:** _vehicle_consumption_ net of unmet_vehicle_consumption
  - **home_batt_loss:** The amount of energy lost in cycling. (ditto for _veh_batt_loss_)
  - **x_earnings:** The product of relevant energy flow and [effective import price] or [effective export price] as appropriate. The [effective import price] is the sum of the wholesale price (from the price input csv) and network_cost_import_per_kwh.
  - **x_rate:** The average price for the flow (c/kWh). Not at all tested.

## Additional Notes

- For troubleshooting, check that all required data files are present and paths are correct.
- You can modify parameters in the sidebar or scenario JSON.
- Example scenarios and outputs are available in the `notebooks/` folder.
