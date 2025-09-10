# V2G - Household electricity flow model

This application models household electricity flows primarily to explore V2G (Vehicle-to-Grid), home battery, and PV (solar) possibilities.

## Getting Started - runing locally (not for cloud users). Use the cloud version in the short term.

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

## Getting Started - streamlit cloud

Go to https://r9gkgczox3bz22zvp6wufb.streamlit.app/
You'll need some input files - particularly pricing. Some defaults will load. Use those initially. Only those stored by the developer are currently available from select boxes.

## Parameters: - Review the User Guide for an overview and navigation

\*\* **Edit parameters** button to select edit mode

- **system_params:**

  **Batteries** have parameters for capacity (kWh), max charge/discharge rates (kW), cycle efficiency (%), and degradation rates. Put capacity to zero if there is no battery.
  There are hidden parameters that will be exposed later. For example, there are degradation parameters that need more work.

  **grid:**  
  Network import/export costs (per kWh), daily fee, and max export rate, FIT (only used if pricing is "tariff").
  As with battery, there are more parameters that will progressively be exposed.

  **global:**

  - Energy consumption per km for the vehicle: kwh_per_km. (This will probably move to the vehicle battery set)
  - Don't touch start_date or export_df_flag.
  - public_charge_rate is the cost of charging away from the home - it has some impact on optimisation (or might do)

- **Synthetic inputs:**

  Data for PV generation, driving behaviour and household consumption are, for now, all generated stochastically. Some parameter names are more
  meaningful than others. If you don't know what it does, it's probably best not to touch it. Importantly, none of these are correlated with
  any real world (eg pricing) data. For example, one might expect household consumption to be correlated with pricing - but such association is
  absent. For any reasonable size battery, this is unlikely to be an issue. Subsequent iterations may allow upload of retail metering data.

  **driving:**  
  There are parameters for up to 4 trips per day. They specify, stochastically,

  - the probability of a trip happening;
  - the (return) distance of such trip (and hence, through kwh_per_km, energy consumption)
  - the time (hr) at which the trip occurs and
  - the period the trip takes (return)
  - whether the trip is on a weekday, weekend, or both
    For example someone who commutes 3 weekdays per week, does a couple of short trips on the weekend and travels overnight once a month might use these 3 parameter sets:

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

  **PV:**

  Capacity has an obvious meaning.
  summer_gen_factor and winter_gen_factor are the multiples of capacity produced on a sunny day in summer and winter respectively. The pattern
  over the year is harmonic as is the pattern over a day (half a period). Presently solar is spread from 8am to 4pm every day - no seasonality.
  sunny_prob is the probability of a day being sunny (regardless of season). If it's sunny you get the full production. If it's not sunny the
  amount of pv generation is governed by cloudy_mean_frac and cloudy_std_frac.

  Although I may implement something more realistic, this seems to give a reasonable starting point. If you know your average daily generation, I'd suggest tuning parameters to get a reasonable value for this.

  **Household consumption:**
  There is an arbitrary number of "activities". The example below, with 4 activities, is for a WorkFromHome scenario. The names below, _Base load, breakfast etc_, do not appear in the app but are here for illustration. Base load runs for 24 hours every day (both weekday and weekend are true) and has limited variability. _Breakfast_ and _Dinner_ cover morning and evening peaks and _WorkDay_ provides for a higher rate during the work day.

```
              Base load            Breakfast           WorkDay               Dinner
            "start_hour": 0,      "start_hour": 6,    "start_hour": 8,      "start_hour": 18,
            "rate": 0.3,          "rate": 1.3,        "rate": 0.5,          "rate": 1.8,
            "rate_sd": 0.1,       "rate_sd": 0.2,     "rate_sd": 0.2,       "rate_sd": 0.2,
            "length": 24.0,       "length": 1.5,      "length": 9.0,        "length": 2.5,
            "length_sd": 0.0,     "length_sd": 0.3,   "length_sd": 0.1,     "length_sd": 0.7,
            "weekday": 1,         "weekday": 1,       "weekday": 1,         "weekday": 1,
            "weekend": 1          "weekend": 0        "weekend": 0          "weekend": 1
```

> > **Price - tariff**

> > For many investigations, volatile wholesale market energy prices are likely to be most relevant. See the next section for them. But a ToU (or flat) tariff can be set. This might be at least useful for comparison purposes. An arbitrary number of periods can be set. For each, there is simply a start_hour (integral) and rate (c/kWh, float). The specified rate applies from start_hour until the next start_hour. There is presently no validation of the ordering....please input values with appropriate caution. Hopefully I'll get to doing more validation.

> > Presently this tariff represents the energy component of the real tafiff. The grid parameter, grid.network_cost_import_per_kwh, is added to this to get the per kWh cost of importing. grid.fit is the value of exports - this is set independently of the tariff. It is not used if a wholesale energy price file is used - in that case, the wholesale energy price is used for exports.

## Price - import NEM data

\*\* **Select price series** button to select price mode

A (small) selection of price files extracted from open-electricity are available (presently just Qld and Vic). This is hourly data as the whole projection is hourly. The price I've used is the average (no volume, or any other, weighting) over 5m prices from open-electricity. Smoothing the 5m volatility will introduce errors into the optimisation. However this error is likely to be low. For example, if the algorithm exports at, say, a 30c/kWh wholesale price, an hour that averaged $2/kWh would likely have all 5m periods above 30c in which case the average would give a correct optimum. For cases where the average exceeded 30c/kWh but there were several 5m periods which were below this, the algorithm would export more energy at a lower price - so it would show a poorer financial outcome than a working system (using 5m pricing) would achieve.

## Projection

\*\* **Project model** button to run the model and view reults

- **Presentation:**
  These notes will appear at the top of the page if the relevant checkbox in the sidebar is true. I'd recommend closing it when you've read what you
  need.

  For parameter input, generally parameter control is at the top.  
  For synthetic data sets, if there are parameters, some summaries, usually graphical, of the synthetic data series generated by the parameters are shown below the parameters. The last chart of these (under this heading: "Short Time Series: Volatility by Season") selects a week randomly from the chosen season.

  Projection results appear after they are created (_project model_ from side bar menu)
  The inital summary shows totals for many (not all) of the projected values. Labels are the variable names - usually reaonably meaningful.
  Units are either kWh, $ or c (for _rates_).  
  With _summary table_ selected in the left select box, the adjacent box can be used to select total (default) or daily averages - the latter
  may be more useful for many figures.

  The Sankey diagram is, I hope, reasonably self explanatory.

  The other 3 options in the upper left combo box on the results page are _single day, weekly and daily_avg_. The right combo can be used to
  select a season. This just determines from which season the randomly selected subset is chosen. The box with the garish red selectors
  can be used to choose which variables appear in the chart. _daily_avg_ is probably not very useful. It is not adjusted for things (eg SoC)
  that shouldn't be averaged. _Daily_ and _weekly_ options are possibly useful for understanding flows - possibly more useful during development
  than for general use.

- **Battery Scheduling:**  
  Uses lookahead logic to determine charging/discharging based on future wholesale prices, PV and consumption. Perfect foresight is assumed.
  The algorithm seems not too bad for a home battery. The vehicle algorith, which allows for when the vehicle is not at home, is trickier - but it produces plausible results on the limited parameter inputs that have been tested. (Dynamic programming might be considered down the track.)

- **Energy Flow Model:**  
  Simulates hour-by-hour flows: grid import/export, PV usage, battery charge/discharge, vehicle consumption, and cycling losses.
  Battery degradation is provided for but is very preliminary (with no results presented).

**Variable names are mostly self-explanatory. A couple here might help:**

- **consumption_kwh:** Household consumption
- **public_veh_charge:** This arises when battery SoC is not sufficient for travel.
- **driving_discharge:** _vehicle_consumption_ net of unmet_vehicle_consumption - so driving the energy for which is provided from home charging.
- **home_batt_loss:** The amount of energy lost in cycling. (ditto for _veh_batt_loss_)
- **x_earnings:** The product of relevant energy flow and [effective import price] or [effective export price] as appropriate. The [effective import price] is the sum of the wholesale price (from the price input csv) and network_cost_import_per_kwh.
- **x_rate:** The average price for the flow (c/kWh). Not well tested.

## Additional Notes

- For troubleshooting, check that all required data files are present and paths are correct.
