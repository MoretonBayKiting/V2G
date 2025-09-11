# V2G App – User Guide

This tool simulates household energy flows (grid, PV, battery, EV) using synthetic or user-supplied data. It is designed for users familiar with the Australian electricity market and energy modeling. It is a very early development version. No promises!
It assumes an Amber pricing model - ie energy flows to and from the grid at wholesale prices with network costs added to imports (export network costs are also parameterised but zero in defaults).

## Getting Started

1. **Launch the App**

   - https://r9gkgczox3bz22zvp6wufb.streamlit.app/
   - The only real data is NEM regional reference node wholesale pricing for ye 30/6/2025
   - The only period to model is ye 30/6/2025
   - Only 1 vehicle, 1 home battery and 1 PV system. Put capacities (kWh for batteries, kW for PV) to zero if the relevant component doesn't exit.

2. **Select mode**

   - A default set of assumptions will load when the system loads. Select one of the following buttons from the top of the sidebar
     - Edit parameters
     - Select price series
     - Project model
   - Default scenarios and price histories are provided.
   - There is no facility to upload either. "scenarios" can be edited to saved (json format) locally.

3. **Edit Parameters**

   - Select a scenario - only stored scenarios are presently available. Then:
     - Select parameter groups (synthetic_data_params, system_params)
     - Select subgroups
       - for synthetic_data_params, choose from driving, pv, consumption, tariffs
       - for system_params, choose from home_battery, vehicle_battery, grid or global.
     - Put battery or pv capacity to zero if they are not to be included (ie they don't exist)
     - Adjust values as needed. Hover for hints (sometimes...).
   - Save changes to update and use the scenario. Download to keep for later use.

4. **Synthetic Data**

   - PV, driving, and consumption profiles (hourly over a year) are generated stochastically from input parameters.
   - A tariff structure can be input here. This is not stocahstic. If this tariff pricing is not used, an historic NEM price series will be chosen.
   - Refer to the more complete documentation for examples of parameters.
   - "Synthetic parameters" generate a one year hourly time series from the stochastic parameters. Look at the charts and totals (eg _"Total km travelled in the year: 13620.2 km (13.6k)"_) to assess whether the parameters produce what something reasonable.
   - There is limited flexibility. Greater flexibility may be added later.
   - Synthetic time series are generated quite independently of pricing data (which is the only real data being used presently). At least each of PV, consumption and pricing series should be correlated. This may be impelemented. In the short term, as this is primarily targeted at V2G considerations and pricing and vehicle use dominate other considerations (IMHO), no correlations have been provided for in this initial draft.

5. **Price Data**

   - Click "Select price series". This will show some summary views of the series. Only Vic and Qld presently available.
   - As noted above, there is a tariff option. This allows construction (and use) of reasonably flexible ToU tariffs (no allowance for demand charges).
     Rates are wholesale energy rates - network_cost_import_kwh, a grid parameter, is added to the energy rate to get a total import price (/kWh).
     Another grid parameter, daily_fee, is as the name indicates - a daily network fee independent of consumption.
   - A possible use of the tool might be to model a house with no PV or batteries and taking ToU pricing and compare that to a battery/PV option with wholesale market pricing.

6. **Run Model**

   - Click "Project model" to simulate energy flows and costs.
   - There are several views of the results:
     - _Summary table_:
       - A Sankey diagram at the bottom provides a good overview - if you're used to Sankey diagrams.
       - Choose either annual totals or daily averages. Some daily averages (eg SoC) may not make much sense.
         - Left table shows amounts of energy through each source/sink/device;
         - Centre table is financial flows - **Net Earnings** sums the flows above that;
         - Right table shows average prices (c/kWh).
     - _weekly_ or _single day_ - probably most useful for debugging or for those who want to get into the weeds
     - _Daily average_: Perhaps not that helpful. Let me know what you think.

7. **Store results**
   - There's a new save/download button below results in main page. That will dump inputs and summary results to a json file. Alternatively, print from the browser.

## Notes

- This is a first cut model - if you can see it, I'm probably just seeking input for further development. No promises that it won't break.
- Time series are hourly for a year.
- Battery degradation projection is a work in progress - not yet visible. But it's well advanced and, I think, an important component.
- The model assumes perfect foresight (eg of future PV, pricing, consumption) for battery scheduling. But the algorithm for optimising this for earnings is relatively primitive. Perhaps the 2 factors offset each other so that the net earnings are about optimum!
- The more complete documentation may or may not be useful.
- For troubleshooting, data requirements, and detailed parameter descriptions - TBA.
