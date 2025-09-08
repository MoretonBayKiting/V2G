# V2G App – User Guide

This tool simulates household energy flows (grid, PV, battery, EV) using synthetic or user-supplied data. It is designed for users familiar with the Australian electricity market and energy modeling. It is a very early development version. No promises!

## Getting Started

1. **Launch the App**

   - https://r9gkgczox3bz22zvp6wufb.streamlit.app/
   - The only real data is NEM regional reference node wholesale pricing for ye 30/6/2025
   - The only period to model is ye 30/6/2025

2. **Select mode**

   - Use the sidebar to select one of the following using a button
     - Edit parameters
     - Select price series
     - Project model
   - Default scenarios and price histories are provided.
   - No facility presently to upload either. But you can download a scenario file.

3. **Edit Parameters**

   - Select a scenario - only stored scenarios are presently available. Then:
     - Select parameter groups (synthetic_data_params, system_params)
     - Select subgroups
       - for synthetic_data_params, choose from driving, pv, consumption
       - for system_params, choose from home_battery, vehicle_battery, grid or global.
     - Put battery or pv capacity to zero if they are not to be included (ie they don't exist)
     - Adjust values as needed. Hover for hints.
   - Save changes to update the scenario.

4. **Synthetic Data**

   - PV, driving, and consumption profiles are generated stochastically from input parameters.
   - Refer to the more complete documentation for examples of driving trip parameters.
   - Synthetic parameters generate hourly time series from the stochastic parameters provided by the user. Look at the charts and totals (eg _"Total km travelled in the year: 13620.2 km (13.6k)"_ to assess whether the parameters produce what you think is reasonable)
   - There is limited flexibility. For example, consumption provides for a base rate and morning and evening peaks - timing of those peaks is persently **not** within user control.
   - Synthetic time series are generated quite independently of pricing data (which is the only real data being used presently). At least each of PV, consumption and pricing series should be correlated. This may be impelemented if warranted. In the short term, as this is primarily targeted at V2G considerations and pricing and vehicle use dominate other considerations (IMHO), no correlations have been provided for.

5. **Price Data**

   - Choose a price file in the sidebar.
   - Click "Select price series". Shows some views of the series. Only Vic and Qld presently available.

6. **Run Model**

   - Click "Project model" to simulate energy flows and costs.
   - There are several views of the results:
     - _Summary table_:
       - Sankey diagram at bottom is probably the best way to get a quick view of the projection - if you're used to Sankey diagrams.
       - Choose either annual totals or daily averages. Some daily averages may not make much sense.
         - Left table shows amounts of energy through each source/sink/device
         - Centre table is financial flows - Net Earnings sums the flows above that
         - Right table shows average prices.
     - _weekly_ or _single day_ - probably most useful for debugging or for those who want to get into the weeds
     - _Daily average_: Perhaps not that helpful. Let me know what you think.

## Notes

- This is a first cut model - if you can see it, I'm probably just seeking input for further development. No promises that it won't break.
- Time series are hourly and aligned by date.
- Battery degradation projection is a work in progress - not yet visible. But it's well advanced and, I think, an important component.
- The model assumes perfect foresight (eg of future PV, pricing, consumption) for battery scheduling. But the algorithm for optimising this for earnings is quite primitive. Perhaps the 2 factors offset each other so that the net earnings are about optimum!
- The more complete documentation may or may not be useful.
- For troubleshooting, data requirements, and detailed parameter descriptions - TBA.
