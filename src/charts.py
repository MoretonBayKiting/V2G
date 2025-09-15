import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns


def boxplot_interval(df_long):
    fig, ax = plt.subplots(figsize=(18, 7))
    sns.boxplot(
        data=df_long, x="Interval", y="kWh", hue="CON/GEN", showfliers=False, ax=ax
    )
    ax.set_title("Distribution of Consumption and Generation (kWh) by Interval")
    ax.set_xlabel("Interval (0 = 00:00-00:30, 47 = 23:30-00:00)")
    ax.set_ylabel("kWh")
    ax.legend(title="Type")
    return fig


def boxplot_aggpd_season(df_long):
    fig, ax = plt.subplots(figsize=(14, 7))
    sns.boxplot(
        data=df_long[df_long["CON/GEN"] == "Generation"],
        x="AggPd",
        y="kWh",
        hue="Season",
        showfliers=False,
        ax=ax,
    )
    ax.set_title("Distribution of exports by 6-Hour Period (AggPd) and Season")
    ax.set_xlabel(
        "6-Hour Period (0=00:00-06:00, 1=06:00-12:00, 2=12:00-18:00, 3=18:00-00:00)"
    )
    ax.set_ylabel("kWh")
    ax.legend(title="Season")
    return fig


import plotly.graph_objects as go


# ...existing code...


def sankey_energy_check(flows):
    # List of internal battery flows to subtract (charge/discharge/losses)
    double_counted_keys = [
        ("PV", "Home Battery"),
        ("PV", "Vehicle Battery"),
        # ("Vehicle Battery", "Driving Discharge"),
        # ("Home Battery", "Consumption"),
        # ("Home Battery", "Export"),
        # ("Home Battery", "Home Battery Loss"),
        # ("Vehicle Battery", "Consumption"),
        # ("Vehicle Battery", "Vehicle Consumption"),
        # ("Vehicle Battery", "Export"),
        # ("Vehicle Battery", "Vehicle Battery Loss"),
        # ("Vehicle Battery", "Driving Discharge"),
    ]
    double_counted_sum = sum(
        val for src, dst, val in flows if (src, dst) in double_counted_keys
    )

    total_flow = sum(val for src, dst, val in flows)
    net_flow = total_flow - double_counted_sum
    # total_source = sum(val for src, dst, val in flows if src in source)
    # total_sink = sum(val for src, dst, val in flows if dst in sink)
    return total_flow, double_counted_sum, net_flow  # , total_source, total_sink


def plot_energy_sankey(totals):
    flows = [
        # PV flows
        ("PV", "Consumption", totals.loc["pv_to_consumption"]),
        ("PV", "Home Battery", totals.loc["home_batt_charge"]),
        ("PV", "Vehicle Battery", totals.loc["veh_batt_charge"]),
        ("PV", "Export", totals.loc["pv_export"]),
        ("PV", "Curtailment", totals.loc["curtailment"]),
        #
        (
            "public_charge",
            "Vehicle Consumption",
            (totals.loc["public_charge"] if "public_charge" in totals.index else 0),
        ),
        # Grid flows
        ("Grid", "Consumption", totals.loc["grid_import"]),
        # Battery discharges
        ("Home Battery", "Consumption", totals.loc["home_batt_discharge"]),
        (
            "Home Battery",
            "Export",
            totals.loc["home_export"] if "home_export" in totals.index else 0,
        ),
        (
            "Home Battery",
            "Home Battery Loss",
            totals.loc["home_batt_loss"] if "home_batt_loss" in totals.index else 0,
        ),
        (
            "Vehicle Battery",
            "Consumption",
            totals.loc["veh_batt_discharge"],
        ),
        # (
        #     "Vehicle Battery",
        #     "Vehicle Consumption",
        #     (
        #         totals.loc["vehicle_consumption"]
        #         if "vehicle_consumption" in totals.index
        #         else 0
        #     ),
        # ),
        (
            "Vehicle Battery",
            "Export",
            totals.loc["vehicle_export"] if "vehicle_export" in totals.index else 0,
        ),
        (
            "Vehicle Battery",
            "Vehicle Battery Loss",
            totals.loc["veh_batt_loss"] if "veh_batt_loss" in totals.index else 0,
        ),
        (
            "Vehicle Battery",
            "Vehicle Consumption",
            (
                totals.loc["driving_discharge"]
                if "driving_discharge" in totals.index
                else 0
            ),
        ),
        # (
        # "Driving Discharge",
        #     "Vehicle Consumption",
        #     (
        #         totals.loc["driving_discharge"]
        #         if "driving_discharge" in totals.index
        #         else 0
        #     ),
        # ),
    ]

    # Build node list
    nodes = list({src for src, _, _ in flows} | {dst for _, dst, _ in flows})
    node_indices = {name: i for i, name in enumerate(nodes)}

    # Build Sankey links
    link = dict(
        source=[node_indices[src] for src, dst, val in flows],
        target=[node_indices[dst] for src, dst, val in flows],
        value=[val for src, dst, val in flows],
        label=[f"{src}→{dst}" for src, dst, val in flows],
    )

    # Energy check using flows
    total_flow, double_counted_sum, net_flow = sankey_energy_check(flows)

    fig = go.Figure(
        go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=nodes,
                # color="lightgray",
            ),
            link=link,
        )
    )
    fig.update_layout(
        title_text="Energy Flow Sankey Diagram",
        font=dict(size=18, color="black", family="Arial"),
        # You can also try font_size=18 for older Plotly versions
    )

    return fig, total_flow, double_counted_sum, net_flow


def rolling_price_spread_components(df, price_col="price", m=72, n=12):
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


def plot_monthly_spread_summary(df, price_col="price"):
    """
    Plots monthly means of spread, highest, and lowest for two parameter sets.
    """
    param_sets = [(72, 12), (24, 6)]
    plt.figure(figsize=(12, 6))
    for m, n in param_sets:
        spread, highest, lowest = rolling_price_spread_components(df, price_col, m, n)
        # Add to DataFrame for grouping
        temp = df[["month"]].copy()
        temp[f"spread_{m}_{n}"] = spread
        temp[f"highest_{m}_{n}"] = highest
        temp[f"lowest_{m}_{n}"] = lowest
        monthly_means = temp.groupby("month")[
            [f"spread_{m}_{n}", f"highest_{m}_{n}", f"lowest_{m}_{n}"]
        ].mean()
        plt.plot(
            monthly_means.index,
            monthly_means[f"spread_{m}_{n}"],
            label=f"Spread ({m}/{n})",
        )
        plt.plot(
            monthly_means.index,
            monthly_means[f"highest_{m}_{n}"],
            "--",
            label=f"Mean High ({m}/{n})",
        )
        plt.plot(
            monthly_means.index,
            monthly_means[f"lowest_{m}_{n}"],
            "--",
            label=f"Mean Low ({m}/{n})",
        )
    plt.xlabel("Month")
    plt.ylabel("c/kWh")
    plt.title("Monthly Means of Rolling Price Spread and Components")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.close()


import numpy as np
import matplotlib.pyplot as plt


def plot_cumulative_earnings_costs(df, st=None):
    """
    Plots cumulative (Pareto) curves for vehicle export earnings, home export earnings, and import costs.
    Optionally displays in Streamlit if st is provided.
    """
    veh_earnings = df["veh_earnings"].fillna(0)
    veh_earnings = veh_earnings[veh_earnings > 0]
    veh_sorted = np.sort(veh_earnings)[::-1]
    veh_cum = np.cumsum(veh_sorted)
    veh_cum /= veh_cum[-1] if len(veh_cum) > 0 else 1

    home_earnings = df["home_earnings"].fillna(0)
    home_earnings = home_earnings[home_earnings > 0]
    home_sorted = np.sort(home_earnings)[::-1]
    home_cum = np.cumsum(home_sorted)
    if len(home_cum) > 0:
        home_cum /= home_cum[-1]

    import_costs = df["grid_import_cost"].fillna(0)
    import_costs = import_costs[import_costs > 0]
    import_sorted = np.sort(import_costs)[::-1]
    import_cum = np.cumsum(import_sorted)
    if len(import_cum) > 0:
        import_cum /= import_cum[-1]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        np.linspace(0, 100, len(veh_cum)),
        veh_cum * 100,
        label="Vehicle Export Earnings",
    )
    if len(home_cum) > 0:
        ax.plot(
            np.linspace(0, 100, len(home_cum)),
            home_cum * 100,
            label="Home Export Earnings",
        )
    if len(import_cum) > 0:
        ax.plot(
            np.linspace(0, 100, len(import_cum)), import_cum * 100, label="Import Costs"
        )
    ax.set_xlabel("% of periods (sorted by value, highest to lowest)")
    ax.set_ylabel("Cumulative % of total")
    ax.set_title("Concentration of Earnings and Costs by Period")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    if st is not None:
        st.pyplot(fig)
    return fig


def plot_flow_stacked_by_price(
    df, price_col="price_kwh", st=None, price_max=100, above=False
):
    """
    Stacked bar: number of periods per price bin (by mutually exclusive flow type).
    Overlay: colored dot per bin, y-position = net value, color = sign/magnitude of value.
    Left y-axis: count; right y-axis: net value ($).
    """
    price_vals = df[price_col] * 100  # $/kWh to c/kWh

    if above:
        mask = price_vals >= price_max
        title_suffix = f" (≥ {price_max}c/kWh)"
    else:
        mask = price_vals < price_max
        title_suffix = f" (< {price_max}c/kWh)"

    price_vals = price_vals[mask]
    df = df[mask]

    if len(price_vals) == 0:
        if st is not None:
            st.info(f"No data in selected price range {title_suffix}.")
        return None

    num_bins = 20
    min_price = price_vals.min()
    max_price = price_vals.max()
    bins = np.linspace(min_price, max_price, num_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Mutually exclusive masks
    grid_import = df["grid_import"] > 0
    home_export = df["home_export"] > 0
    veh_export = df["vehicle_export"] > 0

    both_export = home_export & veh_export
    home_only = home_export & ~veh_export
    veh_only = veh_export & ~home_export
    grid_only = grid_import & ~home_export & ~veh_export
    neither = ~grid_import & ~home_export & ~veh_export

    # Histogram counts
    grid_hist, _ = np.histogram(price_vals[grid_only], bins=bins)
    neither_hist, _ = np.histogram(price_vals[neither], bins=bins)
    home_hist, _ = np.histogram(price_vals[home_only], bins=bins)
    veh_hist, _ = np.histogram(price_vals[veh_only], bins=bins)
    both_hist, _ = np.histogram(price_vals[both_export], bins=bins)

    # Net value per bin (dot y-position)
    net_value = (
        df.get("veh_earnings", 0).fillna(0)
        + df.get("home_earnings", 0).fillna(0)
        - df.get("grid_import_cost", 0).fillna(0)
    )
    bin_indices = np.digitize(price_vals, bins) - 1
    net_value_per_bin = np.zeros(num_bins)
    for i in range(num_bins):
        in_bin = bin_indices == i
        if np.any(in_bin):
            net_value_per_bin[i] = net_value[in_bin].sum()
        else:
            net_value_per_bin[i] = np.nan

    # Build DataFrame
    stacked_df = pd.DataFrame(
        {
            "bin_center": bin_centers,
            "grid_import_only": grid_hist,
            "neither": neither_hist,
            "home_export_only": home_hist,
            "veh_export_only": veh_hist,
            "both_exports": both_hist,
            "net_value": net_value_per_bin,
        }
    )

    vmin = np.nanmin(net_value_per_bin)
    vmax = np.nanmax(net_value_per_bin)
    if vmin == vmax:
        vmin, vmax = -1, 1  # fallback for constant data
    fig, ax1 = plt.subplots(figsize=(10, 5))
    bar_width = (bins[1] - bins[0]) * 0.8
    offset = bar_width / 4  # small offset
    # 1. No-flow ("neither") histogram on left axis
    ax1.bar(
        bin_centers - offset,
        neither_hist,
        width=bar_width,
        label="Neither Import nor Export",
        color="lightgray",
        zorder=1,
    )
    ax1.set_ylabel("No-flow periods in bin")
    ax1.set_xlabel("Price (c/kWh)")
    ax1.set_title(f"Histogram of Period Types by Price Bucket{title_suffix}")
    ax1.legend(loc="upper left")
    ax1.grid(True, axis="y")

    # 2. Stacked flow histogram on first right axis
    ax2 = ax1.twinx()
    bottom = np.zeros_like(grid_hist)
    bars = []
    bars.append(
        ax2.bar(
            bin_centers + offset,
            grid_hist,
            width=bar_width,
            label="Grid Import Only",
            color="tab:blue",
            bottom=bottom,
            zorder=2,
        )
    )
    bottom += grid_hist
    bars.append(
        ax2.bar(
            bin_centers + offset,
            home_hist,
            width=bar_width,
            label="Home Export Only",
            color="tab:orange",
            bottom=bottom,
            zorder=3,
        )
    )
    bottom += home_hist
    bars.append(
        ax2.bar(
            bin_centers + offset,
            veh_hist,
            width=bar_width,
            label="Vehicle Export Only",
            color="tab:green",
            bottom=bottom,
            zorder=4,
        )
    )
    bottom += veh_hist
    bars.append(
        ax2.bar(
            bin_centers + offset,
            both_hist,
            width=bar_width,
            label="Both Exports",
            color="tab:red",
            bottom=bottom,
            zorder=5,
        )
    )
    ax2.set_ylabel("Flow periods in bin")
    # Only add legend for ax2 if you want a separate one, or combine with ax1

    # 3. Net value dots on second right axis (offset)
    from matplotlib.ticker import MaxNLocator

    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("axes", 1.12))  # Offset the third axis
    ax3.yaxis.set_major_locator(MaxNLocator(nbins=6))
    valid = ~np.isnan(net_value_per_bin)
    ax3.scatter(
        bin_centers[valid],
        net_value_per_bin[valid],
        s=120,
        c="black",
        edgecolor="black",
        marker="o",
        label="Net Value per Bin",
        zorder=10,
    )
    ax3.set_ylabel("Net Value per Bin ($)")

    # Optionally, combine legends
    lines_labels = [ax.get_legend_handles_labels() for ax in [ax1, ax2, ax3]]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    ax1.legend(lines, labels, loc="upper left")

    fig.tight_layout()
    if st is not None:
        st.pyplot(fig)
    return fig, stacked_df


def import_export_price_hist(df, st=None, bins=30):
    """
    Plots a histogram of import and export prices for periods with grid import or export.
    """
    import_mask = df["grid_import"] > 0
    import_prices = df.loc[import_mask, "effective_import_price"]
    export_mask = (df["home_export"] > 0) | (df["vehicle_export"] > 0)
    export_prices = df.loc[export_mask, "effective_export_price"]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(
        import_prices,
        bins=bins,
        alpha=0.6,
        label="Import price (when importing)",
        color="tab:blue",
    )
    ax.hist(
        export_prices,
        bins=bins,
        alpha=0.6,
        label="Export price (when exporting)",
        color="tab:orange",
    )
    ax.set_xlabel("Effective Price (c/kWh)")
    ax.set_ylabel("Hours")
    ax.set_title("Distribution of Import and Export Prices (Battery Trading)")
    ax.legend()
    if st is not None:
        st.pyplot(fig)
    return fig
