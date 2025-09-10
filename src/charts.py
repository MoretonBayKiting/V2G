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
