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


def plot_energy_sankey(totals):
    # Example: adjust these keys to match your actual totals index
    sources = ["PV", "Grid", "Vehicle Battery", "Home Battery"]
    sinks = ["Consumption", "Export", "Curtailment", "Vehicle Battery", "Home Battery"]

    # Map flows (example, adjust as needed)
    flows = [
        # PV flows
        ("PV", "Consumption", totals.loc["pv_to_consumption"]),
        ("PV", "Home Battery", totals.loc["home_batt_charge"]),
        ("PV", "Vehicle Battery", totals.loc["veh_batt_charge"]),
        ("PV", "Export", totals.loc["grid_export"]),
        ("PV", "Curtailment", totals.loc["curtailment"]),
        # Grid flows
        ("Grid", "Consumption", totals.loc["grid_import"]),
        # Battery discharges
        ("Home Battery", "Consumption", totals.loc["home_batt_discharge"]),
        (
            "Vehicle Battery",
            "Consumption",
            totals.loc["veh_batt_discharge"] + totals.loc["driving_discharge"],
        ),
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

    fig = go.Figure(
        go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=nodes,
            ),
            link=link,
        )
    )
    fig.update_layout(title_text="Annual Energy Flow Sankey Diagram", font_size=12)
    return fig
