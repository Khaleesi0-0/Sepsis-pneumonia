from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[2]
CLEANED_DIR = ROOT / "data" / "cleaned"
FIGURE_DIR = ROOT / "results" / "figures" / "age"

COVID_START = 2020
COVID_END = 2023

Y_COL = "AAMR"
AGE_GROUP_COL = "Age Group"
YEAR_COL = "Year"
LINE_WIDTH = 1.9
MARKER_SIZE = 5.5
SHADE_COLOR = "#D8C7A3"
TEXT_COLOR = "#1F1A17"
AGE_GROUP_ORDER = ["<25", "25-44", "45-64", "65+"]
AGE_PALETTE = {
    "<25": "#17365D",
    "25-44": "#2A6F4F",
    "45-64": "#C17C00",
    "65+": "#B14A3B",
}

DATASETS = {
    "ards": CLEANED_DIR / "ards_age.csv",
    "pneumonia": CLEANED_DIR / "pneumonia_age.csv",
    "combined": CLEANED_DIR / "combined_age.csv",
}


def _select_year_ticks(years: list[int], max_ticks: int = 8) -> list[int]:
    years = sorted(int(year) for year in years if pd.notna(year))
    if len(years) <= max_ticks:
        return years

    step = max(1, (len(years) + max_ticks - 1) // max_ticks)
    ticks = years[::step]
    if ticks[-1] != years[-1]:
        ticks.append(years[-1])
    return ticks


def _apply_publication_style() -> None:
    sns.set_theme(
        style="white",
        context="paper",
        rc={
            "figure.dpi": 300,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.labelsize": 10.5,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.linewidth": 0.9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 4,
            "ytick.major.size": 4,
            "grid.color": "#e4e0d8",
            "grid.linewidth": 0.55,
            "grid.linestyle": "--",
            "legend.frameon": False,
            "legend.fontsize": 9,
        },
    )


def _prepare_for_plot(df: pd.DataFrame) -> pd.DataFrame:
    plot_df = df.copy()
    plot_df[YEAR_COL] = (
        plot_df[YEAR_COL]
        .astype("string")
        .str.strip()
        .str.replace(r"\s*\(provisional\)$", "", regex=True)
    )
    plot_df[YEAR_COL] = pd.to_numeric(plot_df[YEAR_COL], errors="coerce")
    plot_df[Y_COL] = pd.to_numeric(plot_df[Y_COL], errors="coerce")
    plot_df = plot_df.dropna(subset=[YEAR_COL, Y_COL, AGE_GROUP_COL])

    plot_df = (
        plot_df.groupby([YEAR_COL, AGE_GROUP_COL], as_index=False)[Y_COL]
        .mean()
        .sort_values([YEAR_COL, AGE_GROUP_COL])
    )
    plot_df[AGE_GROUP_COL] = pd.Categorical(
        plot_df[AGE_GROUP_COL],
        categories=AGE_GROUP_ORDER,
        ordered=True,
    )
    return plot_df


def _draw_single_trend(ax: plt.Axes, df: pd.DataFrame, title: str, show_legend: bool = True) -> None:
    sns.lineplot(
        data=df,
        x=YEAR_COL,
        y=Y_COL,
        hue=AGE_GROUP_COL,
        hue_order=AGE_GROUP_ORDER,
        marker="o",
        linewidth=LINE_WIDTH,
        markersize=MARKER_SIZE,
        dashes=False,
        palette=AGE_PALETTE,
        ax=ax,
    )
    ax.axvspan(COVID_START, COVID_END, color=SHADE_COLOR, alpha=0.26, zorder=0)
    ax.set_title(title, loc="left", pad=12, color=TEXT_COLOR)
    ax.set_xlabel("Year")
    ax.set_ylabel("Age-adjusted rate")
    year_ticks = _select_year_ticks(df[YEAR_COL].dropna().unique().tolist())
    ax.set_xticks(year_ticks)
    ax.grid(axis="y", alpha=0.8)
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="x", direction="out", rotation=45)
    ax.tick_params(axis="y", direction="out")

    handles, labels = ax.get_legend_handles_labels()
    if show_legend:
        ax.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.18),
            title=None,
            handlelength=2.0,
            ncol=4,
            columnspacing=1.2,
        )
    else:
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()


def build_trend_figures() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    _apply_publication_style()

    prepared = {}
    for name, csv_path in DATASETS.items():
        raw = pd.read_csv(csv_path)
        prepared[name] = _prepare_for_plot(raw)
    disease_titles = {
        "pneumonia": "Pneumonia",
        "ards": "Pneumonia/ARDS",
        "combined": "Pneumonia/Sepsis",
    }

    for name, df in prepared.items():
        fig, ax = plt.subplots(figsize=(6.8, 4.6))
        _draw_single_trend(ax, df, f"{disease_titles[name]} trend by age group", show_legend=True)
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / f"{name}_trend_by_age.png", dpi=300)
        fig.savefig(FIGURE_DIR / f"{name}_trend_by_age.pdf")
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8), sharex=True, sharey=False)
    panel_order = ["pneumonia", "ards", "combined"]
    panel_titles = disease_titles
    panel_labels = ["A", "B", "C"]

    legend_handles = None
    legend_labels = None
    for ax, key, panel_label in zip(axes, panel_order, panel_labels):
        _draw_single_trend(ax, prepared[key], panel_titles[key], show_legend=False)
        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()
        ax.text(
            0.01,
            0.98,
            panel_label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=12,
            fontweight="bold",
            color=TEXT_COLOR,
        )

    axes[0].set_xlabel("")
    axes[1].set_xlabel("")
    axes[2].set_xlabel("")
    axes[0].set_ylabel("Age-adjusted rate")
    axes[1].set_ylabel("Age-adjusted rate")
    axes[2].set_ylabel("Age-adjusted rate")
    fig.supxlabel("Year")
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=4,
        frameon=False,
        handlelength=2.0,
        columnspacing=1.2,
    )
    fig.text(
        0.99,
        0.02,
        "Shaded area: COVID era (2020-2023)",
        ha="right",
        va="bottom",
        fontsize=8.5,
        color="#4F4A43",
    )

    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "all_trends_by_age_panel.png", dpi=300)
    fig.savefig(FIGURE_DIR / "all_trends_by_age_panel.pdf")
    plt.close(fig)


if __name__ == "__main__":
    build_trend_figures()

