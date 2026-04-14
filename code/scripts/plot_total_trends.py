from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[2]
CLEANED_DIR = ROOT / "data" / "cleaned"
FIGURE_DIR = ROOT / "results" / "figures" / "overall"

YEAR_COL = "Year"
VALUE_COL = "Age Adjusted Rate"
COVID_START = 2020
COVID_END = 2023
FIG_SIZE = (7.2, 4.8)
LINE_WIDTH = 2.0
MARKER_SIZE = 6
SHADE_COLOR = "#D8C7A3"
TEXT_COLOR = "#1F1A17"
PLOT_COLORS = {
    "Pneumonia": "#B14A3B",
    "Pneumonia/ARDS": "#17365D",
    "Pneumonia/Sepsis": "#2A6F4F",
}

DATASETS = {
    "Pneumonia/ARDS": CLEANED_DIR / "ards_sex.csv",
    "Pneumonia": CLEANED_DIR / "pneumonia_sex.csv",
    "Pneumonia/Sepsis": CLEANED_DIR / "combined_sex.csv",
}


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


def _extract_total_series(csv_path: Path, disease_name: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, dtype={YEAR_COL: "string"})

    # Prefer explicit totals from CDC export.
    total_df = df[df.get("Notes", "").astype("string").str.strip().eq("Total")].copy()

    # Fallback if explicit "Total" rows are missing.
    if total_df.empty:
        df[YEAR_COL] = (
            df[YEAR_COL]
            .astype("string")
            .str.strip()
            .str.replace(r"\s*\(provisional\)$", "", regex=True)
        )
        df[YEAR_COL] = pd.to_numeric(df[YEAR_COL], errors="coerce")
        df[VALUE_COL] = pd.to_numeric(df[VALUE_COL], errors="coerce")
        total_df = (
            df.dropna(subset=[YEAR_COL, VALUE_COL])
            .groupby(YEAR_COL, as_index=False)[VALUE_COL]
            .mean()
        )
    else:
        total_df[YEAR_COL] = (
            total_df[YEAR_COL]
            .astype("string")
            .str.strip()
            .str.replace(r"\s*\(provisional\)$", "", regex=True)
        )
        total_df[YEAR_COL] = pd.to_numeric(total_df[YEAR_COL], errors="coerce")
        total_df[VALUE_COL] = pd.to_numeric(total_df[VALUE_COL], errors="coerce")
        total_df = total_df.dropna(subset=[YEAR_COL, VALUE_COL])[[YEAR_COL, VALUE_COL]]

    total_df["Disease"] = disease_name
    return total_df.sort_values(YEAR_COL)


def build_total_trend_plot() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    _apply_publication_style()

    trend_frames = []
    for disease_name, path in DATASETS.items():
        trend_frames.append(_extract_total_series(path, disease_name))

    trend_df = pd.concat(trend_frames, ignore_index=True)

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    sns.lineplot(
        data=trend_df,
        x=YEAR_COL,
        y=VALUE_COL,
        hue="Disease",
        marker="o",
        linewidth=LINE_WIDTH,
        markersize=MARKER_SIZE,
        dashes=False,
        palette=PLOT_COLORS,
        ax=ax,
    )
    ax.axvspan(COVID_START, COVID_END, color=SHADE_COLOR, alpha=0.26, zorder=0)

    ax.set_xlabel("Year")
    ax.set_ylabel("Age-adjusted rate")
    ax.grid(axis="y", alpha=0.8)
    ax.grid(axis="x", visible=False)
    ax.set_xticks(sorted(trend_df[YEAR_COL].dropna().unique()))
    ax.tick_params(direction="out")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        labels,
        title=None,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        ncol=3,
        handlelength=2.2,
        columnspacing=1.4,
    )
    fig.suptitle(
        "Observed age-adjusted mortality trends, 2010-2025",
        y=0.985,
        fontsize=14,
        fontweight="bold",
        color=TEXT_COLOR,
    )
    fig.text(
        0.5,
        0.945,
        "Shaded region marks 2020-2023 (COVID era).",
        ha="center",
        va="center",
        fontsize=9.2,
        color="#4F4A43",
    )

    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.89])
    fig.savefig(FIGURE_DIR / "total_trend_three_diseases.png", dpi=300)
    fig.savefig(FIGURE_DIR / "total_trend_three_diseases.pdf")
    plt.close(fig)


if __name__ == "__main__":
    build_total_trend_plot()

