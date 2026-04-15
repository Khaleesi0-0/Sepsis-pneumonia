from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[2]
CLEANED_DIR = ROOT / "data" / "cleaned"
TABLE_DIR = ROOT / "results" / "tables"
FIGURE_DIR = ROOT / "results" / "figures" / "overall"

YEAR_COL = "Year"
RATE_COL = "Age Adjusted Rate"
NOTES_COL = "Notes"
APC_TABLE_PATH = TABLE_DIR / "joinpoint_apc_segments.csv"

COVID_START = 2020
COVID_END = 2023
SHADE_COLOR = "#D8C7A3"
TEXT_COLOR = "#1F1A17"

PLOT_COLORS = {
    "Pneumonia": "#B14A3B",
    "Pneumonia/ARDS": "#17365D",
    "Pneumonia/Sepsis": "#2A6F4F",
    "Non-COVID Pneumonia": "#8B5A2B",
}

DATASETS = {
    "Pneumonia": CLEANED_DIR / "pneumonia_sex.csv",
    "Pneumonia/ARDS": CLEANED_DIR / "ards_sex.csv",
    "Pneumonia/Sepsis": CLEANED_DIR / "combined_sex.csv",
    "Non-COVID Pneumonia": CLEANED_DIR / "non_covid_pneumonia_sex.csv",
}

SEGMENT_OFFSETS = {
    "Pneumonia": [18, 18, 18],
    "Pneumonia/ARDS": [18, -20, 12],
    "Pneumonia/Sepsis": [8, -8, 16],
    "Non-COVID Pneumonia": [18, -18],
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
    total_df = df[df[NOTES_COL].astype("string").str.strip().eq("Total")].copy()
    total_df[YEAR_COL] = (
        total_df[YEAR_COL]
        .astype("string")
        .str.strip()
        .str.replace(r"\s*\(provisional\)$", "", regex=True)
    )
    total_df[YEAR_COL] = pd.to_numeric(total_df[YEAR_COL], errors="coerce")
    total_df[RATE_COL] = pd.to_numeric(total_df[RATE_COL], errors="coerce")
    total_df = total_df.dropna(subset=[YEAR_COL, RATE_COL])[[YEAR_COL, RATE_COL]]
    total_df["Disease"] = disease_name
    return total_df.sort_values(YEAR_COL).reset_index(drop=True)


def _interpolate_y(years: np.ndarray, rates: np.ndarray, x_pos: float) -> float:
    return float(np.interp(x_pos, years, rates))


def _format_apc_label(apc: float, lo: float, hi: float) -> str:
    return f"{apc:.2f}% ({lo:.2f}, {hi:.2f})"


def build_total_trend_with_apc_plot() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    _apply_publication_style()

    trend_frames = []
    for disease_name, path in DATASETS.items():
        trend_frames.append(_extract_total_series(path, disease_name))
    trend_df = pd.concat(trend_frames, ignore_index=True)
    trend_plot_df = trend_df.copy()
    non_covid_mask = trend_plot_df["Disease"].astype("string").str.strip().eq("Non-COVID Pneumonia")
    trend_plot_df = trend_plot_df[
        ~non_covid_mask | trend_plot_df[YEAR_COL].between(2019, 2025)
    ].copy()

    apc_df = pd.read_csv(APC_TABLE_PATH)
    apc_df["Start Year"] = pd.to_numeric(apc_df["Start Year"], errors="coerce")
    apc_df["End Year"] = pd.to_numeric(apc_df["End Year"], errors="coerce")
    apc_df["APC (% per year)"] = pd.to_numeric(apc_df["APC (% per year)"], errors="coerce")
    apc_df["APC Lower 95% CI"] = pd.to_numeric(apc_df["APC Lower 95% CI"], errors="coerce")
    apc_df["APC Upper 95% CI"] = pd.to_numeric(apc_df["APC Upper 95% CI"], errors="coerce")
    apc_df["Segment"] = pd.to_numeric(apc_df["Segment"], errors="coerce").astype("Int64")

    fig, ax = plt.subplots(figsize=(10.4, 6.2))

    for disease_name in DATASETS.keys():
        disease_series = trend_plot_df[trend_plot_df["Disease"].eq(disease_name)].sort_values(YEAR_COL)
        if disease_series.empty:
            continue
        ax.plot(
            disease_series[YEAR_COL],
            disease_series[RATE_COL],
            marker="o",
            linewidth=2.0,
            markersize=6,
            linestyle="--" if disease_name == "Non-COVID Pneumonia" else "-",
            color=PLOT_COLORS[disease_name],
            label=disease_name,
        )
    ax.axvspan(COVID_START, COVID_END, color=SHADE_COLOR, alpha=0.26, zorder=0)
    ax.axvline(2019, color="#77736C", linestyle=":", linewidth=0.95)
    ax.axvline(2021, color="#77736C", linestyle=":", linewidth=0.95)

    for disease, disease_df in trend_plot_df.groupby("Disease", sort=False):
        disease_apc = apc_df[apc_df["Disease"].astype("string").str.strip().eq(disease)].copy()
        if disease_apc.empty:
            continue

        disease_years = disease_df[YEAR_COL].to_numpy(dtype=float)
        disease_rates = disease_df[RATE_COL].to_numpy(dtype=float)
        disease_apc = disease_apc.sort_values("Segment")
        offsets = SEGMENT_OFFSETS.get(disease, [0, 0, 0])

        for idx, (_, row) in enumerate(disease_apc.iterrows()):
            start_year = float(row["Start Year"])
            end_year = float(row["End Year"])
            x_mid = (start_year + end_year) / 2.0
            y_mid = _interpolate_y(disease_years, disease_rates, x_mid)
            label = _format_apc_label(
                float(row["APC (% per year)"]),
                float(row["APC Lower 95% CI"]),
                float(row["APC Upper 95% CI"]),
            )
            y_offset = offsets[idx] if idx < len(offsets) else 0

            ax.annotate(
                label,
                xy=(x_mid, y_mid),
                xytext=(0, y_offset),
                textcoords="offset points",
                ha="center",
                va="center",
                fontsize=8.3,
                color=PLOT_COLORS[disease],
                bbox={
                    "boxstyle": "round,pad=0.2",
                    "facecolor": "white",
                    "edgecolor": PLOT_COLORS[disease],
                    "alpha": 0.75,
                    "linewidth": 0.7,
                },
            )

    ax.set_title("Overall mortality trend with segmented APC (95% CI)", loc="left", pad=12, color=TEXT_COLOR)
    ax.set_xlabel("Year")
    ax.set_ylabel("Age-adjusted rate")
    ax.set_xticks(sorted(trend_plot_df[YEAR_COL].dropna().unique()))
    ax.grid(axis="y", alpha=0.8)
    ax.grid(axis="x", visible=False)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        labels,
        title=None,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.14),
        ncol=4,
        handlelength=2.2,
        columnspacing=1.4,
    )
    ax.text(
        0.98,
        0.97,
        "Shaded area: COVID era (2020-2023)",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.8,
        color=TEXT_COLOR,
        bbox={
            "boxstyle": "round,pad=0.18",
            "facecolor": "white",
            "edgecolor": "#C8C1B4",
            "alpha": 0.82,
            "linewidth": 0.7,
        },
    )

    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "total_trend_three_diseases_with_apc.png", dpi=300)
    fig.savefig(FIGURE_DIR / "total_trend_three_diseases_with_apc.pdf")
    plt.close(fig)


if __name__ == "__main__":
    build_total_trend_with_apc_plot()
