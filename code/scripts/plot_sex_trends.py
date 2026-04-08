from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[2]
CLEANED_DIR = ROOT / "data" / "cleaned"
FIGURE_DIR = ROOT / "results" / "figures"

COVID_START = 2020
COVID_END = 2023

Y_COL = "Age Adjusted Rate"
SEX_COL = "Sex"
YEAR_COL = "Year"

DATASETS = {
    "sepsis": CLEANED_DIR / "sepsis_sex.csv",
    "pneumonia": CLEANED_DIR / "pneumonia_sex.csv",
    "combined": CLEANED_DIR / "combined_sex.csv",
}


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
    plot_df = plot_df.dropna(subset=[YEAR_COL, Y_COL, SEX_COL])

    # Safety aggregation in case there are repeated rows per year/sex.
    plot_df = (
        plot_df.groupby([YEAR_COL, SEX_COL], as_index=False)[Y_COL]
        .mean()
        .sort_values([YEAR_COL, SEX_COL])
    )
    return plot_df


def _draw_single_trend(ax: plt.Axes, df: pd.DataFrame, title: str) -> None:
    sns.lineplot(
        data=df,
        x=YEAR_COL,
        y=Y_COL,
        hue=SEX_COL,
        marker="o",
        linewidth=2,
        ax=ax,
    )
    ax.axvspan(COVID_START, COVID_END, color="gray", alpha=0.18, label="COVID era")
    ax.set_title(title)
    ax.set_xlabel("Year")
    ax.set_ylabel(Y_COL)
    ax.set_xticks(sorted(df[YEAR_COL].dropna().unique()))
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.6)

    # Keep legend compact and readable.
    handles, labels = ax.get_legend_handles_labels()
    dedup = {}
    for handle, label in zip(handles, labels):
        if label not in dedup:
            dedup[label] = handle
    ax.legend(
        dedup.values(),
        dedup.keys(),
        frameon=False,
        loc="best",
    )


def build_trend_figures() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    prepared = {}
    for name, csv_path in DATASETS.items():
        raw = pd.read_csv(csv_path)
        prepared[name] = _prepare_for_plot(raw)

    # 1) Three individual figures.
    for name, df in prepared.items():
        fig, ax = plt.subplots(figsize=(8, 5))
        _draw_single_trend(ax, df, f"{name.capitalize()} Trend by Sex")
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / f"{name}_trend_by_sex.png", dpi=300)
        plt.close(fig)

    # 2) One arranged multi-panel figure containing all three.
    fig, axes = plt.subplots(3, 1, figsize=(10, 14), sharex=True)
    panel_order = ["sepsis", "pneumonia", "combined"]
    panel_titles = {
        "sepsis": "Sepsis Trend by Sex",
        "pneumonia": "Pneumonia Trend by Sex",
        "combined": "Sepsis + Pneumonia (Other bacterial diseases) Trend by Sex",
    }

    for ax, key in zip(axes, panel_order):
        _draw_single_trend(ax, prepared[key], panel_titles[key])

    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "all_trends_by_sex_panel.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    build_trend_figures()

