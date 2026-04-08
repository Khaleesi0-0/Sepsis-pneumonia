from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[2]
CLEANED_DIR = ROOT / "data" / "cleaned"
FIGURE_DIR = ROOT / "results" / "figures"

YEAR_COL = "Year"
VALUE_COL = "Age Adjusted Rate"
COVID_START = 2020
COVID_END = 2023

DATASETS = {
    "Sepsis": CLEANED_DIR / "sepsis_sex.csv",
    "Pneumonia": CLEANED_DIR / "pneumonia_sex.csv",
    "Combined": CLEANED_DIR / "combined_sex.csv",
}


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
    sns.set_theme(style="whitegrid")

    trend_frames = []
    for disease_name, path in DATASETS.items():
        trend_frames.append(_extract_total_series(path, disease_name))

    trend_df = pd.concat(trend_frames, ignore_index=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(
        data=trend_df,
        x=YEAR_COL,
        y=VALUE_COL,
        hue="Disease",
        marker="o",
        linewidth=2.2,
        ax=ax,
    )
    ax.axvspan(COVID_START, COVID_END, color="gray", alpha=0.18, label="COVID era")

    ax.set_title("Total Trend by Disease")
    ax.set_xlabel("Year")
    ax.set_ylabel("Age Adjusted Rate")
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.6)
    ax.legend(title="", frameon=False, loc="best")
    ax.set_xticks(sorted(trend_df[YEAR_COL].dropna().unique()))

    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "total_trend_three_diseases.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    build_total_trend_plot()
