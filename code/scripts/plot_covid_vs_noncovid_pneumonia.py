from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[2]
CLEANED_DIR = ROOT / "data" / "cleaned"
FIGURE_DIR = ROOT / "results" / "figures"

COVID_FILE = CLEANED_DIR / "pneumonia_covid_sex.csv"
NON_COVID_FILE = CLEANED_DIR / "non_covid_pneumonia_sex.csv"
PNEUMONIA_FILE = CLEANED_DIR / "pneumonia_sex.csv"

YEAR_COL = "Year"
RATE_COL = "Age Adjusted Rate"
NOTES_COL = "Notes"
COVID_START = 2020
COVID_END = 2023


def _load_total_series(csv_path: Path, value_name: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, dtype={YEAR_COL: "string"})

    total = df[df[NOTES_COL].astype("string").str.strip().eq("Total")].copy()
    total[YEAR_COL] = (
        total[YEAR_COL]
        .astype("string")
        .str.strip()
        .str.replace(r"\s*\(provisional\)$", "", regex=True)
    )
    total[YEAR_COL] = pd.to_numeric(total[YEAR_COL], errors="coerce")
    total[RATE_COL] = pd.to_numeric(total[RATE_COL], errors="coerce")
    total = total.dropna(subset=[YEAR_COL, RATE_COL])

    # Drop potential grand-total rows with no specific year.
    total = total[total[YEAR_COL].between(1900, 2100)]

    return total[[YEAR_COL, RATE_COL]].rename(columns={RATE_COL: value_name})


def build_covid_noncovid_relation_plot() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    covid = _load_total_series(COVID_FILE, "Pneumonia + COVID")
    non_covid = _load_total_series(NON_COVID_FILE, "Non-COVID Pneumonia")
    pneumonia = _load_total_series(PNEUMONIA_FILE, "Pneumonia")

    merged = covid.merge(non_covid, on=YEAR_COL, how="inner").sort_values(YEAR_COL)
    trend_df = (
        merged.merge(pneumonia, on=YEAR_COL, how="left")
        .sort_values(YEAR_COL)
    )
    long_df = trend_df.melt(
        id_vars=[YEAR_COL],
        value_vars=["Pneumonia + COVID", "Non-COVID Pneumonia", "Pneumonia"],
        var_name="Series",
        value_name="Age Adjusted Rate",
    )
    rel_df = merged[merged[YEAR_COL] >= 2020].copy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left panel: trend comparison over time.
    sns.lineplot(
        data=long_df,
        x=YEAR_COL,
        y="Age Adjusted Rate",
        hue="Series",
        marker="o",
        linewidth=2.2,
        ax=axes[0],
    )
    axes[0].axvspan(COVID_START, COVID_END, color="gray", alpha=0.18, label="COVID era")
    axes[0].set_title("Pneumonia Trends: COVID-inclusive vs Non-COVID")
    axes[0].set_xlabel("Year")
    axes[0].set_ylabel("Age Adjusted Rate")
    axes[0].set_xticks(sorted(merged[YEAR_COL].unique()))
    axes[0].legend(title="", frameon=False, loc="best")

    # Right panel: direct relation between the two series.
    sns.regplot(
        data=rel_df,
        x="Pneumonia + COVID",
        y="Non-COVID Pneumonia",
        scatter_kws={"s": 50, "alpha": 0.85},
        line_kws={"color": "black", "linewidth": 1.5},
        ci=None,
        ax=axes[1],
    )
    for _, row in rel_df.iterrows():
        axes[1].annotate(
            str(int(row[YEAR_COL])),
            (row["Pneumonia + COVID"], row["Non-COVID Pneumonia"]),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=8,
        )
    axes[1].set_title("Relationship Across Years")
    axes[1].set_xlabel("Pneumonia + COVID (Age Adjusted Rate)")
    axes[1].set_ylabel("Non-COVID Pneumonia (Age Adjusted Rate)")
    axes[1].grid(alpha=0.25, linestyle="--", linewidth=0.6)

    # Tighten axes around 2020+ points for readability.
    if not rel_df.empty:
        x_min = rel_df["Pneumonia + COVID"].min()
        x_max = rel_df["Pneumonia + COVID"].max()
        y_min = rel_df["Non-COVID Pneumonia"].min()
        y_max = rel_df["Non-COVID Pneumonia"].max()
        x_pad = max((x_max - x_min) * 0.08, 0.5)
        y_pad = max((y_max - y_min) * 0.08, 0.5)
        axes[1].set_xlim(x_min - x_pad, x_max + x_pad)
        axes[1].set_ylim(y_min - y_pad, y_max + y_pad)

    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "covid_vs_noncovid_pneumonia_relation.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    build_covid_noncovid_relation_plot()
