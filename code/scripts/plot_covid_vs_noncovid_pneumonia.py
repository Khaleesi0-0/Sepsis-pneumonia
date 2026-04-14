from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[2]
CLEANED_DIR = ROOT / "data" / "cleaned"
FIGURE_DIR = ROOT / "results" / "figures" / "overall"

COVID_FILE = CLEANED_DIR / "pneumonia_covid_sex.csv"
NON_COVID_FILE = CLEANED_DIR / "non_covid_pneumonia_sex.csv"
PNEUMONIA_FILE = CLEANED_DIR / "pneumonia_sex.csv"

YEAR_COL = "Year"
RATE_COL = "Age Adjusted Rate"
NOTES_COL = "Notes"
COVID_START = 2020
COVID_END = 2023
SHADE_COLOR = "#D8C7A3"
TEXT_COLOR = "#1F1A17"
ANNOTATION_COLOR = "#4F4A43"
SERIES_ORDER = ["Pneumonia", "Pneumonia + COVID", "Non-COVID Pneumonia"]
SERIES_PALETTE = {
    "Pneumonia": "#4F4A43",
    "Pneumonia + COVID": "#B14A3B",
    "Non-COVID Pneumonia": "#17365D",
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
    _apply_publication_style()

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
        value_vars=SERIES_ORDER,
        var_name="Series",
        value_name=RATE_COL,
    )
    rel_df = merged[merged[YEAR_COL] >= 2020].copy()

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8))

    # Left panel: trend comparison over time.
    sns.lineplot(
        data=long_df,
        x=YEAR_COL,
        y=RATE_COL,
        hue="Series",
        marker="o",
        linewidth=2.0,
        markersize=5.5,
        dashes=False,
        hue_order=SERIES_ORDER,
        palette=SERIES_PALETTE,
        ax=axes[0],
    )
    axes[0].axvspan(COVID_START, COVID_END, color=SHADE_COLOR, alpha=0.26, zorder=0)
    axes[0].set_title("Pneumonia mortality trends", loc="left", pad=12, color=TEXT_COLOR)
    axes[0].set_xlabel("Year")
    axes[0].set_ylabel("Age-adjusted rate")
    axes[0].set_xticks(sorted(merged[YEAR_COL].unique()))
    axes[0].grid(axis="y", alpha=0.8)
    axes[0].grid(axis="x", visible=False)
    axes[0].tick_params(direction="out")
    axes[0].legend(
        title=None,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        handlelength=2.0,
        ncol=3,
        columnspacing=1.2,
    )

    # Right panel: direct relation between the two series.
    sns.regplot(
        data=rel_df,
        x="Pneumonia + COVID",
        y="Non-COVID Pneumonia",
        scatter_kws={"s": 40, "alpha": 0.9, "color": "#17365D"},
        line_kws={"color": "#4F4A43", "linewidth": 1.4},
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
    axes[1].set_title("Association during COVID-era years", loc="left", pad=12, color=TEXT_COLOR)
    axes[1].set_xlabel("Pneumonia + COVID age-adjusted rate")
    axes[1].set_ylabel("Non-COVID pneumonia age-adjusted rate")
    axes[1].grid(alpha=0.8)
    axes[1].tick_params(direction="out")

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

    for panel_label, ax in zip(["A", "B"], axes):
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

    fig.text(
        0.5,
        0.015,
        "Shaded area: COVID era (2020-2023)",
        ha="center",
        va="bottom",
        fontsize=9,
        color=TEXT_COLOR,
    )

    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "covid_vs_noncovid_pneumonia_relation.png", dpi=300)
    fig.savefig(FIGURE_DIR / "covid_vs_noncovid_pneumonia_relation.pdf")
    plt.close(fig)


if __name__ == "__main__":
    build_covid_noncovid_relation_plot()
