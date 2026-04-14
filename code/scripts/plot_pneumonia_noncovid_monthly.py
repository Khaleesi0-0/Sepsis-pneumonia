from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[2]
CLEANED_DIR = ROOT / "data" / "cleaned"
TABLE_DIR = ROOT / "results" / "tables"
FIGURE_DIR = ROOT / "results" / "figures" / "monthly_regression"

YEAR_COL = "Year"
MONTH_CODE_COL = "Month Code"
AAMR_COL = "Age Adjusted Rate"
PREDICTED_COL = "Predicted AAMR"

START_YEAR = 2018
END_YEAR = 2025
PREDICTED_START_YEAR = 2018
COVID_START = pd.Timestamp("2020-01-01")
COVID_SHADE_END = pd.Timestamp("2023-12-31")
OBSERVED_COLOR = "#17365D"
PREDICTED_COLOR = "#B14A3B"
COMPARISON_COLOR = "#2A6F4F"
SHADE_COLOR = "#D8C7A3"
TEXT_COLOR = "#1F1A17"
ANNOTATION_COLOR = "#4F4A43"
REFERENCE_COLOR = "#77736C"

PNEUMONIA_PATH = CLEANED_DIR / "pneumonia_month.csv"
COVID_PNEUMONIA_PATH = CLEANED_DIR / "pneumonia_covid_month.csv"
PREDICTION_PATH = TABLE_DIR / "monthly_aamr_predicted_2020_2025.csv"


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


def _month_number_from_code(month_code: pd.Series) -> pd.Series:
    return pd.to_numeric(
        month_code.astype("string").str.extract(r"/(\d{2})$", expand=False),
        errors="coerce",
    )


def _normalize_monthly_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out[YEAR_COL] = (
        out[YEAR_COL]
        .astype("string")
        .str.strip()
        .str.replace(r"\s*\(provisional\)$", "", regex=True)
    )
    out[YEAR_COL] = pd.to_numeric(out[YEAR_COL], errors="coerce")
    out["Month Number"] = _month_number_from_code(out[MONTH_CODE_COL])
    out[AAMR_COL] = pd.to_numeric(out[AAMR_COL], errors="coerce")
    out = out.dropna(subset=[YEAR_COL, "Month Number"]).copy()
    out[YEAR_COL] = out[YEAR_COL].astype(int)
    out["Month Number"] = out["Month Number"].astype(int)
    out["Plot Date"] = pd.to_datetime(
        out[MONTH_CODE_COL].astype("string").str.replace("/", "-", regex=False) + "-01",
        errors="coerce",
    )
    out = out[out[YEAR_COL].between(START_YEAR, END_YEAR)].copy()
    return out


def _read_pneumonia_series() -> pd.DataFrame:
    pneumonia = _normalize_monthly_df(pd.read_csv(PNEUMONIA_PATH, dtype={YEAR_COL: "string", MONTH_CODE_COL: "string"}))
    covid = _normalize_monthly_df(pd.read_csv(COVID_PNEUMONIA_PATH, dtype={YEAR_COL: "string", MONTH_CODE_COL: "string"}))

    pneumonia = pneumonia[[YEAR_COL, MONTH_CODE_COL, "Month Number", "Plot Date", AAMR_COL]].rename(
        columns={AAMR_COL: "Pneumonia AAMR"}
    )
    covid = covid[[YEAR_COL, MONTH_CODE_COL, AAMR_COL]].rename(
        columns={AAMR_COL: "COVID-related Pneumonia AAMR"}
    )

    merged = pneumonia.merge(covid, on=[YEAR_COL, MONTH_CODE_COL], how="left")
    merged["COVID-related Pneumonia AAMR"] = merged["COVID-related Pneumonia AAMR"].fillna(0)
    merged["Non-COVID Pneumonia AAMR"] = (
        merged["Pneumonia AAMR"] - merged["COVID-related Pneumonia AAMR"]
    ).clip(lower=0)
    return merged


def _read_pneumonia_predicted() -> tuple[pd.DataFrame, str]:
    pred = pd.read_csv(PREDICTION_PATH, dtype={YEAR_COL: "string", MONTH_CODE_COL: "string"})
    pred = pred[pred["Disease"].astype("string").str.strip().eq("Pneumonia")].copy()
    pred[YEAR_COL] = (
        pred[YEAR_COL]
        .astype("string")
        .str.strip()
        .str.replace(r"\s*\(provisional\)$", "", regex=True)
    )
    pred[YEAR_COL] = pd.to_numeric(pred[YEAR_COL], errors="coerce")
    pred["Month Number"] = _month_number_from_code(pred[MONTH_CODE_COL])
    pred[PREDICTED_COL] = pd.to_numeric(pred[PREDICTED_COL], errors="coerce")
    pred = pred.dropna(subset=[YEAR_COL, "Month Number", PREDICTED_COL]).copy()
    pred[YEAR_COL] = pred[YEAR_COL].astype(int)
    pred = pred[pred[YEAR_COL].between(PREDICTED_START_YEAR, END_YEAR)].copy()
    pred["Plot Date"] = pd.to_datetime(
        pred[MONTH_CODE_COL].astype("string").str.replace("/", "-", regex=False) + "-01",
        errors="coerce",
    )
    model_label = pred["Selected Model Label"].dropna().iloc[0] if not pred.empty else ""
    return pred, model_label


def build_pneumonia_noncovid_plot() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    _apply_publication_style()

    series_df = _read_pneumonia_series()
    pred_df, model_label = _read_pneumonia_predicted()

    fig, ax = plt.subplots(figsize=(11.2, 5.8))
    ax.axvspan(COVID_START, COVID_SHADE_END, color=SHADE_COLOR, alpha=0.26, zorder=0)

    ax.plot(
        series_df["Plot Date"],
        series_df["Pneumonia AAMR"],
        color=OBSERVED_COLOR,
        linewidth=1.9,
        label="Pneumonia (actual)",
        zorder=3,
    )
    ax.plot(
        series_df["Plot Date"],
        series_df["Non-COVID Pneumonia AAMR"],
        color=COMPARISON_COLOR,
        linewidth=1.9,
        label="Non-COVID pneumonia (actual)",
        zorder=3,
    )
    ax.plot(
        pred_df["Plot Date"],
        pred_df[PREDICTED_COL],
        color=PREDICTED_COLOR,
        linewidth=1.9,
        linestyle=(0, (5, 2.3)),
        label="Predicted baseline (model)",
        zorder=4,
    )

    ax.axvline(pd.Timestamp("2020-01-01"), color=REFERENCE_COLOR, linestyle=":", linewidth=0.95)
    ax.set_xlim(pd.Timestamp(f"{START_YEAR}-01-01"), pd.Timestamp(f"{END_YEAR}-12-31"))
    ax.set_xlabel("Year")
    ax.set_ylabel("Monthly AAMR")
    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.8)
    ax.grid(axis="x", visible=False)

    ax.set_title("Pneumonia and non-COVID pneumonia monthly AAMR, 2018-2025", loc="left", pad=28, color=TEXT_COLOR)
    if model_label:
        ax.text(
            0.0,
            1.09,
            f"Predicted baseline model: {model_label}",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8.5,
            color=ANNOTATION_COLOR,
            style="italic",
        )
    ax.text(
        pd.Timestamp("2021-12-31"),
        ax.get_ylim()[1] * 0.97,
        "COVID period (2020-2023)",
        ha="center",
        va="top",
        fontsize=8.5,
        color=ANNOTATION_COLOR,
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=3, frameon=False)

    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    fig.savefig(FIGURE_DIR / "pneumonia_vs_noncovid_monthly_aamr_2018_2025.png", dpi=300)
    fig.savefig(FIGURE_DIR / "pneumonia_vs_noncovid_monthly_aamr_2018_2025.pdf")
    plt.close(fig)


if __name__ == "__main__":
    build_pneumonia_noncovid_plot()
