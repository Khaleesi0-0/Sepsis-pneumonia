from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec


ROOT = Path(__file__).resolve().parents[2]
CLEANED_DIR = ROOT / "data" / "cleaned"
TABLE_DIR = ROOT / "results" / "tables"
FIGURE_DIR = ROOT / "results" / "figures" / "monthly_regression"

YEAR_COL = "Year"
MONTH_COL = "Month"
MONTH_CODE_COL = "Month Code"
AAMR_COL = "Age Adjusted Rate"
PREDICTED_COL = "Predicted AAMR"

ACTUAL_START_YEAR = 2010
ACTUAL_END_YEAR = 2025
PREDICTED_START_YEAR = 2018
SHADE_START = pd.Timestamp("2020-01-01")
SHADE_END = pd.Timestamp("2023-12-31")

VALIDATION_PATH = TABLE_DIR / "monthly_aamr_validation_2018_2019.csv"
PREDICTION_PATH = TABLE_DIR / "monthly_aamr_predicted_2020_2025.csv"

DATASETS = {
    "ARDS (ARDS + Pneumonia)": CLEANED_DIR / "ards_month.csv",
    "Pneumonia": CLEANED_DIR / "pneumonia_month.csv",
    "Sepsis+ Pneumonia": CLEANED_DIR / "combined_month.csv",
}

PANEL_LAYOUT = [
    ("A", "ARDS (ARDS + Pneumonia)", "ARDS (ARDS + Pneumonia)", 0, 0, 1, 2),
    ("B", "Pneumonia", "Pneumonia", 0, 2, 1, 2),
    ("C", "Sepsis+ Pneumonia", "Sepsis+ Pneumonia", 1, 1, 1, 2),
]

OBSERVED_COLOR = "#17365D"
PREDICTED_COLOR = "#B14A3B"
SHADE_COLOR = "#D8C7A3"
TEXT_COLOR = "#1F1A17"
ANNOTATION_COLOR = "#4F4A43"
REFERENCE_COLOR = "#77736C"


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


def _add_plot_date(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if MONTH_CODE_COL in out.columns:
        out["Plot Date"] = pd.to_datetime(
            out[MONTH_CODE_COL].astype("string").str.replace("/", "-", regex=False) + "-01",
            errors="coerce",
        )

    if "Plot Date" not in out.columns or out["Plot Date"].isna().any():
        missing_date = out.get("Plot Date", pd.Series(pd.NaT, index=out.index)).isna()
        out.loc[missing_date, "Plot Date"] = pd.to_datetime(
            {
                "year": out.loc[missing_date, YEAR_COL],
                "month": out.loc[missing_date, "Month Number"],
                "day": 1,
            },
            errors="coerce",
        )
    return out


def _read_actual_monthly_aamr(csv_path: Path, disease: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, dtype={YEAR_COL: "string", MONTH_CODE_COL: "string"})
    df[YEAR_COL] = (
        df[YEAR_COL]
        .astype("string")
        .str.strip()
        .str.replace(r"\s*\(provisional\)$", "", regex=True)
    )
    df[YEAR_COL] = pd.to_numeric(df[YEAR_COL], errors="coerce")
    df["Month Number"] = _month_number_from_code(df[MONTH_CODE_COL])
    df[AAMR_COL] = pd.to_numeric(df[AAMR_COL], errors="coerce")
    df = df.dropna(subset=[YEAR_COL, "Month Number", AAMR_COL]).copy()
    df[YEAR_COL] = df[YEAR_COL].astype(int)
    df["Month Number"] = df["Month Number"].astype(int)
    df["Disease"] = disease
    df["Series"] = "Observed monthly AAMR"
    df["AAMR"] = df[AAMR_COL]
    df = df[df[YEAR_COL].between(ACTUAL_START_YEAR, ACTUAL_END_YEAR)].copy()
    return _add_plot_date(df)


def _read_predicted_monthly_aamr() -> pd.DataFrame:
    validation_df = pd.read_csv(
        VALIDATION_PATH,
        dtype={YEAR_COL: "string", MONTH_CODE_COL: "string"},
    ).assign(Period="Validation")
    prediction_df = pd.read_csv(
        PREDICTION_PATH,
        dtype={YEAR_COL: "string", MONTH_CODE_COL: "string"},
    ).assign(Period="Prediction")

    predicted_df = pd.concat([validation_df, prediction_df], ignore_index=True, sort=False)
    predicted_df[YEAR_COL] = (
        predicted_df[YEAR_COL]
        .astype("string")
        .str.strip()
        .str.replace(r"\s*\(provisional\)$", "", regex=True)
    )
    predicted_df[YEAR_COL] = pd.to_numeric(predicted_df[YEAR_COL], errors="coerce")
    if "Month Number" not in predicted_df.columns:
        predicted_df["Month Number"] = _month_number_from_code(predicted_df[MONTH_CODE_COL])
    else:
        predicted_df["Month Number"] = pd.to_numeric(predicted_df["Month Number"], errors="coerce")
    predicted_df[PREDICTED_COL] = pd.to_numeric(predicted_df[PREDICTED_COL], errors="coerce")
    predicted_df = predicted_df.dropna(
        subset=[YEAR_COL, "Month Number", PREDICTED_COL, "Disease"]
    ).copy()
    predicted_df[YEAR_COL] = predicted_df[YEAR_COL].astype(int)
    predicted_df["Month Number"] = predicted_df["Month Number"].astype(int)
    predicted_df = predicted_df[
        predicted_df[YEAR_COL].between(PREDICTED_START_YEAR, ACTUAL_END_YEAR)
    ].copy()
    predicted_df["Series"] = "Predicted baseline"
    predicted_df["AAMR"] = predicted_df[PREDICTED_COL]
    return _add_plot_date(predicted_df)


def _build_trend_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    actual_df = pd.concat(
        [
            _read_actual_monthly_aamr(csv_path, disease)
            for disease, csv_path in DATASETS.items()
        ],
        ignore_index=True,
        sort=False,
    )
    predicted_df = _read_predicted_monthly_aamr()

    trend_df = pd.concat(
        [
            actual_df[["Disease", "Plot Date", "Series", "AAMR"]],
            predicted_df[["Disease", "Plot Date", "Series", "AAMR"]],
        ],
        ignore_index=True,
    ).dropna(subset=["Disease", "Plot Date", "Series", "AAMR"])

    return trend_df, predicted_df


def _set_axis_limits(ax: plt.Axes, disease_df: pd.DataFrame) -> None:
    y_min = disease_df["AAMR"].min()
    y_max = disease_df["AAMR"].max()
    y_pad = max((y_max - y_min) * 0.13, 0.20)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.set_xlim(
        pd.Timestamp(f"{ACTUAL_START_YEAR}-01-01"),
        pd.Timestamp(f"{ACTUAL_END_YEAR}-12-31"),
    )


def _draw_panel(
    ax: plt.Axes,
    trend_df: pd.DataFrame,
    predicted_df: pd.DataFrame,
    disease: str,
    panel_label: str,
    panel_title: str,
) -> None:
    disease_df = trend_df[trend_df["Disease"].eq(disease)].sort_values("Plot Date")
    actual_series = disease_df[disease_df["Series"].eq("Observed monthly AAMR")]
    predicted_series = disease_df[disease_df["Series"].eq("Predicted baseline")]
    model_label = (
        predicted_df.loc[predicted_df["Disease"].eq(disease), "Selected Model Label"]
        .dropna()
        .iloc[0]
    )

    ax.axvspan(SHADE_START, SHADE_END, color=SHADE_COLOR, alpha=0.26, zorder=0)
    ax.plot(
        actual_series["Plot Date"],
        actual_series["AAMR"],
        color=OBSERVED_COLOR,
        linewidth=1.75,
        label="Observed monthly AAMR",
        zorder=3,
    )
    ax.plot(
        predicted_series["Plot Date"],
        predicted_series["AAMR"],
        color=PREDICTED_COLOR,
        linewidth=1.85,
        linestyle=(0, (5, 2.4)),
        label="Predicted baseline",
        zorder=4,
    )
    ax.axvline(pd.Timestamp("2018-01-01"), color=REFERENCE_COLOR, linestyle=":", linewidth=0.95)
    ax.axvline(pd.Timestamp("2020-01-01"), color=REFERENCE_COLOR, linestyle=":", linewidth=0.95)

    _set_axis_limits(ax, disease_df)
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min

    ax.set_title(panel_title, loc="left", pad=15)
    ax.text(
        -0.035,
        1.09,
        panel_label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=13,
        fontweight="bold",
        color=TEXT_COLOR,
    )
    ax.text(
        0.0,
        1.015,
        model_label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.4,
        color=ANNOTATION_COLOR,
        style="italic",
    )
    ax.text(
        pd.Timestamp("2021-12-31"),
        y_max - (0.075 * y_range),
        "2020-2023",
        ha="center",
        va="center",
        fontsize=8.7,
        color="#6F5E3D",
    )

    ax.set_xlabel("")
    ax.set_ylabel("Monthly AAMR")
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(axis="y", alpha=0.8)
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="x", rotation=45)
    ax.margins(x=0.01)


def build_monthly_aamr_actual_predicted_plot() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    _apply_publication_style()
    trend_df, predicted_df = _build_trend_data()

    fig = plt.figure(figsize=(12.2, 8.6))
    grid = GridSpec(
        2,
        4,
        figure=fig,
        height_ratios=[1.0, 1.08],
        hspace=0.46,
        wspace=0.28,
    )
    axes = []

    for panel_label, disease, panel_title, row, col, rowspan, colspan in PANEL_LAYOUT:
        ax = fig.add_subplot(grid[row : row + rowspan, col : col + colspan])
        _draw_panel(ax, trend_df, predicted_df, disease, panel_label, panel_title)
        axes.append(ax)

    axes[1].set_ylabel("")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=2,
        frameon=False,
        handlelength=2.8,
        columnspacing=1.7,
    )
    fig.supxlabel("Year", y=0.055, fontsize=10.5)
    fig.supylabel("Monthly age-adjusted mortality rate per 100,000", x=0.018, fontsize=10.5)
    fig.suptitle(
        "Observed and predicted monthly age-adjusted mortality rates, 2010-2025",
        y=0.985,
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.945,
        "Predicted baseline begins in 2018; shaded region marks 2020-2023.",
        ha="center",
        va="center",
        fontsize=9.2,
        color=ANNOTATION_COLOR,
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=0.86, bottom=0.11)
    fig.savefig(FIGURE_DIR / "monthly_aamr_actual_2010_2025_predicted_from_2018.png", dpi=300)
    fig.savefig(FIGURE_DIR / "monthly_aamr_actual_2010_2025_predicted_from_2018.pdf")
    plt.close(fig)


if __name__ == "__main__":
    build_monthly_aamr_actual_predicted_plot()
