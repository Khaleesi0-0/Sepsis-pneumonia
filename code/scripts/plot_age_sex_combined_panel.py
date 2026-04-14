from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from plot_age_trends import (
    DATASETS as AGE_DATASETS,
    _apply_publication_style as apply_age_style,
    _draw_single_trend as draw_age_trend,
    _prepare_for_plot as prepare_age_plot,
)
from plot_sex_trends import (
    DATASETS as SEX_DATASETS,
    _draw_single_trend as draw_sex_trend,
    _prepare_for_plot as prepare_sex_plot,
)


ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIR = ROOT / "results" / "figures" / "overall"
TEXT_COLOR = "#1F1A17"
PANEL_LINEWIDTH = 1.00

PANEL_ORDER = ["pneumonia", "ards", "combined"]
DISEASE_TITLES = {
    "pneumonia": "Pneumonia",
    "ards": "Pneumonia/ARDS",
    "combined": "Pneumonia/Sepsis",
}


def build_age_sex_combined_panel() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    apply_age_style()

    prepared_sex = {}
    for name, csv_path in SEX_DATASETS.items():
        prepared_sex[name] = prepare_sex_plot(pd.read_csv(csv_path))

    prepared_age = {}
    for name, csv_path in AGE_DATASETS.items():
        prepared_age[name] = prepare_age_plot(pd.read_csv(csv_path))

    fig, axes = plt.subplots(2, 3, figsize=(15.2, 9.2), sharex=True, sharey=False)
    panel_labels = ["A", "B", "C", "D", "E", "F"]

    sex_legend_handles = None
    sex_legend_labels = None
    age_legend_handles = None
    age_legend_labels = None

    for col_idx, key in enumerate(PANEL_ORDER):
        sex_ax = axes[0, col_idx]
        draw_sex_trend(sex_ax, prepared_sex[key], f"{DISEASE_TITLES[key]}: Sex", show_legend=False)
        for line in sex_ax.lines:
            line.set_linewidth(PANEL_LINEWIDTH)
        sex_ax.set_xlabel("")
        sex_ax.set_ylabel("")
        sex_ax.text(
            0.01,
            0.98,
            panel_labels[col_idx],
            transform=sex_ax.transAxes,
            ha="left",
            va="top",
            fontsize=12,
            fontweight="bold",
            color=TEXT_COLOR,
        )
        if sex_legend_handles is None:
            sex_legend_handles, sex_legend_labels = sex_ax.get_legend_handles_labels()

        age_ax = axes[1, col_idx]
        draw_age_trend(age_ax, prepared_age[key], f"{DISEASE_TITLES[key]}: Age", show_legend=False)
        for line in age_ax.lines:
            line.set_linewidth(PANEL_LINEWIDTH)
        age_ax.set_xlabel("")
        age_ax.set_ylabel("")
        age_ax.text(
            0.01,
            0.98,
            panel_labels[col_idx + 3],
            transform=age_ax.transAxes,
            ha="left",
            va="top",
            fontsize=12,
            fontweight="bold",
            color=TEXT_COLOR,
        )
        if age_legend_handles is None:
            age_legend_handles, age_legend_labels = age_ax.get_legend_handles_labels()

    fig.legend(
        sex_legend_handles[:3],
        sex_legend_labels[:3],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.92),
        ncol=3,
        frameon=False,
        handlelength=2.0,
        columnspacing=1.2,
    )
    fig.legend(
        age_legend_handles,
        age_legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.515),
        ncol=4,
        frameon=False,
        handlelength=2.0,
        columnspacing=1.2,
    )
    fig.suptitle(
        "Observed age- and sex-specific age-adjusted mortality trends, 2010-2025",
        y=0.986,
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.958,
        "Top row: sex-specific rates. Bottom row: age-specific rates.",
        ha="center",
        va="center",
        fontsize=9.2,
        color="#4F4A43",
    )
    fig.supxlabel("Year", y=0.02, fontsize=10.5)
    fig.text(0.014, 0.73, "Age-adjusted rate", rotation=90, va="center", ha="center", fontsize=10.5)
    fig.text(0.014, 0.28, "Age-adjusted rate", rotation=90, va="center", ha="center", fontsize=10.5)
    # Reserve dedicated vertical bands for suptitle + legends to avoid overlap.
    fig.subplots_adjust(left=0.06, right=0.99, top=0.86, bottom=0.11, hspace=0.40, wspace=0.24)
    fig.savefig(FIGURE_DIR / "all_trends_by_age_and_sex_panel.png", dpi=300)
    fig.savefig(FIGURE_DIR / "all_trends_by_age_and_sex_panel.pdf")
    plt.close(fig)


if __name__ == "__main__":
    build_age_sex_combined_panel()

