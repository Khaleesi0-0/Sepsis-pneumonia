from pathlib import Path

import pandas as pd
import re


ROOT = Path(__file__).resolve().parents[2]
CLEANED_DIR = ROOT / "data" / "cleaned"
OUTPUT_DIR = ROOT / "results" / "tables" / "period_summary"

YEAR_COL = "Year"
NOTES_COL = "Notes"
SEX_COL = "Sex"
RACE_COL = "Race 6"
AGE_GROUP_COL = "Age Group"
POD_COL = "Place of Death"
DEATHS_COL = "Deaths"
POP_COL = "Population"
CRUDE_COL = "Crude Rate"
AAMR_COL = "Age Adjusted Rate"
AGE_AAMR_COL = "AAMR"

PERIODS = [
    ("Pre-pandemic (2010–2019)", range(2010, 2020)),
    ("Pandemic (2020–2023)", range(2020, 2024)),
    ("Post-pandemic (2024–2025)", range(2024, 2026)),
]

RACE_ORDER = [
    "American Indian or Alaska Native",
    "Asian",
    "Native Hawaiian or Other Pacific Islander",
    "Black or African American",
    "White",
    "More than one race",
]

AGE_ORDER = ["<25", "25-44", "45-64", "65+"]

METRIC_ORDER = [
    "Total deaths, n",
    "Male deaths, n (%)",
    "Female deaths, n (%)",
    "Population",
    "Crude mortality rate (/100,000)",
    "AAMR (/100,000), age-adjusted (all ages)",
    "Male AAMR (/100,000)",
    "Female AAMR (/100,000)",
    "<25 AAMR (/100,000)",
    "25–44 AAMR (/100,000)",
    "45–64 AAMR (/100,000)",
    "65+ AAMR (/100,000)",
    "American Indian or Alaska Native AAMR (/100,000)",
    "Asian AAMR (/100,000)",
    "Native Hawaiian or Other Pacific Islander AAMR (/100,000)",
    "Black or African American AAMR (/100,000)",
    "White AAMR (/100,000)",
    "More than one race AAMR (/100,000)",
    "Place of death: Medical Facility N(%)",
    "Place of death: Non-medical facility N(%)",
]

DATASETS = {
    "Pneumonia/ARDS": {
        "sex": CLEANED_DIR / "ards_sex.csv",
        "age": CLEANED_DIR / "ards_age.csv",
        "race": CLEANED_DIR / "ards_race.csv",
        "pod": CLEANED_DIR / "ards_pod.csv",
    },
    "Pneumonia": {
        "sex": CLEANED_DIR / "pneumonia_sex.csv",
        "age": CLEANED_DIR / "pneumonia_age.csv",
        "race": CLEANED_DIR / "pneumonia_race.csv",
        "pod": CLEANED_DIR / "pneumonia_pod.csv",
    },
    "Pneumonia/Sepsis": {
        "sex": CLEANED_DIR / "combined_sex.csv",
        "age": CLEANED_DIR / "combined_age.csv",
        "race": CLEANED_DIR / "combined_race.csv",
        "pod": CLEANED_DIR / "combined_pod.csv",
    },
}


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if YEAR_COL in out.columns:
        out[YEAR_COL] = (
            out[YEAR_COL]
            .astype("string")
            .str.strip()
            .str.replace(r"\s*\(provisional\)$", "", regex=True)
        )
        out["__year"] = pd.to_numeric(out[YEAR_COL], errors="coerce")
    else:
        out["__year"] = pd.Series(dtype="float64")

    for col in [DEATHS_COL, POP_COL, CRUDE_COL, AAMR_COL, AGE_AAMR_COL]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    return out


def _subset_period(df: pd.DataFrame, years: range) -> pd.DataFrame:
    return df[df["__year"].isin(list(years))].copy()


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float | None:
    valid = values.notna() & weights.notna() & (weights > 0)
    if not valid.any():
        return pd.NA
    return (values[valid] * weights[valid]).sum() / weights[valid].sum()


def _fmt_n(value: float | int | None) -> str:
    if pd.isna(value):
        return ""
    return f"{int(round(float(value))):,}"


def _fmt_n_pct(value: float | int | None, denom: float | int | None) -> str:
    if pd.isna(value) or pd.isna(denom) or float(denom) == 0:
        return ""
    pct = float(value) / float(denom) * 100
    return f"{int(round(float(value))):,} ({pct:.1f}%)"


def _fmt_rate(value: float | int | None) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.1f}"


def _sex_metrics(sex_df: pd.DataFrame, period_df: pd.DataFrame) -> dict[str, str]:
    totals = period_df[period_df[NOTES_COL].astype("string").str.strip().eq("Total")].copy()
    female = period_df[period_df[SEX_COL].astype("string").str.strip().eq("Female")].copy()
    male = period_df[period_df[SEX_COL].astype("string").str.strip().eq("Male")].copy()

    total_deaths = totals[DEATHS_COL].sum()
    total_pop = totals[POP_COL].sum()

    return {
        "Total deaths, n": _fmt_n(total_deaths),
        "Male deaths, n (%)": _fmt_n_pct(male[DEATHS_COL].sum(), total_deaths),
        "Female deaths, n (%)": _fmt_n_pct(female[DEATHS_COL].sum(), total_deaths),
        "Population": _fmt_n(total_pop),
        "Crude mortality rate (/100,000)": _fmt_rate((total_deaths / total_pop) * 100000 if total_pop else pd.NA),
        "AAMR (/100,000), age-adjusted (all ages)": _fmt_rate(_weighted_mean(totals[AAMR_COL], totals[POP_COL])),
        "Male AAMR (/100,000)": _fmt_rate(_weighted_mean(male[AAMR_COL], male[POP_COL])),
        "Female AAMR (/100,000)": _fmt_rate(_weighted_mean(female[AAMR_COL], female[POP_COL])),
    }


def _age_metrics(period_age_df: pd.DataFrame) -> dict[str, str]:
    out = {}
    for age_group in AGE_ORDER:
        subset = period_age_df[period_age_df[AGE_GROUP_COL].astype("string").str.strip().eq(age_group)].copy()
        label = f"{age_group} AAMR (/100,000)"
        if age_group == "25-44":
            label = "25–44 AAMR (/100,000)"
        if age_group == "45-64":
            label = "45–64 AAMR (/100,000)"
        out[label] = _fmt_rate(_weighted_mean(subset[AGE_AAMR_COL], subset[POP_COL]))
    return out


def _race_metrics(period_race_df: pd.DataFrame) -> dict[str, str]:
    out = {}
    for race in RACE_ORDER:
        subset = period_race_df[period_race_df[RACE_COL].astype("string").str.strip().eq(race)].copy()
        out[f"{race} AAMR (/100,000)"] = _fmt_rate(_weighted_mean(subset[AAMR_COL], subset[POP_COL]))
    return out


def _pod_metrics(period_pod_df: pd.DataFrame, total_deaths: float) -> dict[str, str]:
    pod = period_pod_df.copy()
    pod[POD_COL] = pod[POD_COL].astype("string").str.strip()
    notes = pod[NOTES_COL].astype("string").str.strip()
    pod = pod[(notes.ne("Total")) | (notes.isna())].copy()
    pod = pod[pod[POD_COL].notna()]

    medical = pod[pod[POD_COL].str.startswith("Medical Facility", na=False)][DEATHS_COL].sum()
    non_medical = pod[~pod[POD_COL].str.startswith("Medical Facility", na=False)][DEATHS_COL].sum()

    return {
        "Place of death: Medical Facility N(%)": _fmt_n_pct(medical, total_deaths),
        "Place of death: Non-medical facility N(%)": _fmt_n_pct(non_medical, total_deaths),
    }


def _build_outcome_table(outcome_name: str, paths: dict[str, Path]) -> pd.DataFrame:
    sex_df = _prepare(pd.read_csv(paths["sex"]))
    age_df = _prepare(pd.read_csv(paths["age"]))
    race_df = _prepare(pd.read_csv(paths["race"]))
    pod_df = _prepare(pd.read_csv(paths["pod"]))

    rows = []
    for label, years in PERIODS:
        sex_period = _subset_period(sex_df, years)
        age_period = _subset_period(age_df, years)
        race_period = _subset_period(race_df, years)
        pod_period = _subset_period(pod_df, years)

        sex_metrics = _sex_metrics(sex_df, sex_period)
        total_deaths_num = pd.to_numeric(sex_period[sex_period[NOTES_COL].astype("string").str.strip().eq("Total")][DEATHS_COL], errors="coerce").sum()
        period_metrics = {
            **sex_metrics,
            **_age_metrics(age_period),
            **_race_metrics(race_period),
            **_pod_metrics(pod_period, total_deaths_num),
        }

        for metric in METRIC_ORDER:
            rows.append({"Metric": metric, label: period_metrics.get(metric, "")})

    table = pd.DataFrame(rows)
    table = table.groupby("Metric", as_index=False).first()
    ordered_cols = ["outcome", "Metric"] + [label for label, _ in PERIODS]
    table.insert(0, "outcome", outcome_name)
    table["Metric"] = pd.Categorical(table["Metric"], categories=METRIC_ORDER, ordered=True)
    table = table.sort_values("Metric").reset_index(drop=True)
    return table[ordered_cols]


def build_tables() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_tables = []
    for outcome_name, paths in DATASETS.items():
        table = _build_outcome_table(outcome_name, paths)
        outcome_slug = re.sub(r"[^a-z0-9]+", "_", outcome_name.lower()).strip("_")
        if outcome_slug.startswith("ards_ards_pneumonia"):
            outcome_slug = "ards"
        table.drop(columns=["outcome"]).to_csv(
            OUTPUT_DIR / f"{outcome_slug}_period_summary_table.csv",
            index=False,
        )
        all_tables.append(table)

    pd.concat(all_tables, ignore_index=True).to_csv(OUTPUT_DIR / "all_outcomes_period_summary_table.csv", index=False)


if __name__ == "__main__":
    build_tables()

