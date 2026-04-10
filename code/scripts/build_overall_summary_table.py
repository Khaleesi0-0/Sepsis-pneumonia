from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
CLEANED_DIR = ROOT / "data" / "cleaned"
OUTPUT_DIR = ROOT / "results" / "tables"
OUTPUT_PATH = OUTPUT_DIR / "overall_summary_table.csv"

YEAR_COL = "Year"
NOTES_COL = "Notes"
SEX_COL = "Sex"
AGE_COL = "Age Group"
RACE_COL = "Race 6"
POD_COL = "Place of Death"
DEATHS_COL = "Deaths"
POP_COL = "Population"
CRUDE_COL = "Crude Rate"
AAMR_COL = "Age Adjusted Rate"
AGE_AAMR_COL = "AAMR"

AGE_ORDER = ["<25", "25-44", "45-64", "65+"]
RACE_ORDER = [
    "American Indian or Alaska Native",
    "Asian",
    "Native Hawaiian or Other Pacific Islander",
    "Black or African American",
    "White",
    "More than one race",
]

METRICS = [
    "Total deaths, n",
    "Male deaths, n (%)",
    "Female deaths, n (%)",
    "Total population (standardizable ages)",
    "Crude mortality rate (standardizable ages), /100,000",
    "AAMR (all ages), /100,000",
    "Male AAMR (all ages), /100,000",
    "Female AAMR (all ages), /100,000",
    "Age 20–44 AAMR, /100,000",
    "Age 45–64 AAMR, /100,000",
    "Age <20 AAMR, /100,000",
    "Age ≥65 AAMR, /100,000",
    "American Indian or Alaska Native AAMR, /100,000",
    "Asian AAMR, /100,000",
    "Black or African American AAMR, /100,000",
    "More than one race AAMR, /100,000",
    "Native Hawaiian or Other Pacific Islander AAMR, /100,000",
    "White AAMR, /100,000",
    "Place of death: Medical Facility, n (%)",
    "Place of death: Non-medical facility, n (%)",
]

DATASETS = {
    "Sepsis": {
        "sex": CLEANED_DIR / "sepsis_sex.csv",
        "age": CLEANED_DIR / "sepsis_age.csv",
        "race": CLEANED_DIR / "sepsis_race.csv",
        "pod": CLEANED_DIR / "sepsis_pod.csv",
    },
    "Pneumonia": {
        "sex": CLEANED_DIR / "pneumonia_sex.csv",
        "age": CLEANED_DIR / "pneumonia_age.csv",
        "race": CLEANED_DIR / "pneumonia_race.csv",
        "pod": CLEANED_DIR / "pneumonia_pod.csv",
    },
    "Sepsis & pneumonia": {
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


def _aamr_column(df: pd.DataFrame) -> str:
    if AGE_AAMR_COL in df.columns:
        return AGE_AAMR_COL
    return AAMR_COL


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float | pd.NA:
    valid = values.notna() & weights.notna() & (weights > 0)
    if not valid.any():
        return pd.NA
    return (values[valid] * weights[valid]).sum() / weights[valid].sum()


def _fmt_n(value: float | int | pd.NA) -> str:
    if pd.isna(value):
        return ""
    return f"{int(round(float(value))):,}"


def _fmt_n_pct(value: float | int | pd.NA, denom: float | int | pd.NA) -> str:
    if pd.isna(value) or pd.isna(denom) or float(denom) == 0:
        return ""
    return f"{int(round(float(value))):,} ({(float(value) / float(denom) * 100):.1f}%)"


def _fmt_rate(value: float | int | pd.NA) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.1f}"


def _overall_from_sex(sex_df: pd.DataFrame) -> pd.DataFrame:
    total_rows = sex_df[sex_df[NOTES_COL].astype("string").str.strip().eq("Total")].copy()
    total_rows = total_rows[total_rows["__year"].notna()].copy()
    return total_rows


def _male_female_from_sex(sex_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    sex_rows = sex_df[sex_df["__year"].notna()].copy()
    female = sex_rows[sex_rows[SEX_COL].astype("string").str.strip().eq("Female")].copy()
    male = sex_rows[sex_rows[SEX_COL].astype("string").str.strip().eq("Male")].copy()
    return male, female


def _age_metric(period_age_df: pd.DataFrame, group_label: str) -> str:
    aamr_col = _aamr_column(period_age_df)
    if group_label == "Age 20–44 AAMR, /100,000":
        age_groups = ["<25", "25-44"]
    elif group_label == "Age 45–64 AAMR, /100,000":
        age_groups = ["45-64"]
    elif group_label == "Age <20 AAMR, /100,000":
        age_groups = ["<25"]
    else:
        age_groups = ["65+"]

    subset = period_age_df[period_age_df[AGE_COL].astype("string").str.strip().isin(age_groups)].copy()
    return _fmt_rate(_weighted_mean(subset[aamr_col], subset[POP_COL]))


def _race_metric(period_race_df: pd.DataFrame, race: str) -> str:
    aamr_col = _aamr_column(period_race_df)
    subset = period_race_df[period_race_df[RACE_COL].astype("string").str.strip().eq(race)].copy()
    return _fmt_rate(_weighted_mean(subset[aamr_col], subset[POP_COL]))


def _pod_metrics(period_pod_df: pd.DataFrame, total_deaths: float) -> tuple[str, str]:
    pod = period_pod_df.copy()
    pod[POD_COL] = pod[POD_COL].astype("string").str.strip()
    notes = pod[NOTES_COL].astype("string").str.strip()
    pod = pod[(notes.ne("Total")) | (notes.isna())].copy()
    pod = pod[pod[POD_COL].notna()]

    medical = pod[pod[POD_COL].str.startswith("Medical Facility", na=False)][DEATHS_COL].sum()
    non_medical = pod[~pod[POD_COL].str.startswith("Medical Facility", na=False)][DEATHS_COL].sum()
    return _fmt_n_pct(medical, total_deaths), _fmt_n_pct(non_medical, total_deaths)


def _build_outcome_table(outcome_name: str, paths: dict[str, Path]) -> pd.DataFrame:
    sex_df = _prepare(pd.read_csv(paths["sex"]))
    age_df = _prepare(pd.read_csv(paths["age"]))
    race_df = _prepare(pd.read_csv(paths["race"]))
    pod_df = _prepare(pd.read_csv(paths["pod"]))

    total_rows = _overall_from_sex(sex_df)
    male_rows, female_rows = _male_female_from_sex(sex_df)

    total_deaths = total_rows[DEATHS_COL].sum()
    total_population = total_rows[POP_COL].sum()

    metric_map = {
        "Total deaths, n": _fmt_n(total_deaths),
        "Male deaths, n (%)": _fmt_n_pct(male_rows[DEATHS_COL].sum(), total_deaths),
        "Female deaths, n (%)": _fmt_n_pct(female_rows[DEATHS_COL].sum(), total_deaths),
        "Total population (standardizable ages)": _fmt_n(total_population),
        "Crude mortality rate (standardizable ages), /100,000": _fmt_rate((total_deaths / total_population) * 100000 if total_population else pd.NA),
        "AAMR (all ages), /100,000": _fmt_rate(_weighted_mean(total_rows[AAMR_COL], total_rows[POP_COL])),
        "Male AAMR (all ages), /100,000": _fmt_rate(_weighted_mean(male_rows[AAMR_COL], male_rows[POP_COL])),
        "Female AAMR (all ages), /100,000": _fmt_rate(_weighted_mean(female_rows[AAMR_COL], female_rows[POP_COL])),
        "Age 20–44 AAMR, /100,000": _age_metric(age_df, "Age 20–44 AAMR, /100,000"),
        "Age 45–64 AAMR, /100,000": _age_metric(age_df, "Age 45–64 AAMR, /100,000"),
        "Age <20 AAMR, /100,000": _age_metric(age_df, "Age <20 AAMR, /100,000"),
        "Age ≥65 AAMR, /100,000": _age_metric(age_df, "Age ≥65 AAMR, /100,000"),
        "American Indian or Alaska Native AAMR, /100,000": _race_metric(race_df, "American Indian or Alaska Native"),
        "Asian AAMR, /100,000": _race_metric(race_df, "Asian"),
        "Black or African American AAMR, /100,000": _race_metric(race_df, "Black or African American"),
        "More than one race AAMR, /100,000": _race_metric(race_df, "More than one race"),
        "Native Hawaiian or Other Pacific Islander AAMR, /100,000": _race_metric(race_df, "Native Hawaiian or Other Pacific Islander"),
        "White AAMR, /100,000": _race_metric(race_df, "White"),
    }

    med, non_med = _pod_metrics(pod_df, total_deaths)
    metric_map["Place of death: Medical Facility, n (%)"] = med
    metric_map["Place of death: Non-medical facility, n (%)"] = non_med

    return pd.DataFrame(
        {
            "Metric": METRICS,
            outcome_name: [metric_map.get(metric, "") for metric in METRICS],
        }
    )


def build_table() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tables = [_build_outcome_table(outcome_name, paths) for outcome_name, paths in DATASETS.items()]
    merged = tables[0]
    for table in tables[1:]:
        merged = merged.merge(table, on="Metric", how="left")

    merged.to_csv(OUTPUT_PATH, index=False)


if __name__ == "__main__":
    build_table()
