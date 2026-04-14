from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
CLEANED_DIR = ROOT / "data" / "cleaned"
TABLE_DIR = ROOT / "results" / "tables"
OUTPUT_PATH = TABLE_DIR / "outcome_year_summary_table.csv"

YEAR_COL = "Year"
NOTES_COL = "Notes"
SEX_COL = "Sex"
RACE_COL = "Race 6"
DEATHS_COL = "Deaths"
POP_COL = "Population"
CRUDE_COL = "Crude Rate"
AAMR_COL = "Age Adjusted Rate"

RACE_ORDER = [
    "American Indian or Alaska Native",
    "Asian",
    "Native Hawaiian or Other Pacific Islander",
    "Black or African American",
    "White",
    "More than one race",
]

OUTPUT_COLUMNS = [
    "outcome",
    "Year",
    "Overall (n)",
    "Women (n)",
    "Men (n)",
    "American Indian or Alaska Native (n)",
    "Asian (n)",
    "Native Hawaiian or Other Pacific Islander (n)",
    "Black or African American (n)",
    "White (n)",
    "More than one race (n)",
    "Population",
    "Crude rate (/100,000)",
    "AAMR (/100,000), age-adjusted (all ages)",
]

DATASETS = {
    "Pneumonia/ARDS": {
        "sex": CLEANED_DIR / "ards_sex.csv",
        "race": CLEANED_DIR / "ards_race.csv",
    },
    "Pneumonia": {
        "sex": CLEANED_DIR / "pneumonia_sex.csv",
        "race": CLEANED_DIR / "pneumonia_race.csv",
    },
    "Pneumonia/Sepsis": {
        "sex": CLEANED_DIR / "combined_sex.csv",
        "race": CLEANED_DIR / "combined_race.csv",
    },
}


def _read_numeric_year(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out[YEAR_COL] = (
        out[YEAR_COL]
        .astype("string")
        .str.strip()
        .str.replace(r"\s*\(provisional\)$", "", regex=True)
    )
    out[YEAR_COL] = pd.to_numeric(out[YEAR_COL], errors="coerce")
    out[DEATHS_COL] = pd.to_numeric(out.get(DEATHS_COL), errors="coerce")
    out[POP_COL] = pd.to_numeric(out.get(POP_COL), errors="coerce")
    out[CRUDE_COL] = pd.to_numeric(out.get(CRUDE_COL), errors="coerce")
    out[AAMR_COL] = pd.to_numeric(out.get(AAMR_COL), errors="coerce")
    return out


def _yearly_overall_from_sex(sex_df: pd.DataFrame) -> pd.DataFrame:
    overall = sex_df[
        sex_df[NOTES_COL].astype("string").str.strip().eq("Total")
        & sex_df[YEAR_COL].notna()
    ].copy()
    return overall[[YEAR_COL, DEATHS_COL, POP_COL, CRUDE_COL, AAMR_COL]]


def _yearly_sex_counts(sex_df: pd.DataFrame) -> pd.DataFrame:
    sex_rows = sex_df[
        sex_df[YEAR_COL].notna() & sex_df[SEX_COL].astype("string").str.strip().isin(["Female", "Male"])
    ].copy()
    grouped = (
        sex_rows.groupby([YEAR_COL, SEX_COL], as_index=False)[DEATHS_COL]
        .sum()
        .pivot(index=YEAR_COL, columns=SEX_COL, values=DEATHS_COL)
        .fillna(0)
        .reset_index()
    )
    grouped = grouped.rename(columns={"Female": "Women (n)", "Male": "Men (n)"})
    return grouped


def _yearly_race_counts(race_df: pd.DataFrame) -> pd.DataFrame:
    race_rows = race_df[race_df[YEAR_COL].notna()].copy()
    race_rows[RACE_COL] = race_rows[RACE_COL].astype("string").str.strip()
    race_rows = race_rows[race_rows[RACE_COL].isin(RACE_ORDER)].copy()
    grouped = (
        race_rows.groupby([YEAR_COL, RACE_COL], as_index=False)[DEATHS_COL]
        .sum()
        .pivot(index=YEAR_COL, columns=RACE_COL, values=DEATHS_COL)
        .fillna(0)
        .reset_index()
    )
    rename_map = {race: f"{race} (n)" for race in RACE_ORDER}
    grouped = grouped.rename(columns=rename_map)
    return grouped


def _build_total_row(outcome_name: str, yearly_df: pd.DataFrame, sex_df: pd.DataFrame) -> dict:
    row = {"outcome": outcome_name, "Year": "Total"}

    for col in [
        "Overall (n)",
        "Women (n)",
        "Men (n)",
        "American Indian or Alaska Native (n)",
        "Asian (n)",
        "Native Hawaiian or Other Pacific Islander (n)",
        "Black or African American (n)",
        "White (n)",
        "More than one race (n)",
        "Population",
    ]:
        row[col] = pd.to_numeric(yearly_df[col], errors="coerce").sum()

    row["Crude rate (/100,000)"] = (
        (row["Overall (n)"] / row["Population"]) * 100000 if row["Population"] else pd.NA
    )

    grand_total = sex_df[
        sex_df[NOTES_COL].astype("string").str.strip().eq("Total") & sex_df[YEAR_COL].isna()
    ].copy()
    if not grand_total.empty and pd.notna(grand_total[AAMR_COL].iloc[0]):
        row["AAMR (/100,000), age-adjusted (all ages)"] = grand_total[AAMR_COL].iloc[0]
    else:
        weights = pd.to_numeric(yearly_df["Population"], errors="coerce")
        values = pd.to_numeric(yearly_df["AAMR (/100,000), age-adjusted (all ages)"], errors="coerce")
        valid = weights.notna() & values.notna() & (weights > 0)
        row["AAMR (/100,000), age-adjusted (all ages)"] = (
            (values[valid] * weights[valid]).sum() / weights[valid].sum()
            if valid.any()
            else pd.NA
        )

    return row


def _build_outcome_table(outcome_name: str, sex_path: Path, race_path: Path) -> pd.DataFrame:
    sex_df = _read_numeric_year(pd.read_csv(sex_path))
    race_df = _read_numeric_year(pd.read_csv(race_path))

    overall = _yearly_overall_from_sex(sex_df).rename(
        columns={
            DEATHS_COL: "Overall (n)",
            POP_COL: "Population",
            CRUDE_COL: "Crude rate (/100,000)",
            AAMR_COL: "AAMR (/100,000), age-adjusted (all ages)",
        }
    )
    sex_counts = _yearly_sex_counts(sex_df)
    race_counts = _yearly_race_counts(race_df)

    merged = overall.merge(sex_counts, on=YEAR_COL, how="left").merge(race_counts, on=YEAR_COL, how="left")
    merged["outcome"] = outcome_name
    merged = merged.rename(columns={YEAR_COL: "__year_num"})
    merged["Year"] = merged["__year_num"].astype("Int64").astype("string")
    merged = merged.drop(columns=["__year_num"])

    for col in OUTPUT_COLUMNS:
        if col not in merged.columns:
            merged[col] = 0

    merged = merged[OUTPUT_COLUMNS].sort_values("Year").reset_index(drop=True)
    total_row = _build_total_row(outcome_name, merged, sex_df)
    merged = pd.concat([merged, pd.DataFrame([total_row])], ignore_index=True)

    return merged


def build_table() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    for outcome_name, paths in DATASETS.items():
        frames.append(_build_outcome_table(outcome_name, paths["sex"], paths["race"]))

    table = pd.concat(frames, ignore_index=True)
    table.to_csv(OUTPUT_PATH, index=False)


if __name__ == "__main__":
    build_table()

