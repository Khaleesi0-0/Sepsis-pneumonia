from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PRIMARY_DIR = ROOT / "data" / "primary"
OUTPUT_DIR = ROOT / "data" / "cleaned"

SEPSIS_FILES = ["Amonth2010.csv", "Amonth2018.csv"]
PNEUMONIA_FILES = ["Jmonth2010.csv", "Jmonth2018.csv"]
COMBINED_FILES = ["AJmonth2010.csv", "AJmonth2018.csv"]
UJ_COVID_PNEUMONIA_FILES = ["UJmonth.csv"]

SEPSIS_MONTHAGE_FILES = ["Amonthage2010.csv", "Amonthage2018.csv"]
PNEUMONIA_MONTHAGE_FILES = ["Jmonthage2010.csv", "Jmonthage2018.csv"]
COMBINED_MONTHAGE_FILES = ["AJmonthage2010.csv", "AJmonthage2018.csv"]
UJ_COVID_PNEUMONIA_MONTHAGE_FILES = ["UJmonthage.csv"]

SEPSIS_AGE_FILES = ["Aage2010.csv", "Aage2018.csv"]
PNEUMONIA_AGE_FILES = ["Jage2010.csv", "Jage2018.csv"]
COMBINED_AGE_FILES = ["AJage2010.csv", "AJage2018.csv"]

SEX_REFERENCE_FILES = {
    "sepsis": "sepsis_sex.csv",
    "pneumonia": "pneumonia_sex.csv",
    "combined": "combined_sex.csv",
    "covid_pneumonia": "UJsex2018.csv",
}

TARGET_SUBCHAPTER = "Other bacterial diseases"
PNEUMONIA_SUBCHAPTER = "Influenza and pneumonia"
SUBCHAPTER_COL = "MCD - ICD Sub-Chapter"
YEAR_COL = "Year"
YEAR_CODE_COL = "Year Code"
MONTH_COL = "Month"
NOTES_COL = "Notes"
DEATHS_COL = "Deaths"
POPULATION_COL = "Population"
CRUDE_RATE_COL = "Crude Rate"
AAMR_COL = "Age Adjusted Rate"
AGE_COL = "Ten-Year Age Groups"
AGE_CODE_COL = "Ten-Year Age Groups Code"
REQUIRED_COLUMNS = {SUBCHAPTER_COL, YEAR_COL, MONTH_COL, "Deaths"}
AGE_REQUIRED_COLUMNS = {SUBCHAPTER_COL, YEAR_COL, AGE_COL, DEATHS_COL, POPULATION_COL}

STANDARD_WEIGHTS = {
    "< 1 year": 0.013818,
    "1-4 years": 0.055317,
    "5-14 years": 0.145565,
    "15-24 years": 0.138646,
    "25-34 years": 0.135573,
    "35-44 years": 0.162613,
    "45-54 years": 0.134834,
    "55-64 years": 0.087247,
    "65-74 years": 0.066037,
    "75-84 years": 0.044842,
    "85+ years": 0.015508,
}


def _find_header_row(csv_path: Path, required_columns: set[str] = REQUIRED_COLUMNS) -> int:
    with csv_path.open("r", encoding="utf-8-sig") as handle:
        for idx, line in enumerate(handle):
            if all(column in line for column in required_columns):
                return idx
    raise ValueError(f"Could not find data header in file: {csv_path}")


def _read_clean_csv(csv_path: Path, required_columns: set[str] = REQUIRED_COLUMNS) -> pd.DataFrame:
    header_row = _find_header_row(csv_path, required_columns)
    df = pd.read_csv(
        csv_path,
        skiprows=header_row,
        dtype={YEAR_COL: "string", YEAR_CODE_COL: "string"},
    )
    return _keep_data_rows(df)


def _keep_data_rows(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    year_text = (
        out.get(YEAR_COL)
        .astype("string")
        .str.strip()
        .str.replace(r"\s*\(provisional\)$", "", regex=True)
    )
    out["__year_num"] = pd.to_numeric(year_text, errors="coerce")
    if MONTH_COL in out.columns:
        out = out[out[MONTH_COL].notna()].copy()
    out = out[out["__year_num"].notna()].copy()
    out = out.drop(columns="__year_num")
    return out


def _normalize_year_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in [YEAR_COL, YEAR_CODE_COL]:
        if col in out.columns:
            out[col] = (
                out[col]
                .astype("string")
                .str.strip()
                .str.replace(r"\s*\(provisional\)$", "", regex=True)
            )
    if MONTH_COL in out.columns:
        out[MONTH_COL] = (
            out[MONTH_COL]
            .astype("string")
            .str.strip()
            .str.replace(r"\s*\(provisional\)$", "", regex=True)
        )
    return out


def _read_and_merge(file_names: list[str]) -> pd.DataFrame:
    frames = []
    for file_name in file_names:
        frames.append(_read_clean_csv(PRIMARY_DIR / file_name))
    merged = pd.concat(frames, ignore_index=True, sort=False)
    return _normalize_year_columns(merged)


def _read_and_merge_age_files(file_names: list[str]) -> pd.DataFrame:
    frames = []
    for file_name in file_names:
        frames.append(_read_clean_csv(PRIMARY_DIR / file_name, AGE_REQUIRED_COLUMNS))
    merged = pd.concat(frames, ignore_index=True, sort=False)
    return _normalize_year_columns(merged)


def _filter_combined_subchapter(df: pd.DataFrame) -> pd.DataFrame:
    if SUBCHAPTER_COL not in df.columns:
        return df.copy()
    return df[
        df[SUBCHAPTER_COL]
        .astype("string")
        .str.strip()
        .str.contains(r"other bacterial diseases|ther bacterial diseases", case=False, regex=True, na=False)
    ].copy()


def _filter_pneumonia_subchapter(df: pd.DataFrame) -> pd.DataFrame:
    if SUBCHAPTER_COL not in df.columns:
        return df.copy()
    return df[
        df[SUBCHAPTER_COL]
        .astype("string")
        .str.strip()
        .str.contains(r"influenza and pneumonia", case=False, regex=True, na=False)
    ].copy()


def _load_year_population(sex_file_name: str) -> pd.Series:
    sex_path_candidates = [
        OUTPUT_DIR / sex_file_name,
        PRIMARY_DIR / sex_file_name,
    ]
    sex_path = next((path for path in sex_path_candidates if path.exists()), None)
    if sex_path is None:
        searched = ", ".join(str(path) for path in sex_path_candidates)
        raise FileNotFoundError(f"Could not find sex population file '{sex_file_name}'. Searched: {searched}")
    sex_df = pd.read_csv(
        sex_path,
        dtype={YEAR_COL: "string", YEAR_CODE_COL: "string"},
    )
    sex_df = _normalize_year_columns(_keep_data_rows(sex_df))
    total_rows = sex_df[sex_df.get(NOTES_COL).astype("string").str.strip().eq("Total")].copy()
    total_rows[YEAR_COL] = total_rows[YEAR_COL].astype("string").str.strip()
    total_rows[POPULATION_COL] = pd.to_numeric(total_rows[POPULATION_COL], errors="coerce")
    total_rows = total_rows[total_rows[YEAR_COL].str.fullmatch(r"\d{4}", na=False)]
    total_rows = total_rows.dropna(subset=[POPULATION_COL])
    return total_rows.drop_duplicates(subset=[YEAR_COL]).set_index(YEAR_COL)[POPULATION_COL]


def _load_age_population(
    age_file_names: list[str],
    filter_combined: bool = False,
    filter_pneumonia: bool = False,
) -> pd.DataFrame:
    age_df = _read_and_merge_age_files(age_file_names)
    if filter_combined:
        age_df = _filter_combined_subchapter(age_df)
    if filter_pneumonia:
        age_df = _filter_pneumonia_subchapter(age_df)

    age_df = age_df[age_df[AGE_COL].isin(STANDARD_WEIGHTS)].copy()
    age_df[POPULATION_COL] = pd.to_numeric(age_df[POPULATION_COL], errors="coerce")
    age_df = age_df.dropna(subset=[POPULATION_COL])
    return age_df[[YEAR_COL, AGE_COL, POPULATION_COL]].drop_duplicates()


def _calculate_monthly_aamr(
    monthage_file_names: list[str],
    age_file_names: list[str],
    filter_combined: bool = False,
    filter_pneumonia: bool = False,
) -> pd.DataFrame:
    monthage_df = _read_and_merge_age_files(monthage_file_names)
    if filter_combined:
        monthage_df = _filter_combined_subchapter(monthage_df)
    if filter_pneumonia:
        monthage_df = _filter_pneumonia_subchapter(monthage_df)

    monthage_df = monthage_df[monthage_df[AGE_COL].isin(STANDARD_WEIGHTS)].copy()
    monthage_df["Standard Weight"] = monthage_df[AGE_COL].map(STANDARD_WEIGHTS)
    monthage_df[DEATHS_COL] = pd.to_numeric(monthage_df[DEATHS_COL], errors="coerce")

    age_population = _load_age_population(
        age_file_names,
        filter_combined=filter_combined,
        filter_pneumonia=filter_pneumonia,
    )
    month_keys = monthage_df[[YEAR_COL, MONTH_COL, "Month Code"]].drop_duplicates()
    age_keys = pd.DataFrame({AGE_COL: list(STANDARD_WEIGHTS)})
    complete_month_age = month_keys.merge(age_keys, how="cross")

    monthage_df = monthage_df.merge(
        complete_month_age,
        on=[YEAR_COL, MONTH_COL, "Month Code", AGE_COL],
        how="right",
    )
    monthage_df[DEATHS_COL] = monthage_df[DEATHS_COL].fillna(0)
    monthage_df["Standard Weight"] = monthage_df[AGE_COL].map(STANDARD_WEIGHTS)

    monthage_df = monthage_df.merge(
        age_population,
        on=[YEAR_COL, AGE_COL],
        how="left",
        suffixes=("", "_age"),
    )
    monthage_df[POPULATION_COL] = pd.to_numeric(monthage_df[f"{POPULATION_COL}_age"], errors="coerce")
    monthage_df["__age_rate"] = (monthage_df[DEATHS_COL] / monthage_df[POPULATION_COL]) * 100000
    monthage_df["__weighted_rate"] = monthage_df["__age_rate"] * monthage_df["Standard Weight"]
    monthage_df["__valid_age_group"] = (
        monthage_df[POPULATION_COL].notna()
        & monthage_df["Standard Weight"].notna()
    )

    grouped = (
        monthage_df.groupby([YEAR_COL, MONTH_COL, "Month Code"], dropna=False)
        .agg(
            **{
                AAMR_COL: ("__weighted_rate", "sum"),
                "__valid_age_groups": ("__valid_age_group", "sum"),
            }
        )
        .reset_index()
    )
    grouped.loc[grouped["__valid_age_groups"].lt(len(STANDARD_WEIGHTS)), AAMR_COL] = pd.NA
    grouped[AAMR_COL] = grouped[AAMR_COL].round(1)
    return grouped.drop(columns="__valid_age_groups")


def _fill_population_and_rates(
    df: pd.DataFrame,
    sex_file_name: str,
    monthage_file_names: list[str],
    age_file_names: list[str],
    filter_combined: bool = False,
) -> pd.DataFrame:
    out = _normalize_year_columns(_keep_data_rows(df))
    out[YEAR_COL] = out[YEAR_COL].astype("string").str.strip()
    out[DEATHS_COL] = pd.to_numeric(out[DEATHS_COL], errors="coerce")
    out[DEATHS_COL] = out[DEATHS_COL].fillna(0)

    year_population = _load_year_population(sex_file_name)
    out[POPULATION_COL] = out[YEAR_COL].map(year_population)

    monthly_rate = (out[DEATHS_COL] / out[POPULATION_COL]) * 100000
    if CRUDE_RATE_COL in out.columns:
        out[CRUDE_RATE_COL] = monthly_rate.round(1)

    monthly_aamr = _calculate_monthly_aamr(
        monthage_file_names=monthage_file_names,
        age_file_names=age_file_names,
        filter_combined=filter_combined,
    )
    out = out.merge(monthly_aamr, on=[YEAR_COL, MONTH_COL, "Month Code"], how="left", suffixes=("", "_calculated"))
    calc_col = f"{AAMR_COL}_calculated"
    if calc_col in out.columns:
        out[AAMR_COL] = out[calc_col]
        out = out.drop(columns=calc_col)
    return out


def _fill_population_and_rates_with_year_population_aamr(
    df: pd.DataFrame,
    sex_file_name: str,
    monthage_file_names: list[str],
    age_file_names: list[str],
    filter_pneumonia: bool = False,
) -> pd.DataFrame:
    out = _normalize_year_columns(_keep_data_rows(df))
    out[YEAR_COL] = out[YEAR_COL].astype("string").str.strip()
    out[DEATHS_COL] = pd.to_numeric(out[DEATHS_COL], errors="coerce")
    out[DEATHS_COL] = out[DEATHS_COL].fillna(0)

    year_population = _load_year_population(sex_file_name)
    out[POPULATION_COL] = out[YEAR_COL].map(year_population)

    monthly_rate = (out[DEATHS_COL] / out[POPULATION_COL]) * 100000
    if CRUDE_RATE_COL in out.columns:
        out[CRUDE_RATE_COL] = monthly_rate.round(1)

    monthly_aamr = _calculate_monthly_aamr(
        monthage_file_names=monthage_file_names,
        age_file_names=age_file_names,
        filter_pneumonia=filter_pneumonia,
    )
    out = out.merge(monthly_aamr, on=[YEAR_COL, MONTH_COL, "Month Code"], how="left", suffixes=("", "_calculated"))
    calc_col = f"{AAMR_COL}_calculated"
    if calc_col in out.columns:
        out[AAMR_COL] = out[calc_col]
        out = out.drop(columns=calc_col)
    return out


def build_outputs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sepsis_df = _fill_population_and_rates(
        _read_and_merge(SEPSIS_FILES),
        SEX_REFERENCE_FILES["sepsis"],
        SEPSIS_MONTHAGE_FILES,
        SEPSIS_AGE_FILES,
    )
    pneumonia_df = _fill_population_and_rates(
        _read_and_merge(PNEUMONIA_FILES),
        SEX_REFERENCE_FILES["pneumonia"],
        PNEUMONIA_MONTHAGE_FILES,
        PNEUMONIA_AGE_FILES,
    )
    combined_df = _fill_population_and_rates(
        _filter_combined_subchapter(_read_and_merge(COMBINED_FILES)),
        SEX_REFERENCE_FILES["combined"],
        COMBINED_MONTHAGE_FILES,
        COMBINED_AGE_FILES,
        filter_combined=True,
    )
    covid_pneumonia_df = _fill_population_and_rates_with_year_population_aamr(
        _filter_pneumonia_subchapter(_read_and_merge(UJ_COVID_PNEUMONIA_FILES)),
        SEX_REFERENCE_FILES["covid_pneumonia"],
        UJ_COVID_PNEUMONIA_MONTHAGE_FILES,
        PNEUMONIA_AGE_FILES,
        filter_pneumonia=True,
    )

    sepsis_df.to_csv(OUTPUT_DIR / "sepsis_month.csv", index=False)
    pneumonia_df.to_csv(OUTPUT_DIR / "pneumonia_month.csv", index=False)
    combined_df.to_csv(OUTPUT_DIR / "combined_month.csv", index=False)
    covid_pneumonia_df.to_csv(OUTPUT_DIR / "pneumonia_covid_month.csv", index=False)


if __name__ == "__main__":
    build_outputs()
