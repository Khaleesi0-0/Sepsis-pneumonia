from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PRIMARY_DIR = ROOT / "data" / "primary"
OUTPUT_DIR = ROOT / "data" / "cleaned"

ARDS_FILES = ["ARDSage2010.xls", "ARDSage2018.xls"]
PNEUMONIA_FILES = ["Jage2010.csv", "Jage2018.csv"]
COMBINED_FILES = ["AJage2010.csv", "AJage2018.csv"]

ARDS_SUBCHAPTER = "Influenza and pneumonia"
TARGET_SUBCHAPTER = "Other bacterial diseases"
AGE_COL = "Ten-Year Age Groups"
AGE_CODE_COL = "Ten-Year Age Groups Code"
YEAR_COL = "Year"
YEAR_CODE_COL = "Year Code"
SUBCHAPTER_COL = "MCD - ICD Sub-Chapter"
SUBCHAPTER_CODE_COL = "MCD - ICD Sub-Chapter Code"
NOTES_COL = "Notes"
DEATHS_COL = "Deaths"
POPULATION_COL = "Population"
CRUDE_RATE_COL = "Crude Rate"
COLLAPSED_AGE_COL = "Age Group"
AAMR_COL = "AAMR"

REQUIRED_COLUMNS = {YEAR_COL, AGE_COL, DEATHS_COL, POPULATION_COL}
GROUP_ORDER = ["<25", "25-44", "45-64", "65+"]
AGE_GROUP_MAP = {
    "< 1 year": "<25",
    "1-4 years": "<25",
    "5-14 years": "<25",
    "15-24 years": "<25",
    "25-34 years": "25-44",
    "35-44 years": "25-44",
    "45-54 years": "45-64",
    "55-64 years": "45-64",
    "65-74 years": "65+",
    "75-84 years": "65+",
    "85+ years": "65+",
}
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


def _find_header_row(csv_path: Path) -> int:
    with csv_path.open("r", encoding="utf-8-sig") as handle:
        for idx, line in enumerate(handle):
            if all(column in line for column in REQUIRED_COLUMNS):
                return idx

    raise ValueError(f"Could not find data header in file: {csv_path}")


def _read_clean_csv(csv_path: Path) -> pd.DataFrame:
    header_row = _find_header_row(csv_path)
    return pd.read_csv(
        csv_path,
        skiprows=header_row,
        sep=None,
        engine="python",
        dtype={YEAR_COL: "string", YEAR_CODE_COL: "string"},
    )


def _normalize_year_columns(df: pd.DataFrame) -> pd.DataFrame:
    for column in [YEAR_COL, YEAR_CODE_COL]:
        if column in df.columns:
            df[column] = (
                df[column]
                .astype("string")
                .str.strip()
                .str.replace(r"\s*\(provisional\)$", "", regex=True)
            )

    return df


def _read_and_merge(file_names: list[str]) -> pd.DataFrame:
    frames = []
    for file_name in file_names:
        frames.append(_read_clean_csv(PRIMARY_DIR / file_name))

    merged = pd.concat(frames, ignore_index=True, sort=False)
    return _normalize_year_columns(merged)


def _clean_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned[DEATHS_COL] = pd.to_numeric(cleaned[DEATHS_COL], errors="coerce")
    cleaned[POPULATION_COL] = pd.to_numeric(cleaned[POPULATION_COL], errors="coerce")
    if CRUDE_RATE_COL in cleaned.columns:
        cleaned[CRUDE_RATE_COL] = pd.to_numeric(cleaned[CRUDE_RATE_COL], errors="coerce")
    cleaned[YEAR_COL] = pd.to_numeric(cleaned[YEAR_COL], errors="coerce")
    cleaned[YEAR_CODE_COL] = pd.to_numeric(cleaned[YEAR_CODE_COL], errors="coerce").astype("Int64").astype("string")
    return cleaned


def _filter_combined_subchapter(df: pd.DataFrame) -> pd.DataFrame:
    if SUBCHAPTER_COL not in df.columns:
        return df.copy()
    return df[df[SUBCHAPTER_COL].eq(TARGET_SUBCHAPTER)].copy()


def _filter_ards_subchapter(df: pd.DataFrame) -> pd.DataFrame:
    if SUBCHAPTER_COL not in df.columns:
        return df.copy()
    return df[df[SUBCHAPTER_COL].eq(ARDS_SUBCHAPTER)].copy()


def _build_collapsed_age_groups(df: pd.DataFrame) -> pd.DataFrame:
    collapsed = df.copy()
    collapsed[COLLAPSED_AGE_COL] = collapsed[AGE_COL].map(AGE_GROUP_MAP)
    collapsed = collapsed[collapsed[COLLAPSED_AGE_COL].notna()].copy()
    collapsed = collapsed.dropna(subset=[YEAR_COL, DEATHS_COL, POPULATION_COL])
    collapsed["Standard Weight"] = pd.to_numeric(
        collapsed[AGE_COL].astype("string").map(STANDARD_WEIGHTS),
        errors="coerce",
    )
    collapsed = collapsed.dropna(subset=["Standard Weight"]).copy()
    collapsed[CRUDE_RATE_COL] = (collapsed[DEATHS_COL] / collapsed[POPULATION_COL]) * 100000
    collapsed["__weighted_rate"] = collapsed[CRUDE_RATE_COL] * collapsed["Standard Weight"]

    grouped = (
        collapsed.groupby(
            [NOTES_COL, SUBCHAPTER_COL, SUBCHAPTER_CODE_COL, YEAR_COL, YEAR_CODE_COL, COLLAPSED_AGE_COL],
            dropna=False,
            as_index=False,
        )[[DEATHS_COL, POPULATION_COL, "Standard Weight", "__weighted_rate"]]
        .sum()
    )
    grouped[CRUDE_RATE_COL] = (grouped[DEATHS_COL] / grouped[POPULATION_COL]) * 100000
    grouped[AAMR_COL] = grouped["__weighted_rate"] / grouped["Standard Weight"]
    grouped = grouped.drop(columns="__weighted_rate")
    grouped[COLLAPSED_AGE_COL] = pd.Categorical(grouped[COLLAPSED_AGE_COL], categories=GROUP_ORDER, ordered=True)
    grouped = grouped.sort_values([YEAR_COL, COLLAPSED_AGE_COL]).reset_index(drop=True)
    return grouped


def _prepare_age_outputs(
    file_names: list[str],
    filter_combined: bool = False,
    filter_ards: bool = False,
) -> pd.DataFrame:
    merged = _read_and_merge(file_names)
    merged = _clean_numeric_columns(merged)
    if filter_combined:
        merged = _filter_combined_subchapter(merged)
    if filter_ards:
        merged = _filter_ards_subchapter(merged)

    collapsed = _build_collapsed_age_groups(merged)
    return collapsed


def build_outputs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ards_grouped = _prepare_age_outputs(ARDS_FILES, filter_ards=True)
    pneumonia_grouped = _prepare_age_outputs(PNEUMONIA_FILES)
    combined_grouped = _prepare_age_outputs(COMBINED_FILES, filter_combined=True)

    ards_grouped.to_csv(OUTPUT_DIR / "ards_age.csv", index=False)
    pneumonia_grouped.to_csv(OUTPUT_DIR / "pneumonia_age.csv", index=False)
    combined_grouped.to_csv(OUTPUT_DIR / "combined_age.csv", index=False)


if __name__ == "__main__":
    build_outputs()
