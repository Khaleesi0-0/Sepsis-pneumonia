from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PRIMARY_DIR = ROOT / "data" / "primary"
OUTPUT_DIR = ROOT / "data" / "cleaned"

SEPSIS_FILES = ["Arace2010.csv", "Arace2018.csv"]
PNEUMONIA_FILES = ["Jrace2010.csv", "Jrace2018.csv"]
COMBINED_FILES = ["AJrace2010.csv", "AJrace2018.csv"]

TARGET_SUBCHAPTER = "Other bacterial diseases"
SUBCHAPTER_COL = "MCD - ICD Sub-Chapter"
YEAR_COL = "Year"
YEAR_CODE_COL = "Year Code"
RACE_STD_COL = "Race 6"
RACE_STD_CODE_COL = "Race 6 Code"

RACE_COL_CANDIDATES = ["Single Race 6", "Race"]
RACE_CODE_COL_CANDIDATES = ["Single Race 6 Code", "Race Code"]
REQUIRED_COLUMNS = {SUBCHAPTER_COL, YEAR_COL, "Deaths"}
ALLOWED_RACE6 = {
    "American Indian or Alaska Native",
    "Asian",
    "Black or African American",
    "Native Hawaiian or Other Pacific Islander",
    "White",
    "More than one race",
}


def _find_header_row(csv_path: Path) -> int:
    with csv_path.open("r", encoding="utf-8-sig") as handle:
        for idx, line in enumerate(handle):
            if all(column in line for column in REQUIRED_COLUMNS) and any(
                race_col in line for race_col in RACE_COL_CANDIDATES
            ):
                return idx

    raise ValueError(f"Could not find data header in file: {csv_path}")


def _read_clean_csv(csv_path: Path) -> pd.DataFrame:
    header_row = _find_header_row(csv_path)
    return pd.read_csv(
        csv_path,
        skiprows=header_row,
        dtype={YEAR_COL: "string", YEAR_CODE_COL: "string"},
    )


def _normalize_year_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in [YEAR_COL, YEAR_CODE_COL]:
        if col in df.columns:
            df[col] = (
                df[col]
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


def _coalesce_race_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    race_series = pd.Series(pd.NA, index=out.index, dtype="string")
    for col in RACE_COL_CANDIDATES:
        if col in out.columns:
            race_series = race_series.fillna(out[col].astype("string"))

    race_code_series = pd.Series(pd.NA, index=out.index, dtype="string")
    for col in RACE_CODE_COL_CANDIDATES:
        if col in out.columns:
            race_code_series = race_code_series.fillna(out[col].astype("string"))

    out[RACE_STD_COL] = race_series.str.strip()
    out[RACE_STD_CODE_COL] = race_code_series.str.strip()

    # Harmonize legacy 2010 category name to Single Race 6 convention.
    out[RACE_STD_COL] = out[RACE_STD_COL].replace(
        {"Asian or Pacific Islander": "Asian"}
    )

    return out


def _filter_race6_rows(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out[out[RACE_STD_COL].isin(ALLOWED_RACE6)].copy()
    # Explicitly drop Not Available rows from 2018+ extract.
    out = out[~out[RACE_STD_COL].eq("Not Available")].copy()
    return out


def _filter_combined_subchapter(df: pd.DataFrame) -> pd.DataFrame:
    if SUBCHAPTER_COL not in df.columns:
        return df.copy()
    return df[
        df[SUBCHAPTER_COL]
        .astype("string")
        .str.strip()
        .str.contains(r"other bacterial diseases|ther bacterial diseases", case=False, regex=True, na=False)
    ].copy()


def _drop_source_race_columns(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [col for col in (RACE_COL_CANDIDATES + RACE_CODE_COL_CANDIDATES) if col in df.columns]
    return df.drop(columns=drop_cols)


def _prepare_race_output(file_names: list[str], filter_combined: bool = False) -> pd.DataFrame:
    merged = _read_and_merge(file_names)
    merged = _coalesce_race_columns(merged)
    if filter_combined:
        merged = _filter_combined_subchapter(merged)
    merged = _filter_race6_rows(merged)
    merged = _drop_source_race_columns(merged)
    return merged


def build_outputs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sepsis_df = _prepare_race_output(SEPSIS_FILES)
    pneumonia_df = _prepare_race_output(PNEUMONIA_FILES)
    combined_df = _prepare_race_output(COMBINED_FILES, filter_combined=True)

    sepsis_df.to_csv(OUTPUT_DIR / "sepsis_race.csv", index=False)
    pneumonia_df.to_csv(OUTPUT_DIR / "pneumonia_race.csv", index=False)
    combined_df.to_csv(OUTPUT_DIR / "combined_race.csv", index=False)


if __name__ == "__main__":
    build_outputs()
