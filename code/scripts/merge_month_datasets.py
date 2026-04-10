from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PRIMARY_DIR = ROOT / "data" / "primary"
OUTPUT_DIR = ROOT / "data" / "cleaned"

SEPSIS_FILES = ["Amonth2010.csv", "Amonth2018.csv"]
PNEUMONIA_FILES = ["Jmonth2010.csv", "Jmonth2018.csv"]
COMBINED_FILES = ["AJmonth2010.csv", "AJmonth2018.csv"]

TARGET_SUBCHAPTER = "Other bacterial diseases"
SUBCHAPTER_COL = "MCD - ICD Sub-Chapter"
YEAR_COL = "Year"
YEAR_CODE_COL = "Year Code"
MONTH_COL = "Month"
REQUIRED_COLUMNS = {SUBCHAPTER_COL, YEAR_COL, MONTH_COL, "Deaths"}


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
        dtype={YEAR_COL: "string", YEAR_CODE_COL: "string"},
    )


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
    return out


def _read_and_merge(file_names: list[str]) -> pd.DataFrame:
    frames = []
    for file_name in file_names:
        frames.append(_read_clean_csv(PRIMARY_DIR / file_name))
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


def build_outputs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sepsis_df = _read_and_merge(SEPSIS_FILES)
    pneumonia_df = _read_and_merge(PNEUMONIA_FILES)
    combined_df = _filter_combined_subchapter(_read_and_merge(COMBINED_FILES))

    sepsis_df.to_csv(OUTPUT_DIR / "sepsis_month.csv", index=False)
    pneumonia_df.to_csv(OUTPUT_DIR / "pneumonia_month.csv", index=False)
    combined_df.to_csv(OUTPUT_DIR / "combined_month.csv", index=False)


if __name__ == "__main__":
    build_outputs()
