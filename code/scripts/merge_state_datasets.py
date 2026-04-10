from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PRIMARY_DIR = ROOT / "data" / "primary"
OUTPUT_DIR = ROOT / "data" / "cleaned"

SEPSIS_FILES = ["Astate2010.csv", "Astate2018.csv"]
PNEUMONIA_FILES = ["Jstate2010.csv", "Jstate2018.csv"]
COMBINED_FILES = ["AJstate2010.csv", "AJstate2018.csv"]

TARGET_SUBCHAPTER = "Other bacterial diseases"
SUBCHAPTER_COL = "MCD - ICD Sub-Chapter"
YEAR_COL = "Year"
YEAR_CODE_COL = "Year Code"

STATE_COL_CANDIDATES = ["Residence State", "State"]
STATE_CODE_COL_CANDIDATES = ["Residence State Code", "State Code"]
STATE_STD_COL = "State"
STATE_STD_CODE_COL = "State Code"
REQUIRED_COLUMNS = {SUBCHAPTER_COL, YEAR_COL, "Deaths"}


def _find_header_row(csv_path: Path) -> int:
    with csv_path.open("r", encoding="utf-8-sig") as handle:
        for idx, line in enumerate(handle):
            if all(column in line for column in REQUIRED_COLUMNS) and any(
                state_col in line for state_col in STATE_COL_CANDIDATES
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


def _coalesce_state_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    state_series = None
    for col in STATE_COL_CANDIDATES:
        if col in out.columns:
            candidate = out[col].astype("string")
            state_series = candidate if state_series is None else state_series.combine_first(candidate)

    state_code_series = None
    for col in STATE_CODE_COL_CANDIDATES:
        if col in out.columns:
            candidate = out[col].astype("string")
            state_code_series = candidate if state_code_series is None else state_code_series.combine_first(candidate)

    out[STATE_STD_COL] = state_series.str.strip() if state_series is not None else pd.Series(pd.NA, index=out.index, dtype="string")
    out[STATE_STD_CODE_COL] = state_code_series.str.strip() if state_code_series is not None else pd.Series(pd.NA, index=out.index, dtype="string")
    drop_cols = [
        col
        for col in (STATE_COL_CANDIDATES + STATE_CODE_COL_CANDIDATES)
        if col in out.columns and col not in {STATE_STD_COL, STATE_STD_CODE_COL}
    ]
    out = out.drop(columns=drop_cols)
    return out


def _read_and_merge(file_names: list[str]) -> pd.DataFrame:
    frames = []
    for file_name in file_names:
        frames.append(_read_clean_csv(PRIMARY_DIR / file_name))
    merged = pd.concat(frames, ignore_index=True, sort=False)
    merged = _normalize_year_columns(merged)
    merged = _coalesce_state_columns(merged)
    return merged


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

    sepsis_df.to_csv(OUTPUT_DIR / "sepsis_state.csv", index=False)
    pneumonia_df.to_csv(OUTPUT_DIR / "pneumonia_state.csv", index=False)
    combined_df.to_csv(OUTPUT_DIR / "combined_state.csv", index=False)


if __name__ == "__main__":
    build_outputs()
