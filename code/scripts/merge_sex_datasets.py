from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PRIMARY_DIR = ROOT / "data" / "primary"
OUTPUT_DIR = ROOT / "data" / "cleaned"


ARDS_FILES = ["ARDSsex2010.csv", "ARDSsex2018.csv"]
PNEUMONIA_FILES = ["Jsex2010.csv", "Jsex2018.csv"]
COMBINED_FILES = ["AJsex2010.csv", "AJsex2018.csv"]
PNEUMONIA_COVID_FILES = ["UJsex2018.csv"]

ARDS_SUBCHAPTER = "Influenza and pneumonia"
TARGET_SUBCHAPTER = "Other bacterial diseases"
PNEUMONIA_SUBCHAPTER = "Influenza and pneumonia"
REQUIRED_COLUMNS = {"MCD - ICD Sub-Chapter", "Year", "Sex"}
MERGE_KEYS = [
    "Notes",
    "MCD - ICD Sub-Chapter",
    "MCD - ICD Sub-Chapter Code",
    "Year",
    "Year Code",
    "Sex",
    "Sex Code",
]
SUBTRACT_COLUMNS = [
    "Deaths",
    "Crude Rate",
    "Crude Rate Lower 95% Confidence Interval",
    "Crude Rate Upper 95% Confidence Interval",
    "Age Adjusted Rate",
    "Age Adjusted Rate Lower 95% Confidence Interval",
    "Age Adjusted Rate Upper 95% Confidence Interval",
]


def _find_header_row(csv_path: Path) -> int:
    with csv_path.open("r", encoding="utf-8-sig") as handle:
        for idx, line in enumerate(handle):
            normalized = line.strip().strip('"')
            if not normalized:
                continue
            if all(col in line for col in REQUIRED_COLUMNS):
                return idx

    raise ValueError(f"Could not find data header in file: {csv_path}")


def _read_clean_csv(csv_path: Path) -> pd.DataFrame:
    header_row = _find_header_row(csv_path)
    return pd.read_csv(
        csv_path,
        skiprows=header_row,
        sep=None,
        engine="python",
        dtype={"Year": "string", "Year Code": "string"},
    )


def _read_and_merge(file_names: list[str]) -> pd.DataFrame:
    frames = []
    for name in file_names:
        csv_path = PRIMARY_DIR / name
        frame = _read_clean_csv(csv_path)
        frames.append(frame)

    merged = pd.concat(frames, ignore_index=True, sort=False)
    return _normalize_year_columns(merged)


def _normalize_year_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Convert "2025 (provisional)" -> "2025" during merge.
    if "Year" in df.columns:
        df["Year"] = (
            df["Year"]
            .astype("string")
            .str.strip()
            .str.replace(r"\s*\(provisional\)$", "", regex=True)
        )

    if "Year Code" in df.columns:
        df["Year Code"] = (
            df["Year Code"]
            .astype("string")
            .str.strip()
            .str.replace(r"\s*\(provisional\)$", "", regex=True)
        )

    return df


def _filter_subchapter(df: pd.DataFrame, subchapter: str) -> pd.DataFrame:
    if "MCD - ICD Sub-Chapter" not in df.columns:
        return df.copy()
    return df[df["MCD - ICD Sub-Chapter"].eq(subchapter)].copy()


def _derive_non_covid_pneumonia(
    pneumonia_df: pd.DataFrame, pneumonia_covid_df: pd.DataFrame
) -> pd.DataFrame:
    # Restrict to rows with a valid year to avoid subtracting grand-total rows.
    left = pneumonia_df.copy()
    right = pneumonia_covid_df.copy()

    left["__year_num"] = pd.to_numeric(left["Year"], errors="coerce")
    right["__year_num"] = pd.to_numeric(right["Year"], errors="coerce")
    left = left[left["__year_num"].notna()].copy()
    right = right[right["__year_num"].notna()].copy()
    left = left.drop(columns="__year_num")
    right = right.drop(columns="__year_num")

    merged = left.merge(
        right,
        on=MERGE_KEYS,
        how="left",
        suffixes=("", "_pneu_covid"),
    )

    for col in SUBTRACT_COLUMNS:
        right_col = f"{col}_pneu_covid"
        if col in merged.columns and right_col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0) - pd.to_numeric(
                merged[right_col], errors="coerce"
            ).fillna(0)

    # Keep pneumonia population denominator as-is.
    drop_cols = [c for c in merged.columns if c.endswith("_pneu_covid")]
    non_covid = merged.drop(columns=drop_cols)
    return non_covid


def build_outputs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ards_df = _filter_subchapter(_read_and_merge(ARDS_FILES), ARDS_SUBCHAPTER)
    pneumonia_df = _read_and_merge(PNEUMONIA_FILES)
    combined_df = _read_and_merge(COMBINED_FILES)
    pneumonia_covid_df = _read_and_merge(PNEUMONIA_COVID_FILES)

    if "MCD - ICD Sub-Chapter" in combined_df.columns:
        combined_df = combined_df[
            combined_df["MCD - ICD Sub-Chapter"].eq(TARGET_SUBCHAPTER)
        ].copy()
    pneumonia_covid_df = _filter_subchapter(pneumonia_covid_df, PNEUMONIA_SUBCHAPTER)
    non_covid_pneumonia_df = _derive_non_covid_pneumonia(
        pneumonia_df=pneumonia_df,
        pneumonia_covid_df=pneumonia_covid_df,
    )

    ards_df.to_csv(OUTPUT_DIR / "ards_sex.csv", index=False)
    pneumonia_df.to_csv(OUTPUT_DIR / "pneumonia_sex.csv", index=False)
    combined_df.to_csv(OUTPUT_DIR / "combined_sex.csv", index=False)
    pneumonia_covid_df.to_csv(OUTPUT_DIR / "pneumonia_covid_sex.csv", index=False)
    non_covid_pneumonia_df.to_csv(OUTPUT_DIR / "non_covid_pneumonia_sex.csv", index=False)


if __name__ == "__main__":
    build_outputs()
