from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
CLEANED_DIR = ROOT / "data" / "cleaned"
TABLE_DIR = ROOT / "results" / "tables"

STATE_COL = "State"
STATE_COL_CANDIDATES = ["State", "Residence State"]
STATE_CODE_COL = "State Code"
YEAR_COL = "Year"
DEATHS_COL = "Deaths"
POP_COL = "Population"
AAMR_COL = "Age Adjusted Rate"

PERIODS = [
    ("Pre-pandemic (2010-2019)", range(2010, 2020)),
    ("Pandemic (2020-2023)", range(2020, 2024)),
    ("Post-pandemic (2024-2025)", range(2024, 2026)),
]

DATASETS = {
    "Sepsis": CLEANED_DIR / "sepsis_state.csv",
    "Pneumonia": CLEANED_DIR / "pneumonia_state.csv",
    "Sepsis + pneumonia": CLEANED_DIR / "combined_state.csv",
}

KEEP_STATES = {
    "Alabama",
    "Alaska",
    "Arizona",
    "Arkansas",
    "California",
    "Colorado",
    "Connecticut",
    "Delaware",
    "District of Columbia",
    "Florida",
    "Georgia",
    "Hawaii",
    "Idaho",
    "Illinois",
    "Indiana",
    "Iowa",
    "Kansas",
    "Kentucky",
    "Louisiana",
    "Maine",
    "Maryland",
    "Massachusetts",
    "Michigan",
    "Minnesota",
    "Mississippi",
    "Missouri",
    "Montana",
    "Nebraska",
    "Nevada",
    "New Hampshire",
    "New Jersey",
    "New Mexico",
    "New York",
    "North Carolina",
    "North Dakota",
    "Ohio",
    "Oklahoma",
    "Oregon",
    "Pennsylvania",
    "Rhode Island",
    "South Carolina",
    "South Dakota",
    "Tennessee",
    "Texas",
    "Utah",
    "Vermont",
    "Virginia",
    "Washington",
    "West Virginia",
    "Wisconsin",
    "Wyoming",
}


def _resolve_state_column(df: pd.DataFrame) -> str:
    for col in STATE_COL_CANDIDATES:
        if col in df.columns:
            return col
    raise KeyError(
        "Could not find a state column. Expected one of "
        f"{STATE_COL_CANDIDATES}, but found: {list(df.columns)}"
    )


def _prepare_state_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    state_source_col = _resolve_state_column(df)
    df[STATE_COL] = df[state_source_col].astype("string").str.strip()
    df[YEAR_COL] = (
        df[YEAR_COL]
        .astype("string")
        .str.strip()
        .str.replace(r"\s*\(provisional\)$", "", regex=True)
    )
    df["__year"] = pd.to_numeric(df[YEAR_COL], errors="coerce")

    for col in [DEATHS_COL, POP_COL, AAMR_COL]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[df[STATE_COL].isin(KEEP_STATES)].copy()
    df = df[df["__year"].notna()].copy()
    return df


def _period_aggregate(df: pd.DataFrame, years: range) -> pd.DataFrame:
    period_df = df[df["__year"].isin(list(years))].copy()
    rows = []

    group_cols = [STATE_COL]
    if STATE_CODE_COL in period_df.columns:
        group_cols.append(STATE_CODE_COL)

    for group_key, state_df in period_df.groupby(group_cols, dropna=False):
        if isinstance(group_key, tuple):
            state_name, state_code = group_key
        else:
            state_name, state_code = group_key, pd.NA

        total_deaths = state_df[DEATHS_COL].sum()
        total_pop = state_df[POP_COL].sum()
        aamr = (
            (state_df[AAMR_COL] * state_df[POP_COL]).sum() / total_pop
            if total_pop and pd.notna(total_pop)
            else pd.NA
        )
        crude_rate = (total_deaths / total_pop * 100000) if total_pop and pd.notna(total_pop) else pd.NA

        rows.append(
            {
                STATE_COL: state_name,
                STATE_CODE_COL: state_code,
                "Deaths": total_deaths,
                "Population": total_pop,
                "Crude rate (/100,000)": crude_rate,
                "AAMR (/100,000)": aamr,
            }
        )

    return pd.DataFrame(rows)


def build_state_map_data_table() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for outcome, path in DATASETS.items():
        raw = _prepare_state_data(path)
        for period_label, years in PERIODS:
            period_table = _period_aggregate(raw, years)
            period_table.insert(0, "Outcome", outcome)
            period_table.insert(1, "Period", period_label)
            rows.append(period_table)

    table = pd.concat(rows, ignore_index=True)
    table = table[
        [
            "Outcome",
            "Period",
            STATE_COL,
            STATE_CODE_COL,
            "Deaths",
            "Population",
            "Crude rate (/100,000)",
            "AAMR (/100,000)",
        ]
    ]
    table = table.sort_values(["Outcome", "Period", STATE_COL]).reset_index(drop=True)
    table.to_csv(TABLE_DIR / "state_map_period_data.csv", index=False)


if __name__ == "__main__":
    build_state_map_data_table()
