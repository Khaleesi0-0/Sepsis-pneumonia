from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
CLEANED_DIR = ROOT / "data" / "cleaned"
TABLE_DIR = ROOT / "results" / "tables"

YEAR_COL = "Year"
NOTES_COL = "Notes"
RATE_COL = "Age Adjusted Rate"

DATASETS = {
    "Pneumonia": CLEANED_DIR / "pneumonia_sex.csv",
    "Pneumonia/ARDS": CLEANED_DIR / "ards_sex.csv",
    "Pneumonia/Sepsis": CLEANED_DIR / "combined_sex.csv",
    "Non-COVID Pneumonia": CLEANED_DIR / "non_covid_pneumonia_sex.csv",
}

MAX_JOINPOINTS = 3
MIN_SEGMENT_POINTS = 3
COMMON_SEGMENTS = [
    ("2010-2018", 2010, 2018),
    ("2019-2021", 2019, 2021),
    ("2022-2025", 2022, 2025),
]

DISEASE_SEGMENTS = {
    "Non-COVID Pneumonia": [
        ("2019-2021", 2019, 2021),
        ("2022-2025", 2022, 2025),
    ],
}


def _prepare_total_series(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, dtype={YEAR_COL: "string"})
    df[YEAR_COL] = (
        df[YEAR_COL]
        .astype("string")
        .str.strip()
        .str.replace(r"\s*\(provisional\)$", "", regex=True)
    )
    df[YEAR_COL] = pd.to_numeric(df[YEAR_COL], errors="coerce")
    df[RATE_COL] = pd.to_numeric(df[RATE_COL], errors="coerce")

    total = df[
        df[NOTES_COL].astype("string").str.strip().eq("Total")
        & df[YEAR_COL].notna()
        & df[RATE_COL].notna()
    ][[YEAR_COL, RATE_COL]].copy()
    total = total[(total[YEAR_COL] >= 1900) & (total[YEAR_COL] <= 2100)]
    total[YEAR_COL] = total[YEAR_COL].astype(int)
    return total.sort_values(YEAR_COL).reset_index(drop=True)


def _fit_log_linear_segment(segment_df: pd.DataFrame) -> dict[str, float]:
    if len(segment_df) < 2:
        return {"slope": np.nan, "rss": np.nan, "slope_se": np.nan, "dof": np.nan}

    x = segment_df[YEAR_COL].to_numpy(dtype=float)
    y = np.log(segment_df[RATE_COL].to_numpy(dtype=float))
    x_design = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(x_design, y, rcond=None)

    fitted = x_design @ beta
    residual = y - fitted
    rss = float(residual @ residual)
    dof = len(x) - 2
    slope_se = np.nan
    if dof > 0:
        sigma2 = rss / dof
        xtx_inv = np.linalg.inv(x_design.T @ x_design)
        slope_se = float(np.sqrt(sigma2 * xtx_inv[1, 1]))

    return {
        "slope": float(beta[1]),
        "rss": rss,
        "slope_se": slope_se,
        "dof": float(dof),
    }


def _compute_segment_apc(segment_df: pd.DataFrame) -> tuple[float | None, float | None, float | None]:
    if len(segment_df) < 2:
        return pd.NA, pd.NA, pd.NA

    fit = _fit_log_linear_segment(segment_df)
    slope = fit["slope"]
    apc = (np.exp(slope) - 1.0) * 100.0

    if pd.isna(fit["slope_se"]) or fit["dof"] <= 0:
        return apc, pd.NA, pd.NA

    z = 1.96
    slope_lo = slope - z * fit["slope_se"]
    slope_hi = slope + z * fit["slope_se"]
    apc_lo = (np.exp(slope_lo) - 1.0) * 100.0
    apc_hi = (np.exp(slope_hi) - 1.0) * 100.0
    return apc, apc_lo, apc_hi


def _valid_split_set(split_idx: tuple[int, ...], n_points: int) -> bool:
    boundaries = (0,) + split_idx + (n_points,)
    lengths = [boundaries[i + 1] - boundaries[i] for i in range(len(boundaries) - 1)]
    return all(length >= MIN_SEGMENT_POINTS for length in lengths)


def _segment_slices(split_idx: tuple[int, ...], n_points: int) -> list[tuple[int, int]]:
    boundaries = (0,) + split_idx + (n_points,)
    return [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]


def _evaluate_segmentation(total: pd.DataFrame, split_idx: tuple[int, ...]) -> dict[str, object]:
    n = len(total)
    slices = _segment_slices(split_idx, n)
    rss_total = 0.0
    for start, end in slices:
        seg = total.iloc[start:end]
        fit = _fit_log_linear_segment(seg)
        rss_total += fit["rss"]

    if rss_total <= 0:
        rss_total = 1e-12

    # Piecewise log-linear model with intercept+slope per segment.
    n_params = 2 * len(slices)
    bic = n * np.log(rss_total / n) + n_params * np.log(n)
    return {"split_idx": split_idx, "bic": float(bic), "rss": float(rss_total), "n_params": n_params}


def _select_joinpoints(total: pd.DataFrame) -> dict[str, object]:
    n = len(total)
    if n < (2 * MIN_SEGMENT_POINTS):
        return {"split_idx": tuple(), "bic": np.nan, "rss": np.nan, "n_params": 2}

    candidate_positions = list(range(1, n))
    max_k = min(MAX_JOINPOINTS, (n // MIN_SEGMENT_POINTS) - 1)

    best: dict[str, object] | None = None
    for k in range(0, max_k + 1):
        for split_idx in combinations(candidate_positions, k):
            if not _valid_split_set(split_idx, n):
                continue
            fit = _evaluate_segmentation(total, split_idx)
            if best is None or fit["bic"] < best["bic"]:
                best = fit

    if best is None:
        return {"split_idx": tuple(), "bic": np.nan, "rss": np.nan, "n_params": 2}
    return best


def build_joinpoint_apc_table() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    segment_rows: list[dict[str, object]] = []
    model_rows: list[dict[str, object]] = []

    for disease, csv_path in DATASETS.items():
        total = _prepare_total_series(csv_path)
        segments = DISEASE_SEGMENTS.get(disease, COMMON_SEGMENTS)
        joinpoint_years = "N/A"
        selected_joinpoints = max(len(segments) - 1, 0)
        if len(segments) > 1:
            joinpoint_years = "; ".join(str(seg[2] + 1) for seg in segments[:-1])
        # Use the same user-specified segment boundaries for all diseases.
        model_rows.append(
            {
                "Disease": disease,
                "N Points": int(len(total)),
                "Selected Joinpoints": selected_joinpoints,
                "Joinpoint Years": joinpoint_years,
                "BIC": pd.NA,
                "RSS": pd.NA,
            }
        )

        for seg_num, (period_label, start_year, end_year) in enumerate(segments, start=1):
            segment = total[total[YEAR_COL].between(start_year, end_year)].copy()
            apc, apc_lo, apc_hi = _compute_segment_apc(segment)

            segment_rows.append(
                {
                    "Disease": disease,
                    "Segment": seg_num,
                    "Period": period_label,
                    "Start Year": start_year,
                    "End Year": end_year,
                    "Years in Segment": int(len(segment)),
                    "APC (% per year)": apc,
                    "APC Lower 95% CI": apc_lo,
                    "APC Upper 95% CI": apc_hi,
                }
            )

    out = pd.DataFrame(segment_rows)
    for col in ["APC (% per year)", "APC Lower 95% CI", "APC Upper 95% CI"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").round(2)

    models = pd.DataFrame(model_rows)
    for col in ["BIC", "RSS"]:
        models[col] = pd.to_numeric(models[col], errors="coerce").round(3)

    out.to_csv(TABLE_DIR / "joinpoint_apc_segments.csv", index=False)
    models.to_csv(TABLE_DIR / "joinpoint_selected_models.csv", index=False)


if __name__ == "__main__":
    build_joinpoint_apc_table()

