from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from model_monthly_aamr_regression import (
    MODEL_SPECS,
    _design_matrix,
    _get_model_spec,
    _model_training_subset,
)


ROOT = Path(__file__).resolve().parents[2]
CLEANED_DIR = ROOT / "data" / "cleaned"
TABLE_DIR = ROOT / "results" / "tables"

YEAR_COL = "Year"
MONTH_COL = "Month"
MONTH_CODE_COL = "Month Code"
DEATHS_COL = "Deaths"
PREDICTED_DEATHS_COL = "Predicted Deaths"
EXCESS_DEATHS_COL = "Actual - Predicted Deaths"

TRAIN_START_YEAR = 2010
MODEL_TRAIN_END_YEAR = 2019
VALIDATION_YEARS = [2018, 2019]
PREDICT_START_YEAR = 2020
PREDICT_END_YEAR = 2025
COVID_START_YEAR = 2020
COVID_END_YEAR = 2023

SELECTED_MODELS_PATH = TABLE_DIR / "monthly_aamr_validation_metrics.csv"

DATASETS = {
    "Pneumonia/ARDS": CLEANED_DIR / "ards_month.csv",
    "Pneumonia": CLEANED_DIR / "pneumonia_month.csv",
    "Pneumonia/Sepsis": CLEANED_DIR / "combined_month.csv",
}

DISEASE_NAME_MAP = {
    "ARDS (ARDS + Pneumonia)": "Pneumonia/ARDS",
    "Sepsis+ Pneumonia": "Pneumonia/Sepsis",
    "ARDS/Pneumonia": "Pneumonia/ARDS",
    "Sepsis/Pneumonia": "Pneumonia/Sepsis",
}


def _month_number_from_code(month_code: pd.Series) -> pd.Series:
    return pd.to_numeric(
        month_code.astype("string").str.extract(r"/(\d{2})$", expand=False),
        errors="coerce",
    )


def _transform_target(values: np.ndarray, target: str) -> np.ndarray:
    if target == "identity":
        return values
    if target == "log":
        return np.log(values)
    raise ValueError(f"Unknown target transform: {target}")


def _inverse_transform(values: np.ndarray, target: str) -> np.ndarray:
    if target == "identity":
        return values
    if target == "log":
        return np.exp(values)
    raise ValueError(f"Unknown target transform: {target}")


def _read_monthly_deaths(csv_path: Path, disease: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, dtype={YEAR_COL: "string", MONTH_CODE_COL: "string"})
    df[YEAR_COL] = (
        df[YEAR_COL]
        .astype("string")
        .str.strip()
        .str.replace(r"\s*\(provisional\)$", "", regex=True)
    )
    df[YEAR_COL] = pd.to_numeric(df[YEAR_COL], errors="coerce")
    df["Month Number"] = _month_number_from_code(df[MONTH_CODE_COL])
    df[DEATHS_COL] = pd.to_numeric(df[DEATHS_COL], errors="coerce")
    df = df.dropna(subset=[YEAR_COL, "Month Number", DEATHS_COL]).copy()
    df = df[df[DEATHS_COL] > 0].copy()
    df[YEAR_COL] = df[YEAR_COL].astype(int)
    df["Month Number"] = df["Month Number"].astype(int)
    df["Time Index"] = ((df[YEAR_COL] - TRAIN_START_YEAR) * 12) + (df["Month Number"] - 1)
    df["Disease"] = disease
    return df.sort_values([YEAR_COL, "Month Number"]).reset_index(drop=True)


def _anchor_values(train_df: pd.DataFrame, target_df: pd.DataFrame, model_spec: dict[str, object]) -> np.ndarray:
    anchor = model_spec.get("Anchor")
    if anchor != "recent_3_year_month_mean":
        raise ValueError(f"Unknown anchor specification: {anchor}")

    train_subset = _model_training_subset(train_df, model_spec)
    recent_years = sorted(train_subset[YEAR_COL].dropna().unique())[-3:]
    by_month = (
        train_subset[train_subset[YEAR_COL].isin(recent_years)]
        .groupby("Month Number")[DEATHS_COL]
        .mean()
    )
    return target_df["Month Number"].map(by_month).to_numpy(dtype=float)


def _fit_regression(train_df: pd.DataFrame, model_spec: dict[str, object]) -> np.ndarray:
    train_subset = _model_training_subset(train_df, model_spec)
    x = _design_matrix(train_subset, model_spec["Feature Set"])
    y = _transform_target(train_subset[DEATHS_COL].to_numpy(dtype=float), model_spec["Target"])

    half_life = model_spec.get("Half Life Months")
    if half_life is not None:
        max_time = train_subset["Time Index"].max()
        weights = 0.5 ** ((max_time - train_subset["Time Index"].to_numpy(dtype=float)) / float(half_life))
        sqrt_weights = np.sqrt(weights)
        x = x * sqrt_weights[:, None]
        y = y * sqrt_weights

    coefficients, *_ = np.linalg.lstsq(x, y, rcond=None)
    return coefficients


def _predict(
    model_spec: dict[str, object],
    coefficients: np.ndarray,
    train_df: pd.DataFrame,
    target_df: pd.DataFrame,
) -> np.ndarray:
    predicted = _design_matrix(target_df, model_spec["Feature Set"]) @ coefficients
    predicted = _inverse_transform(predicted, model_spec["Target"])

    blend_weight = model_spec.get("Blend Weight")
    if blend_weight is None:
        return predicted

    anchor = _anchor_values(train_df=train_df, target_df=target_df, model_spec=model_spec)
    return ((1 - float(blend_weight)) * predicted) + (float(blend_weight) * anchor)


def _read_selected_models() -> pd.DataFrame:
    selected = pd.read_csv(SELECTED_MODELS_PATH)
    selected["Disease"] = selected["Disease"].astype("string").str.strip().replace(DISEASE_NAME_MAP)
    selected["Selected Model"] = selected["Selected Model"].astype("string").str.strip()
    selected["Selected Model Label"] = selected["Selected Model Label"].astype("string").str.strip()
    selected = selected[["Disease", "Selected Model", "Selected Model Label"]].drop_duplicates()
    return selected


def _rolling_validation_prediction(df: pd.DataFrame, model_spec: dict[str, object]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for validation_year in VALIDATION_YEARS:
        train_df = df[df[YEAR_COL].between(TRAIN_START_YEAR, validation_year - 1)].copy()
        target_df = df[df[YEAR_COL].eq(validation_year)].copy()
        coefficients = _fit_regression(train_df, model_spec)
        target_df["Validation Year"] = validation_year
        target_df["Training Years"] = f"{TRAIN_START_YEAR}-{validation_year - 1}"
        target_df[PREDICTED_DEATHS_COL] = _predict(
            model_spec=model_spec,
            coefficients=coefficients,
            train_df=train_df,
            target_df=target_df,
        )
        target_df[EXCESS_DEATHS_COL] = target_df[DEATHS_COL] - target_df[PREDICTED_DEATHS_COL]
        frames.append(target_df)
    return pd.concat(frames, ignore_index=True)


def _predict_2020_2025(df: pd.DataFrame, model_spec: dict[str, object]) -> pd.DataFrame:
    train_df = df[df[YEAR_COL].between(TRAIN_START_YEAR, MODEL_TRAIN_END_YEAR)].copy()
    target_df = df[df[YEAR_COL].between(PREDICT_START_YEAR, PREDICT_END_YEAR)].copy()
    coefficients = _fit_regression(train_df, model_spec)
    target_df["Validation Year"] = pd.NA
    target_df["Training Years"] = f"{TRAIN_START_YEAR}-{MODEL_TRAIN_END_YEAR}"
    target_df[PREDICTED_DEATHS_COL] = _predict(
        model_spec=model_spec,
        coefficients=coefficients,
        train_df=train_df,
        target_df=target_df,
    )
    target_df[EXCESS_DEATHS_COL] = target_df[DEATHS_COL] - target_df[PREDICTED_DEATHS_COL]
    return target_df


def _covid_summary(prediction_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    covid_df = prediction_df[prediction_df[YEAR_COL].between(COVID_START_YEAR, COVID_END_YEAR)].copy()
    yearly = (
        covid_df.groupby(["Disease", YEAR_COL], as_index=False)
        .agg(
            **{
                "Mean Actual Deaths": (DEATHS_COL, "mean"),
                "Mean Predicted Deaths": (PREDICTED_DEATHS_COL, "mean"),
                "Mean Excess Deaths": (EXCESS_DEATHS_COL, "mean"),
                "Total Actual Deaths": (DEATHS_COL, "sum"),
                "Total Predicted Deaths": (PREDICTED_DEATHS_COL, "sum"),
                "Total Excess Deaths": (EXCESS_DEATHS_COL, "sum"),
            }
        )
        .sort_values(["Disease", YEAR_COL])
    )

    overall = (
        covid_df.groupby("Disease", as_index=False)
        .agg(
            **{
                "Year": (YEAR_COL, lambda _: f"{COVID_START_YEAR}-{COVID_END_YEAR}"),
                "Mean Actual Deaths": (DEATHS_COL, "mean"),
                "Mean Predicted Deaths": (PREDICTED_DEATHS_COL, "mean"),
                "Mean Excess Deaths": (EXCESS_DEATHS_COL, "mean"),
                "Total Actual Deaths": (DEATHS_COL, "sum"),
                "Total Predicted Deaths": (PREDICTED_DEATHS_COL, "sum"),
                "Total Excess Deaths": (EXCESS_DEATHS_COL, "sum"),
            }
        )
        .sort_values("Disease")
    )
    return yearly, overall


def _finalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Disease"] = out["Disease"].astype("string").replace(DISEASE_NAME_MAP)
    for col in [DEATHS_COL, PREDICTED_DEATHS_COL, EXCESS_DEATHS_COL]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").round(1)
    return out


def build_monthly_deaths_excess_outputs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    selected_models = _read_selected_models()

    validation_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []

    for disease, path in DATASETS.items():
        disease_df = _read_monthly_deaths(path, disease)
        model_row = selected_models[selected_models["Disease"].eq(disease)]
        if model_row.empty:
            raise ValueError(f"No selected model found for disease: {disease}")

        model_name = model_row["Selected Model"].iloc[0]
        model_label = model_row["Selected Model Label"].iloc[0]
        model_spec = _get_model_spec(model_name)

        validation = _rolling_validation_prediction(disease_df, model_spec)
        validation["Model"] = model_name
        validation["Model Label"] = model_label
        validation_frames.append(validation)

        prediction = _predict_2020_2025(disease_df, model_spec)
        prediction["Model"] = model_name
        prediction["Model Label"] = model_label
        prediction_frames.append(prediction)

    validation_df = _finalize_columns(pd.concat(validation_frames, ignore_index=True))
    prediction_df = _finalize_columns(pd.concat(prediction_frames, ignore_index=True))

    validation_keep = [
        "Disease",
        "Model",
        "Model Label",
        "Validation Year",
        "Training Years",
        YEAR_COL,
        MONTH_COL,
        MONTH_CODE_COL,
        DEATHS_COL,
        PREDICTED_DEATHS_COL,
        EXCESS_DEATHS_COL,
    ]
    prediction_keep = [
        "Disease",
        "Model",
        "Model Label",
        "Validation Year",
        "Training Years",
        YEAR_COL,
        MONTH_COL,
        MONTH_CODE_COL,
        DEATHS_COL,
        PREDICTED_DEATHS_COL,
        EXCESS_DEATHS_COL,
    ]

    validation_df[validation_keep].to_csv(
        TABLE_DIR / "monthly_deaths_validation_2018_2019.csv",
        index=False,
    )
    prediction_df[prediction_keep].to_csv(
        TABLE_DIR / "monthly_deaths_predicted_2020_2025.csv",
        index=False,
    )

    yearly_summary, overall_summary = _covid_summary(prediction_df)
    num_cols = [
        "Mean Actual Deaths",
        "Mean Predicted Deaths",
        "Mean Excess Deaths",
        "Total Actual Deaths",
        "Total Predicted Deaths",
        "Total Excess Deaths",
    ]
    yearly_summary[num_cols] = yearly_summary[num_cols].apply(pd.to_numeric, errors="coerce").round(1)
    overall_summary[num_cols] = overall_summary[num_cols].apply(pd.to_numeric, errors="coerce").round(1)

    yearly_summary.to_csv(
        TABLE_DIR / "monthly_deaths_excess_summary_covid_2020_2023_by_year.csv",
        index=False,
    )
    overall_summary.to_csv(
        TABLE_DIR / "monthly_deaths_excess_summary_covid_2020_2023_overall.csv",
        index=False,
    )


if __name__ == "__main__":
    build_monthly_deaths_excess_outputs()
