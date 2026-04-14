from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[2]
CLEANED_DIR = ROOT / "data" / "cleaned"
TABLE_DIR = ROOT / "results" / "tables"
FIGURE_DIR = ROOT / "results" / "figures" / "monthly_regression"

YEAR_COL = "Year"
MONTH_COL = "Month"
MONTH_CODE_COL = "Month Code"
AAMR_COL = "Age Adjusted Rate"
PREDICTED_COL = "Predicted AAMR"
DIFFERENCE_COL = "Actual - Predicted"

TRAIN_START_YEAR = 2010
VALIDATION_YEARS = [2018, 2019]
PRESPECIFIED_MODEL_NAME = "quadratic_month_fe"
MODEL_TRAIN_END_YEAR = 2019
PREDICT_START_YEAR = 2020
PREDICT_END_YEAR = 2025
MIN_PROJECTED_AAMR = 0.0
MIN_2025_TO_2019_MEAN_RATIO = 0.80

DATASETS = {
    "Pneumonia/ARDS": CLEANED_DIR / "ards_month.csv",
    "Pneumonia": CLEANED_DIR / "pneumonia_month.csv",
    "Pneumonia/Sepsis": CLEANED_DIR / "combined_month.csv",
}

PLOT_COLORS = {
    "Actual": "#1b3c73",
    "Predicted": "#b33b2e",
}

SENSITIVITY_COLORS = {
    "Selected best model": "#1b3c73",
    "Prespecified quadratic model": "#b33b2e",
}

MODEL_SPECS = [
    {
        "Name": "linear_month_fe",
        "Label": "Linear trend + month effects",
        "Feature Set": "month_fe",
        "Target": "identity",
    },
    {
        "Name": "quadratic_month_fe",
        "Label": "Quadratic trend + month effects",
        "Feature Set": "quadratic_month_fe",
        "Target": "identity",
    },
    {
        "Name": "cubic_month_fe",
        "Label": "Cubic trend + month effects",
        "Feature Set": "cubic_month_fe",
        "Target": "identity",
    },
    {
        "Name": "linear_fourier",
        "Label": "Linear trend + Fourier seasonality",
        "Feature Set": "fourier",
        "Target": "identity",
    },
    {
        "Name": "quadratic_fourier",
        "Label": "Quadratic trend + Fourier seasonality",
        "Feature Set": "quadratic_fourier",
        "Target": "identity",
    },
    {
        "Name": "log_linear_month_fe",
        "Label": "Log-linear trend + month effects",
        "Feature Set": "month_fe",
        "Target": "log",
    },
    {
        "Name": "log_quadratic_month_fe",
        "Label": "Log-quadratic trend + month effects",
        "Feature Set": "quadratic_month_fe",
        "Target": "log",
    },
    {
        "Name": "log_cubic_month_fe",
        "Label": "Log-cubic trend + month effects",
        "Feature Set": "cubic_month_fe",
        "Target": "log",
    },
    *[
        {
            "Name": f"recent_{start_year}_quadratic_month_fe",
            "Label": f"{start_year}+ quadratic trend + month effects",
            "Feature Set": "quadratic_month_fe",
            "Target": "identity",
            "Train Start Year": start_year,
        }
        for start_year in [2012, 2013, 2014, 2015]
    ],
    *[
        {
            "Name": f"recency_weighted_{half_life}_quadratic_month_fe",
            "Label": f"Recency-weighted quadratic trend + month effects ({half_life}-month half-life)",
            "Feature Set": "quadratic_month_fe",
            "Target": "identity",
            "Half Life Months": half_life,
        }
        for half_life in [24, 36, 48, 60, 72]
    ],
    *[
        {
            "Name": f"log_recency_weighted_{half_life}_quadratic_month_fe",
            "Label": f"Log recency-weighted quadratic trend + month effects ({half_life}-month half-life)",
            "Feature Set": "quadratic_month_fe",
            "Target": "log",
            "Half Life Months": half_life,
        }
        for half_life in [24, 36, 48, 60, 72]
    ],
    *[
        {
            "Name": f"anchored_quadratic_month_fe_{int(blend_weight * 100)}",
            "Label": f"Quadratic trend + month effects, anchored {int(blend_weight * 100)}% to recent seasonal mean",
            "Feature Set": "quadratic_month_fe",
            "Target": "identity",
            "Blend Weight": blend_weight,
            "Anchor": "recent_3_year_month_mean",
        }
        for blend_weight in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    ],
    *[
        {
            "Name": f"log_anchored_quadratic_month_fe_{int(blend_weight * 100)}",
            "Label": f"Log-quadratic trend + month effects, anchored {int(blend_weight * 100)}% to recent seasonal mean",
            "Feature Set": "quadratic_month_fe",
            "Target": "log",
            "Blend Weight": blend_weight,
            "Anchor": "recent_3_year_month_mean",
        }
        for blend_weight in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    ],
    *[
        {
            "Name": f"anchored_quadratic_fourier_{int(blend_weight * 100)}",
            "Label": f"Quadratic trend + Fourier seasonality, anchored {int(blend_weight * 100)}% to recent seasonal mean",
            "Feature Set": "quadratic_fourier",
            "Target": "identity",
            "Blend Weight": blend_weight,
            "Anchor": "recent_3_year_month_mean",
        }
        for blend_weight in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    ],
    *[
        {
            "Name": f"log_anchored_quadratic_fourier_{int(blend_weight * 100)}",
            "Label": f"Log-quadratic trend + Fourier seasonality, anchored {int(blend_weight * 100)}% to recent seasonal mean",
            "Feature Set": "quadratic_fourier",
            "Target": "log",
            "Blend Weight": blend_weight,
            "Anchor": "recent_3_year_month_mean",
        }
        for blend_weight in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    ],
]


def _get_model_spec(model_name: str) -> dict[str, str]:
    for model_spec in MODEL_SPECS:
        if model_spec["Name"] == model_name:
            return model_spec
    raise ValueError(f"Unknown model specification: {model_name}")


def _apply_publication_style() -> None:
    sns.set_theme(
        style="white",
        context="paper",
        rc={
            "figure.dpi": 300,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.labelsize": 10.5,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.linewidth": 0.9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 4,
            "ytick.major.size": 4,
            "grid.color": "#e4e0d8",
            "grid.linewidth": 0.55,
            "grid.linestyle": "--",
            "legend.frameon": False,
            "legend.fontsize": 9,
        },
    )


def _month_number_from_code(month_code: pd.Series) -> pd.Series:
    return pd.to_numeric(
        month_code.astype("string").str.extract(r"/(\d{2})$", expand=False),
        errors="coerce",
    )


def _read_monthly_aamr(csv_path: Path, disease: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, dtype={YEAR_COL: "string", MONTH_CODE_COL: "string"})
    df[YEAR_COL] = (
        df[YEAR_COL]
        .astype("string")
        .str.strip()
        .str.replace(r"\s*\(provisional\)$", "", regex=True)
    )
    df[YEAR_COL] = pd.to_numeric(df[YEAR_COL], errors="coerce")
    df["Month Number"] = _month_number_from_code(df[MONTH_CODE_COL])
    df[AAMR_COL] = pd.to_numeric(df[AAMR_COL], errors="coerce")
    df = df.dropna(subset=[YEAR_COL, "Month Number", AAMR_COL]).copy()
    df[YEAR_COL] = df[YEAR_COL].astype(int)
    df["Month Number"] = df["Month Number"].astype(int)
    df["Time Index"] = ((df[YEAR_COL] - TRAIN_START_YEAR) * 12) + (df["Month Number"] - 1)
    df["Disease"] = disease
    return df.sort_values([YEAR_COL, "Month Number"]).reset_index(drop=True)


def _month_effects(df: pd.DataFrame) -> pd.DataFrame:
    month_dummies = pd.get_dummies(df["Month Number"], prefix="month", dtype=float)
    for month in range(2, 13):
        col = f"month_{month}"
        if col not in month_dummies.columns:
            month_dummies[col] = 0.0
    return month_dummies[[f"month_{month}" for month in range(2, 13)]]


def _month_specific_terms(df: pd.DataFrame) -> pd.DataFrame:
    month_terms = pd.DataFrame(index=df.index)
    for month in range(1, 13):
        indicator = df["Month Number"].eq(month).astype(float)
        month_terms[f"month_{month}_intercept"] = indicator
        month_terms[f"month_{month}_trend"] = indicator * df["Time Index"].astype(float)
    return month_terms


def _fourier_terms(df: pd.DataFrame) -> pd.DataFrame:
    angle = (2 * np.pi * (df["Month Number"].astype(float) - 1)) / 12
    return pd.DataFrame(
        {
            "sin_1": np.sin(angle),
            "cos_1": np.cos(angle),
            "sin_2": np.sin(2 * angle),
            "cos_2": np.cos(2 * angle),
        },
        index=df.index,
    )


def _design_matrix(df: pd.DataFrame, feature_set: str) -> np.ndarray:
    design = pd.concat(
        [
            pd.Series(1.0, index=df.index, name="Intercept"),
            df["Time Index"].astype(float).rename("Time Index"),
        ],
        axis=1,
    )

    if feature_set == "month_fe":
        design = pd.concat([design, _month_effects(df)], axis=1)
    elif feature_set == "quadratic_month_fe":
        design = pd.concat(
            [
                design,
                (df["Time Index"].astype(float) ** 2).rename("Time Index Squared"),
                _month_effects(df),
            ],
            axis=1,
        )
    elif feature_set == "cubic_month_fe":
        time_index = df["Time Index"].astype(float)
        design = pd.concat(
            [
                design,
                (time_index**2).rename("Time Index Squared"),
                (time_index**3).rename("Time Index Cubed"),
                _month_effects(df),
            ],
            axis=1,
        )
    elif feature_set == "fourier":
        design = pd.concat([design, _fourier_terms(df)], axis=1)
    elif feature_set == "quadratic_fourier":
        design = pd.concat(
            [
                design,
                (df["Time Index"].astype(float) ** 2).rename("Time Index Squared"),
                _fourier_terms(df),
            ],
            axis=1,
        )
    elif feature_set == "month_specific_linear":
        design = _month_specific_terms(df)
    else:
        raise ValueError(f"Unknown feature set: {feature_set}")

    return design.to_numpy(dtype=float)


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


def _model_training_subset(train_df: pd.DataFrame, model_spec: dict[str, str]) -> pd.DataFrame:
    start_year = model_spec.get("Train Start Year")
    if start_year is None:
        return train_df.copy()
    return train_df[train_df[YEAR_COL].ge(int(start_year))].copy()


def _fit_linear_regression(train_df: pd.DataFrame, model_spec: dict[str, str]) -> np.ndarray:
    train_df = _model_training_subset(train_df, model_spec)
    x = _design_matrix(train_df, model_spec["Feature Set"])
    y = train_df[AAMR_COL].to_numpy(dtype=float)
    y = _transform_target(y, model_spec["Target"])
    half_life = model_spec.get("Half Life Months")
    if half_life is not None:
        max_time = train_df["Time Index"].max()
        weights = 0.5 ** ((max_time - train_df["Time Index"].to_numpy(dtype=float)) / float(half_life))
        sqrt_weights = np.sqrt(weights)
        x = x * sqrt_weights[:, None]
        y = y * sqrt_weights
    coefficients, *_ = np.linalg.lstsq(x, y, rcond=None)
    return coefficients


def _anchor_values(df: pd.DataFrame, train_df: pd.DataFrame, model_spec: dict[str, str]) -> np.ndarray:
    train_df = _model_training_subset(train_df, model_spec)
    anchor = model_spec.get("Anchor")
    if anchor == "recent_3_year_month_mean":
        recent_years = sorted(train_df[YEAR_COL].dropna().unique())[-3:]
        anchor_by_month = (
            train_df[train_df[YEAR_COL].isin(recent_years)]
            .groupby("Month Number")[AAMR_COL]
            .mean()
        )
        return df["Month Number"].map(anchor_by_month).to_numpy(dtype=float)
    raise ValueError(f"Unknown anchor specification: {anchor}")


def _predict(
    df: pd.DataFrame,
    coefficients: np.ndarray,
    model_spec: dict[str, str],
    train_df: pd.DataFrame | None = None,
) -> np.ndarray:
    predicted = _design_matrix(df, model_spec["Feature Set"]) @ coefficients
    predicted = _inverse_transform(predicted, model_spec["Target"])

    blend_weight = model_spec.get("Blend Weight")
    if blend_weight is None:
        return predicted
    if train_df is None:
        raise ValueError("Anchored model predictions require the training data.")

    anchor = _anchor_values(df, train_df, model_spec)
    return ((1 - float(blend_weight)) * predicted) + (float(blend_weight) * anchor)


def _validation_metrics(validation_df: pd.DataFrame) -> dict[str, float]:
    actual = validation_df[AAMR_COL].to_numpy(dtype=float)
    predicted = validation_df[PREDICTED_COL].to_numpy(dtype=float)
    residual = actual - predicted
    mae = np.mean(np.abs(residual))
    rmse = np.sqrt(np.mean(residual**2))
    mape = np.mean(np.abs(residual / actual)) * 100
    ss_res = np.sum(residual**2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot else np.nan
    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE (%)": mape,
        "R-squared": r_squared,
    }


def _rolling_validation_prediction(df: pd.DataFrame, model_spec: dict[str, str]) -> pd.DataFrame:
    validation_frames = []
    for validation_year in VALIDATION_YEARS:
        train_df = df[df[YEAR_COL].between(TRAIN_START_YEAR, validation_year - 1)].copy()
        target_df = df[df[YEAR_COL].eq(validation_year)].copy()
        coefficients = _fit_linear_regression(train_df, model_spec)
        target_df["Validation Year"] = validation_year
        target_df["Training Years"] = f"{TRAIN_START_YEAR}-{validation_year - 1}"
        target_df[PREDICTED_COL] = _predict(target_df, coefficients, model_spec, train_df).round(1)
        target_df[DIFFERENCE_COL] = (
            target_df[AAMR_COL] - target_df[PREDICTED_COL]
        ).round(1)
        validation_frames.append(target_df)
    return pd.concat(validation_frames, ignore_index=True)


def _projection_diagnostics(df: pd.DataFrame, model_spec: dict[str, str]) -> dict[str, float | bool]:
    projection_df = _predict_2020_2025(df, model_spec)
    annual_projected = projection_df.groupby(YEAR_COL)[PREDICTED_COL].mean()
    baseline_2019_mean = df[df[YEAR_COL].eq(MODEL_TRAIN_END_YEAR)][AAMR_COL].mean()
    projected_2025_mean = annual_projected.get(PREDICT_END_YEAR, np.nan)
    projected_ratio = projected_2025_mean / baseline_2019_mean if baseline_2019_mean else np.nan
    minimum_projected = projection_df[PREDICTED_COL].min()
    projection_plausible = (
        pd.notna(minimum_projected)
        and pd.notna(projected_ratio)
        and minimum_projected >= MIN_PROJECTED_AAMR
        and projected_ratio >= MIN_2025_TO_2019_MEAN_RATIO
    )
    return {
        "Projection Plausible": bool(projection_plausible),
        "Minimum Projected AAMR": minimum_projected,
        "2019 Mean AAMR": baseline_2019_mean,
        "2025 Mean Predicted AAMR": projected_2025_mean,
        "2025/2019 Predicted Ratio": projected_ratio,
    }


def _evaluate_models(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    evaluation_rows = []
    best_spec = None
    best_rmse = np.inf
    best_unrestricted_spec = None
    best_unrestricted_rmse = np.inf

    for model_spec in MODEL_SPECS:
        evaluated = _rolling_validation_prediction(df, model_spec)
        metrics = _validation_metrics(evaluated)
        projection_diagnostics = _projection_diagnostics(df, model_spec)
        metrics["Model"] = model_spec["Name"]
        metrics["Model Label"] = model_spec["Label"]
        metrics.update(projection_diagnostics)
        evaluation_rows.append(metrics)

        if metrics["RMSE"] < best_unrestricted_rmse:
            best_unrestricted_rmse = metrics["RMSE"]
            best_unrestricted_spec = model_spec

        if projection_diagnostics["Projection Plausible"] and metrics["RMSE"] < best_rmse:
            best_rmse = metrics["RMSE"]
            best_spec = model_spec

    if best_spec is None:
        if best_unrestricted_spec is None:
            raise ValueError("No model could be evaluated.")
        best_spec = best_unrestricted_spec

    evaluation_df = pd.DataFrame(evaluation_rows)
    return evaluation_df, best_spec


def _add_model_metadata(df: pd.DataFrame, analysis_model: str, model_spec: dict[str, str]) -> pd.DataFrame:
    out = df.copy()
    out["Analysis Model"] = analysis_model
    out["Model"] = model_spec["Name"]
    out["Model Label"] = model_spec["Label"]
    return out


def _predict_2020_2025(df: pd.DataFrame, model_spec: dict[str, str]) -> pd.DataFrame:
    model_train = df[df[YEAR_COL].between(TRAIN_START_YEAR, MODEL_TRAIN_END_YEAR)].copy()
    prediction_target = df[df[YEAR_COL].between(PREDICT_START_YEAR, PREDICT_END_YEAR)].copy()
    model_coefficients = _fit_linear_regression(model_train, model_spec)
    prediction_target[PREDICTED_COL] = _predict(
        prediction_target,
        model_coefficients,
        model_spec,
        model_train,
    ).round(1)
    prediction_target[DIFFERENCE_COL] = (
        prediction_target[AAMR_COL] - prediction_target[PREDICTED_COL]
    ).round(1)
    return prediction_target


def _build_validation_and_prediction_tables() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    validation_frames = []
    prediction_frames = []
    metric_rows = []
    comparison_frames = []
    sensitivity_validation_frames = []
    sensitivity_prediction_frames = []
    sensitivity_metric_rows = []
    sensitivity_summary_frames = []
    prespecified_spec = _get_model_spec(PRESPECIFIED_MODEL_NAME)

    for disease, csv_path in DATASETS.items():
        df = _read_monthly_aamr(csv_path, disease)

        model_comparison, best_spec = _evaluate_models(df)
        model_comparison["Disease"] = disease
        comparison_frames.append(model_comparison)

        validation_target = _rolling_validation_prediction(df, best_spec)
        validation_target["Selected Model"] = best_spec["Name"]
        validation_target["Selected Model Label"] = best_spec["Label"]
        validation_frames.append(validation_target)

        metrics = _validation_metrics(validation_target)
        metrics["Disease"] = disease
        metrics["Selected Model"] = best_spec["Name"]
        metrics["Selected Model Label"] = best_spec["Label"]
        metric_rows.append(metrics)

        prediction_target = _predict_2020_2025(df, best_spec)
        prediction_target["Selected Model"] = best_spec["Name"]
        prediction_target["Selected Model Label"] = best_spec["Label"]
        prediction_frames.append(prediction_target)

        sensitivity_specs = [
            ("Selected best model", best_spec),
            ("Prespecified quadratic model", prespecified_spec),
        ]
        for analysis_model, model_spec in sensitivity_specs:
            sensitivity_validation = _rolling_validation_prediction(df, model_spec)
            sensitivity_validation = _add_model_metadata(
                sensitivity_validation,
                analysis_model,
                model_spec,
            )
            sensitivity_validation_frames.append(sensitivity_validation)

            sensitivity_metrics = _validation_metrics(sensitivity_validation)
            sensitivity_metrics["Disease"] = disease
            sensitivity_metrics["Analysis Model"] = analysis_model
            sensitivity_metrics["Model"] = model_spec["Name"]
            sensitivity_metrics["Model Label"] = model_spec["Label"]
            sensitivity_metric_rows.append(sensitivity_metrics)

            sensitivity_prediction = _predict_2020_2025(df, model_spec)
            sensitivity_prediction = _add_model_metadata(
                sensitivity_prediction,
                analysis_model,
                model_spec,
            )
            sensitivity_prediction_frames.append(sensitivity_prediction)

            sensitivity_summary = (
                sensitivity_prediction.groupby(
                    ["Disease", "Analysis Model", "Model", "Model Label", YEAR_COL],
                    as_index=False,
                )
                .agg(
                    **{
                        "Mean Actual AAMR": (AAMR_COL, "mean"),
                        "Mean Predicted AAMR": (PREDICTED_COL, "mean"),
                        "Mean Actual - Predicted": (DIFFERENCE_COL, "mean"),
                    }
                )
            )
            sensitivity_summary_frames.append(sensitivity_summary)

    validation_df = pd.concat(validation_frames, ignore_index=True)
    prediction_df = pd.concat(prediction_frames, ignore_index=True)
    comparison_df = pd.concat(comparison_frames, ignore_index=True)
    sensitivity_validation_df = pd.concat(sensitivity_validation_frames, ignore_index=True)
    sensitivity_prediction_df = pd.concat(sensitivity_prediction_frames, ignore_index=True)
    sensitivity_summary_df = pd.concat(sensitivity_summary_frames, ignore_index=True)
    metrics_df = pd.DataFrame(metric_rows)
    sensitivity_metrics_df = pd.DataFrame(sensitivity_metric_rows)
    metrics_df = metrics_df[
        ["Disease", "Selected Model", "Selected Model Label", "MAE", "RMSE", "MAPE (%)", "R-squared"]
    ]
    sensitivity_metrics_df = sensitivity_metrics_df[
        ["Disease", "Analysis Model", "Model", "Model Label", "MAE", "RMSE", "MAPE (%)", "R-squared"]
    ].sort_values(["Disease", "Analysis Model"])
    sensitivity_summary_df[
        ["Mean Actual AAMR", "Mean Predicted AAMR", "Mean Actual - Predicted"]
    ] = sensitivity_summary_df[
        ["Mean Actual AAMR", "Mean Predicted AAMR", "Mean Actual - Predicted"]
    ].round(3)
    comparison_df = comparison_df[
        [
            "Disease",
            "Model",
            "Model Label",
            "Projection Plausible",
            "Minimum Projected AAMR",
            "2025 Mean Predicted AAMR",
            "2025/2019 Predicted Ratio",
            "MAE",
            "RMSE",
            "MAPE (%)",
            "R-squared",
        ]
    ].sort_values(["Disease", "RMSE"])
    return (
        validation_df,
        prediction_df,
        metrics_df,
        comparison_df,
        sensitivity_validation_df,
        sensitivity_prediction_df,
        sensitivity_metrics_df,
        sensitivity_summary_df,
    )


def _write_tables(
    validation_df: pd.DataFrame,
    prediction_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    sensitivity_validation_df: pd.DataFrame,
    sensitivity_prediction_df: pd.DataFrame,
    sensitivity_metrics_df: pd.DataFrame,
    sensitivity_summary_df: pd.DataFrame,
) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    validation_keep_cols = [
        "Disease",
        "Selected Model",
        "Selected Model Label",
        "Validation Year",
        "Training Years",
        YEAR_COL,
        MONTH_COL,
        MONTH_CODE_COL,
        AAMR_COL,
        PREDICTED_COL,
        DIFFERENCE_COL,
    ]
    prediction_keep_cols = [
        "Disease",
        "Selected Model",
        "Selected Model Label",
        YEAR_COL,
        MONTH_COL,
        MONTH_CODE_COL,
        AAMR_COL,
        PREDICTED_COL,
        DIFFERENCE_COL,
    ]
    sensitivity_validation_keep_cols = [
        "Disease",
        "Analysis Model",
        "Model",
        "Model Label",
        "Validation Year",
        "Training Years",
        YEAR_COL,
        MONTH_COL,
        MONTH_CODE_COL,
        AAMR_COL,
        PREDICTED_COL,
        DIFFERENCE_COL,
    ]
    sensitivity_prediction_keep_cols = [
        "Disease",
        "Analysis Model",
        "Model",
        "Model Label",
        YEAR_COL,
        MONTH_COL,
        MONTH_CODE_COL,
        AAMR_COL,
        PREDICTED_COL,
        DIFFERENCE_COL,
    ]
    validation_df[validation_keep_cols].to_csv(
        TABLE_DIR / "monthly_aamr_validation_2018_2019.csv",
        index=False,
    )
    prediction_df[prediction_keep_cols].to_csv(
        TABLE_DIR / "monthly_aamr_predicted_2020_2025.csv",
        index=False,
    )
    metrics_df.to_csv(TABLE_DIR / "monthly_aamr_validation_metrics.csv", index=False)
    comparison_df.to_csv(TABLE_DIR / "monthly_aamr_model_comparison_2018_2019.csv", index=False)
    sensitivity_validation_df[sensitivity_validation_keep_cols].to_csv(
        TABLE_DIR / "monthly_aamr_validation_selected_vs_prespecified_2018_2019.csv",
        index=False,
    )
    sensitivity_prediction_df[sensitivity_prediction_keep_cols].to_csv(
        TABLE_DIR / "monthly_aamr_predicted_selected_vs_prespecified_2020_2025.csv",
        index=False,
    )
    sensitivity_metrics_df.to_csv(
        TABLE_DIR / "monthly_aamr_validation_metrics_selected_vs_prespecified.csv",
        index=False,
    )
    sensitivity_summary_df.to_csv(
        TABLE_DIR / "monthly_aamr_excess_summary_selected_vs_prespecified_2020_2025.csv",
        index=False,
    )


def _plot_validation(validation_df: pd.DataFrame) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    _apply_publication_style()

    panel_order = ["Pneumonia/ARDS", "Pneumonia", "Pneumonia/Sepsis"]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8), sharex=True)

    for ax, disease in zip(axes, panel_order):
        disease_df = validation_df[validation_df["Disease"].eq(disease)].sort_values([YEAR_COL, "Month Number"])
        model_label = disease_df["Selected Model Label"].iloc[0]
        disease_df["Validation Time"] = (
            (disease_df[YEAR_COL] - min(VALIDATION_YEARS)) * 12
            + disease_df["Month Number"]
        )
        plot_df = disease_df.melt(
            id_vars=["Validation Time", YEAR_COL, "Month Number", MONTH_COL],
            value_vars=[AAMR_COL, PREDICTED_COL],
            var_name="Series",
            value_name="AAMR",
        )
        plot_df["Series"] = plot_df["Series"].replace(
            {
                AAMR_COL: "Actual",
                PREDICTED_COL: "Predicted",
            }
        )

        sns.lineplot(
            data=plot_df,
            x="Validation Time",
            y="AAMR",
            hue="Series",
            marker="o",
            linewidth=1.9,
            markersize=5,
            dashes=False,
            palette=PLOT_COLORS,
            ax=ax,
        )
        ax.set_title(f"{disease}\n{model_label}")
        ax.set_xlabel("")
        ax.set_ylabel("Monthly AAMR")
        tick_positions = [1, 4, 7, 10, 13, 16, 19, 22]
        tick_labels = ["Jan 2018", "Apr", "Jul", "Oct", "Jan 2019", "Apr", "Jul", "Oct"]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right")
        ax.axvline(12.5, color="#8c8c8c", linestyle=":", linewidth=1.0)
        ax.grid(axis="y", alpha=0.8)
        ax.grid(axis="x", visible=False)
        if ax is not axes[0]:
            ax.set_ylabel("")
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.06),
        ncol=2,
        frameon=False,
        handlelength=2.0,
        columnspacing=1.4,
    )
    fig.supxlabel("Validation month")
    fig.suptitle("Validation of monthly AAMR regression: rolling predictions for 2018 and 2019", y=1.13)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "monthly_aamr_validation_2018_2019.png", dpi=300)
    fig.savefig(FIGURE_DIR / "monthly_aamr_validation_2018_2019.pdf")
    plt.close(fig)


def _plot_sensitivity_excess(sensitivity_prediction_df: pd.DataFrame) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    _apply_publication_style()

    panel_order = ["Pneumonia/ARDS", "Pneumonia", "Pneumonia/Sepsis"]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8), sharex=True)

    for ax, disease in zip(axes, panel_order):
        disease_df = sensitivity_prediction_df[
            sensitivity_prediction_df["Disease"].eq(disease)
        ].sort_values([YEAR_COL, "Month Number", "Analysis Model"])
        disease_df["Prediction Time"] = (
            (disease_df[YEAR_COL] - PREDICT_START_YEAR) * 12
            + disease_df["Month Number"]
        )

        sns.lineplot(
            data=disease_df,
            x="Prediction Time",
            y=DIFFERENCE_COL,
            hue="Analysis Model",
            linewidth=1.6,
            dashes=False,
            palette=SENSITIVITY_COLORS,
            ax=ax,
        )
        ax.axhline(0, color="#5f5f5f", linewidth=0.9, linestyle="--")
        ax.set_title(disease)
        ax.set_xlabel("")
        ax.set_ylabel("Actual - predicted AAMR")
        tick_positions = [1, 13, 25, 37, 49, 61]
        tick_labels = ["2020", "2021", "2022", "2023", "2024", "2025"]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right")
        ax.grid(axis="y", alpha=0.8)
        ax.grid(axis="x", visible=False)
        if ax is not axes[0]:
            ax.set_ylabel("")
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.06),
        ncol=2,
        frameon=False,
        handlelength=2.0,
        columnspacing=1.4,
    )
    fig.supxlabel("Prediction year")
    fig.suptitle("Sensitivity of excess monthly AAMR: selected model vs prespecified quadratic model", y=1.13)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "monthly_aamr_excess_selected_vs_prespecified_2020_2025.png", dpi=300)
    fig.savefig(FIGURE_DIR / "monthly_aamr_excess_selected_vs_prespecified_2020_2025.pdf")
    plt.close(fig)


def build_monthly_aamr_regression_outputs() -> None:
    (
        validation_df,
        prediction_df,
        metrics_df,
        comparison_df,
        sensitivity_validation_df,
        sensitivity_prediction_df,
        sensitivity_metrics_df,
        sensitivity_summary_df,
    ) = _build_validation_and_prediction_tables()
    _write_tables(
        validation_df,
        prediction_df,
        metrics_df,
        comparison_df,
        sensitivity_validation_df,
        sensitivity_prediction_df,
        sensitivity_metrics_df,
        sensitivity_summary_df,
    )
    _plot_validation(validation_df)
    _plot_sensitivity_excess(sensitivity_prediction_df)


if __name__ == "__main__":
    build_monthly_aamr_regression_outputs()

