from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FIGURE_ROOT = ROOT / "results" / "figures"

DEST_BY_PREFIX = {
    "all_trends_by_sex_panel": "sex",
    "sepsis_trend_by_sex": "sex",
    "pneumonia_trend_by_sex": "sex",
    "combined_trend_by_sex": "sex",
    "all_trends_by_age_panel": "age",
    "sepsis_trend_by_age": "age",
    "pneumonia_trend_by_age": "age",
    "combined_trend_by_age": "age",
    "total_trend_three_diseases": "overall",
    "covid_vs_noncovid_pneumonia_relation": "overall",
}


def _target_subfolder(file_name: str) -> str | None:
    for prefix, folder in DEST_BY_PREFIX.items():
        if file_name.startswith(prefix):
            return folder
    return None


def organize_figures() -> None:
    for folder in {"sex", "age", "overall"}:
        (FIGURE_ROOT / folder).mkdir(parents=True, exist_ok=True)

    for file_path in FIGURE_ROOT.iterdir():
        if file_path.is_dir():
            continue
        target_folder = _target_subfolder(file_path.name)
        if target_folder is None:
            continue
        destination = FIGURE_ROOT / target_folder / file_path.name
        file_path.replace(destination)


if __name__ == "__main__":
    organize_figures()
