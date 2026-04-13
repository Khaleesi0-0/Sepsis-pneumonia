from __future__ import annotations

import importlib
import subprocess
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd


def _ensure_geopandas():
    try:
        return importlib.import_module("geopandas")
    except ModuleNotFoundError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "geopandas"])
        return importlib.import_module("geopandas")


gpd = _ensure_geopandas()

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import MaxNLocator, StrMethodFormatter
from shapely.geometry import MultiPolygon
from shapely.affinity import rotate as shp_rotate
from shapely.affinity import scale as shp_scale
from shapely.affinity import translate as shp_translate


ROOT = Path(__file__).resolve().parents[2]
CLEANED_DIR = ROOT / "data" / "cleaned"
FIGURE_DIR = ROOT / "results" / "figures" / "maps"
CACHE_DIR = ROOT / "results" / "maps_cache"

STATE_SHAPE_URL = "https://www2.census.gov/geo/tiger/TIGER2024/STATE/tl_2024_us_state.zip"
STATE_COL = "State"
STATE_COL_CANDIDATES = ["State", "Residence State"]
YEAR_COL = "Year"
YEAR_CODE_COL = "Year Code"
DEATHS_COL = "Deaths"
POP_COL = "Population"
AAMR_COL = "Age Adjusted Rate"
NOTES_COL = "Notes"

PERIODS = [
    ("Pre-pandemic\n(2010–2019)", range(2010, 2020)),
    ("Pandemic\n(2020–2023)", range(2020, 2024)),
    ("Post-pandemic\n(2024–2025)", range(2024, 2026)),
]

DISEASE_STYLES = {
    "ARDS (ARDS + Pneumonia)": {"cmap": "Blues", "cbar_label": "AAMR (/100,000)", "title_color": "#1f4e79"},
    "Pneumonia": {"cmap": "Oranges", "cbar_label": "AAMR (/100,000)", "title_color": "#8a4b08"},
    "Sepsis+ Pneumonia": {"cmap": "Greens", "cbar_label": "AAMR (/100,000)", "title_color": "#215a33"},
}

DATASETS = {
    "ARDS (ARDS + Pneumonia)": CLEANED_DIR / "ards_state.csv",
    "Pneumonia": CLEANED_DIR / "pneumonia_state.csv",
    "Sepsis+ Pneumonia": CLEANED_DIR / "combined_state.csv",
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


def _ensure_states_shapefile() -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = CACHE_DIR / "tl_2024_us_state.zip"
    shp_path = CACHE_DIR / "tl_2024_us_state.shp"

    if shp_path.exists():
        return shp_path

    if not zip_path.exists():
        urlretrieve(STATE_SHAPE_URL, zip_path)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(CACHE_DIR)

    return shp_path


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
    drop_cols = [col for col in STATE_COL_CANDIDATES if col in df.columns and col != state_source_col]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    df[YEAR_COL] = (
        df[YEAR_COL]
        .astype("string")
        .str.strip()
        .str.replace(r"\s*\(provisional\)$", "", regex=True)
    )
    df["__year"] = pd.to_numeric(df[YEAR_COL], errors="coerce")
    df[DEATHS_COL] = pd.to_numeric(df[DEATHS_COL], errors="coerce")
    df[POP_COL] = pd.to_numeric(df[POP_COL], errors="coerce")
    df[AAMR_COL] = pd.to_numeric(df[AAMR_COL], errors="coerce")
    df = df[df[STATE_COL].astype("string").isin(KEEP_STATES)].copy()
    df = df[df["__year"].notna()].copy()
    return df


def _period_aggregate(df: pd.DataFrame, years: range) -> pd.DataFrame:
    period_df = df[df["__year"].isin(list(years))].copy()
    rows = []
    for state_name, state_df in period_df.groupby(STATE_COL):
        total_deaths = state_df[DEATHS_COL].sum()
        total_pop = state_df[POP_COL].sum()
        aammr = (
            (state_df[AAMR_COL] * state_df[POP_COL]).sum() / total_pop
            if total_pop and pd.notna(total_pop)
            else pd.NA
        )
        rows.append(
            {
                STATE_COL: state_name,
                DEATHS_COL: total_deaths,
                POP_COL: total_pop,
                AAMR_COL: aammr,
            }
        )
    grouped = pd.DataFrame(rows)
    return grouped


def _load_maps() -> gpd.GeoDataFrame:
    shp_path = _ensure_states_shapefile()
    geo = gpd.read_file(shp_path)
    geo = geo[geo["NAME"].isin(KEEP_STATES)].copy()
    geo = geo.to_crs("EPSG:2163")
    return _reposition_alaska_hawaii(geo)


def _reposition_alaska_hawaii(geo: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    out = geo.copy()
    mainland = out[~out["NAME"].isin(["Alaska", "Hawaii"])].copy()
    if mainland.empty:
        return out

    minx, miny, maxx, maxy = mainland.total_bounds
    width = maxx - minx
    height = maxy - miny

    targets = {
        "Alaska": {
            "scale": (2, 2),
            "anchor": (minx - 0.15 * width, miny - 0.060 * height),
            "rotation": 32,
        },
        "Hawaii": {
            "scale": (0.7, 0.7),
            "anchor": (minx + 0.2 * width, miny + 0.01* height),
            "rotation": -35,
        },
    }

    for state_name, params in targets.items():
        mask = out["NAME"].eq(state_name)
        if not mask.any():
            continue
        geom = out.loc[mask, "geometry"].iloc[0]
        if state_name == "Hawaii":
            geom = _largest_polygons(geom, n=3)
        src_minx, src_miny, src_maxx, src_maxy = geom.bounds
        src_w = src_maxx - src_minx
        src_h = src_maxy - src_miny
        if src_w == 0 or src_h == 0:
            continue

        target_w = width * 0.18 if state_name == "Hawaii" else width * 0.24
        target_h = height * 0.10 if state_name == "Hawaii" else height * 0.18
        sx, sy = params["scale"]
        scaled = shp_scale(
            geom,
            xfact=sx * (target_w / src_w),
            yfact=sy * (target_h / src_h),
            origin="center",
        )
        rotated = shp_rotate(scaled, params["rotation"], origin="center")
        tgt_x, tgt_y = params["anchor"]
        tgt_cx = tgt_x + target_w / 2
        tgt_cy = tgt_y + target_h / 2
        cur_minx, cur_miny, cur_maxx, cur_maxy = rotated.bounds
        cur_cx = (cur_minx + cur_maxx) / 2
        cur_cy = (cur_miny + cur_maxy) / 2
        shifted = shp_translate(rotated, xoff=tgt_cx - cur_cx, yoff=tgt_cy - cur_cy)
        out.loc[mask, "geometry"] = [shifted]

    return out


def _largest_polygons(geom, n: int):
    if isinstance(geom, MultiPolygon):
        parts = sorted(geom.geoms, key=lambda part: part.area, reverse=True)[:n]
        return MultiPolygon(parts)
    return geom


def _plot_panel(ax, geo: gpd.GeoDataFrame, values: pd.DataFrame, norm, cmap, bounds: tuple[float, float, float, float]) -> None:
    merged = geo.merge(values, left_on="NAME", right_on=STATE_COL, how="left")
    merged.plot(
        column=AAMR_COL,
        cmap=cmap,
        norm=norm,
        ax=ax,
        linewidth=0.55,
        edgecolor="#c7c7c7",
        missing_kwds={"color": "#efefef", "edgecolor": "#c7c7c7", "hatch": "///"},
    )
    merged.boundary.plot(ax=ax, linewidth=0.35, edgecolor="#8f8f8f", alpha=0.8)
    minx, miny, maxx, maxy = bounds
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_facecolor("#fbfbfb")
    ax.set_axis_off()
    ax.set_aspect("equal")


def build_state_maps() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    geo = _load_maps()
    map_bounds = tuple(geo.total_bounds)

    prepared = {}
    all_period_values = {}
    for disease_name, path in DATASETS.items():
        raw = _prepare_state_data(path)
        period_frames = []
        for period_label, years in PERIODS:
            agg = _period_aggregate(raw, years)
            agg["Period"] = period_label
            period_frames.append(agg)
        prepared[disease_name] = pd.concat(period_frames, ignore_index=True)
        all_period_values[disease_name] = prepared[disease_name][AAMR_COL].dropna()

    fig, axes = plt.subplots(
        nrows=len(PERIODS),
        ncols=len(DATASETS),
        figsize=(16, 10.5),
        constrained_layout=False,
    )
    colorbar_specs = []

    for col_idx, (disease_name, values) in enumerate(prepared.items()):
        style = DISEASE_STYLES[disease_name]
        cmap = plt.get_cmap(style["cmap"])
        series = all_period_values[disease_name]
        norm = Normalize(vmin=float(series.min()), vmax=float(series.max()))

        for row_idx, (period_label, _) in enumerate(PERIODS):
            ax = axes[row_idx, col_idx]
            period_values = values[values["Period"].eq(period_label)].copy()
            _plot_panel(ax, geo, period_values, norm, cmap, map_bounds)
            if row_idx == 0:
                ax.set_title(disease_name, fontsize=12, fontweight="bold", color=style["title_color"])
            if col_idx == 0:
                ax.text(
                    -0.06,
                    0.5,
                    period_label,
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="right",
                    fontsize=10.5,
                    fontweight="bold",
                )

        colorbar_specs.append(
            {
                "col_idx": col_idx,
                "style": style,
                "mappable": ScalarMappable(norm=norm, cmap=cmap),
            }
        )

    fig.suptitle(
        "State-level age-adjusted mortality rates by disease and period",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.045, right=0.975, top=0.925, bottom=0.155, wspace=0.05, hspace=0.11)

    for spec in colorbar_specs:
        col_idx = spec["col_idx"]
        style = spec["style"]
        sm = spec["mappable"]
        sm.set_array([])
        col_axes = [axes[row_idx, col_idx] for row_idx in range(len(PERIODS))]
        left = min(ax.get_position().x0 for ax in col_axes)
        right = max(ax.get_position().x1 for ax in col_axes)
        width = right - left
        center = left + width / 2
        bar_width = width * 0.52
        cax = fig.add_axes([center - bar_width / 2, 0.13, bar_width, 0.012])
        cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
        cbar.ax.xaxis.set_major_locator(MaxNLocator(nbins=5, prune=None))
        cbar.ax.xaxis.set_major_formatter(StrMethodFormatter("{x:.0f}"))
        cbar.ax.tick_params(
            axis="x",
            labelsize=7.5,
            width=0.6,
            length=2.5,
            pad=1,
            direction="out",
            colors="#333333",
        )
        cbar.outline.set_linewidth(0.5)
        cbar.outline.set_edgecolor("#666666")
        cbar.ax.set_facecolor("white")
        cbar.ax.set_xlabel(style["cbar_label"], fontsize=8, labelpad=2, color="#333333")

    fig.savefig(FIGURE_DIR / "state_maps_by_disease_period.png", dpi=300)
    fig.savefig(FIGURE_DIR / "state_maps_by_disease_period.pdf")
    plt.close(fig)


if __name__ == "__main__":
    build_state_maps()
