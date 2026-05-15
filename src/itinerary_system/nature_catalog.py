"""Nature-aware POI features, scoring, preview artifacts, and metrics."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from .config import TripConfig
from .region_scenarios import get_nature_region_definitions, get_scenario_definition, scenario_summary_rows
from .request_schema import INTEREST_AXES, normalize_interest_weights, preset_to_interest_weights

NATURE_POI_COLUMNS = [
    "is_nature",
    "is_national_park",
    "is_state_park",
    "is_protected_area",
    "is_scenic_viewpoint",
    "is_hiking",
    "nature_score",
    "city_score",
    "culture_score",
    "history_score",
    "scenic_score",
    "hiking_score",
    "outdoor_intensity",
    "weather_sensitivity",
    "seasonality_risk",
    "park_type",
    "nature_region",
    "interest_fit",
    "park_bonus",
    "interest_adjusted_value",
    "interest_delta",
    "reason_selected",
]

NUMERIC_NATURE_COLUMNS = {
    "nature_score",
    "city_score",
    "culture_score",
    "history_score",
    "scenic_score",
    "hiking_score",
    "outdoor_intensity",
    "weather_sensitivity",
    "seasonality_risk",
    "interest_fit",
    "park_bonus",
    "interest_adjusted_value",
    "interest_delta",
}

BOOL_NATURE_COLUMNS = {
    "is_nature",
    "is_national_park",
    "is_state_park",
    "is_protected_area",
    "is_scenic_viewpoint",
    "is_hiking",
}

TEXT_NATURE_COLUMNS = {"park_type", "nature_region", "reason_selected"}

NATURE_KEYWORDS = "park|beach|trail|hiking|viewpoint|scenic|reserve|forest|canyon|falls|waterfall|mountain|peak|coast|lake|river|wildlife"
CITY_KEYWORDS = "downtown|market|shopping|restaurant|strip|theatre|theater|nightlife|pier|wharf|city|urban"
CULTURE_KEYWORDS = "museum|gallery|arts|theatre|theater|aquarium|zoo|observatory|garden|campus|library"
HISTORY_KEYWORDS = "historic|history|heritage|monument|mission|fort|landmark|memorial|archaeological"


def interest_enabled(config: TripConfig) -> bool:
    return bool(config.get("interest", "enabled", False))


def interest_value_column(config: TripConfig) -> str:
    return "interest_adjusted_value" if interest_enabled(config) else "final_poi_value"


def interest_weights_from_config(config: TripConfig) -> dict[str, float]:
    interest_section = config.section("interest")
    mode = str(interest_section.get("mode", "custom"))
    presets = interest_section.get("presets")
    if mode != "custom":
        return preset_to_interest_weights(mode, presets)
    if isinstance(interest_section.get("weights"), dict):
        return normalize_interest_weights(interest_section["weights"])
    trip_profile = str(config.get("trip", "interest_profile", "balanced_interest"))
    return preset_to_interest_weights(trip_profile, presets)


def _numeric(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column in frame.columns:
        return pd.to_numeric(frame[column], errors="coerce").fillna(default).astype(float)
    return pd.Series(default, index=frame.index, dtype=float)


def _bool_from_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column in frame.columns:
        text = frame[column].astype(str).str.lower()
        return text.isin({"true", "1", "yes", "y"})
    return pd.Series(False, index=frame.index)


def _contains(frame: pd.DataFrame, pattern: str) -> pd.Series:
    text_parts = []
    for column in ["name", "category", "source_list", "park_type", "nature_region", "osm_tags"]:
        if column in frame.columns:
            text_parts.append(frame[column].fillna("").astype(str))
    if not text_parts:
        return pd.Series(False, index=frame.index)
    text = text_parts[0]
    for part in text_parts[1:]:
        text = text + " " + part
    return text.str.contains(pattern, case=False, regex=True, na=False)


def ensure_nature_columns(poi_df: pd.DataFrame) -> pd.DataFrame:
    """Backfill nature/interest columns so old CSVs remain loadable."""
    output = poi_df.copy()
    for column in NATURE_POI_COLUMNS:
        if column in output.columns:
            continue
        if column in BOOL_NATURE_COLUMNS:
            output[column] = False
        elif column in TEXT_NATURE_COLUMNS:
            output[column] = ""
        else:
            output[column] = 0.0
    for column in BOOL_NATURE_COLUMNS:
        output[column] = (
            _bool_from_column(output, column)
            if output[column].dtype == object
            else output[column].fillna(False).astype(bool)
        )
    for column in NUMERIC_NATURE_COLUMNS:
        output[column] = pd.to_numeric(output[column], errors="coerce").fillna(0.0).astype(float)
    for column in TEXT_NATURE_COLUMNS:
        output[column] = output[column].fillna("").astype(str)
    return output


def infer_interest_attributes(poi_df: pd.DataFrame, config: TripConfig | None = None) -> pd.DataFrame:
    """Infer POI interest vectors and nature flags from open tags/categories."""
    output = ensure_nature_columns(poi_df)
    if output.empty:
        return output

    nature_like = _contains(output, NATURE_KEYWORDS)
    city_like = _contains(output, CITY_KEYWORDS)
    culture_like = _contains(output, CULTURE_KEYWORDS)
    history_like = _contains(output, HISTORY_KEYWORDS)
    national_like = _contains(output, "national_park|national park|nps")
    state_like = _contains(output, "state_park|state park|state natural reserve")
    protected_like = _contains(output, "protected_area|protected area|reserve|conservation|preserve")
    viewpoint_like = _contains(output, "viewpoint|scenic|overlook|vista|bridge")
    hiking_like = _contains(output, "hiking|trail|trailhead|mountain|peak|canyon")

    output["is_nature"] = output["is_nature"] | nature_like
    output["is_national_park"] = output["is_national_park"] | national_like
    output["is_state_park"] = output["is_state_park"] | state_like
    output["is_protected_area"] = (
        output["is_protected_area"] | protected_like | output["is_national_park"] | output["is_state_park"]
    )
    output["is_scenic_viewpoint"] = output["is_scenic_viewpoint"] | viewpoint_like
    output["is_hiking"] = output["is_hiking"] | hiking_like

    output["nature_score"] = np.maximum(
        _numeric(output, "nature_score"),
        (
            0.35 * output["is_nature"].astype(float)
            + 0.25 * output["is_protected_area"].astype(float)
            + 0.20 * output["is_scenic_viewpoint"].astype(float)
            + 0.20 * output["is_hiking"].astype(float)
        ).clip(0.0, 1.0),
    )
    output["city_score"] = np.maximum(
        _numeric(output, "city_score"),
        (0.65 * city_like.astype(float) + 0.25 * (~nature_like).astype(float)).clip(0.0, 1.0),
    )
    output["culture_score"] = np.maximum(
        _numeric(output, "culture_score"),
        (0.85 * culture_like.astype(float) + 0.20 * history_like.astype(float)).clip(0.0, 1.0),
    )
    output["history_score"] = np.maximum(
        _numeric(output, "history_score"), (0.90 * history_like.astype(float)).clip(0.0, 1.0)
    )
    output["scenic_score"] = np.maximum(
        _numeric(output, "scenic_score"),
        (0.70 * output["is_scenic_viewpoint"].astype(float) + 0.30 * output["nature_score"]).clip(0.0, 1.0),
    )
    output["hiking_score"] = np.maximum(
        _numeric(output, "hiking_score"),
        (0.85 * output["is_hiking"].astype(float) + 0.15 * output["nature_score"]).clip(0.0, 1.0),
    )
    output["outdoor_intensity"] = np.maximum(
        _numeric(output, "outdoor_intensity"),
        (0.45 * output["nature_score"] + 0.35 * output["hiking_score"] + 0.20 * output["scenic_score"]).clip(0.0, 1.0),
    )
    output["weather_sensitivity"] = np.maximum(
        _numeric(output, "weather_sensitivity"), (0.25 + 0.65 * output["outdoor_intensity"]).clip(0.0, 1.0)
    )
    output["seasonality_risk"] = np.maximum(
        _numeric(output, "seasonality_risk"),
        (0.15 * output["is_nature"].astype(float) + 0.25 * output["is_hiking"].astype(float)).clip(0.0, 1.0),
    )

    output.loc[output["is_national_park"] & output["park_type"].eq(""), "park_type"] = "national_park"
    output.loc[output["is_state_park"] & output["park_type"].eq(""), "park_type"] = "state_park"
    output.loc[output["is_protected_area"] & output["park_type"].eq(""), "park_type"] = "protected_area"
    output.loc[output["nature_region"].eq("") & output["is_nature"], "nature_region"] = output.loc[
        output["nature_region"].eq("") & output["is_nature"], "city"
    ].astype(str)
    return output


# Math contract:
# interest_fit_i = p^T a_i
# park_bonus_i = 1.00*NP_i + 0.60*SP_i + 0.45*PA_i + 0.25*SV_i
# u_i_interest = final_poi_value_i
#              + lambda_fit * interest_fit_i
#              + lambda_park * park_bonus_i
#              - lambda_weather * weather_sensitivity_i * weather_risk_i
#              - lambda_season * seasonality_risk_i
#              - lambda_detour * detour_minutes_i
def compute_interest_adjusted_values(poi_df: pd.DataFrame, config: TripConfig) -> pd.DataFrame:
    output = infer_interest_attributes(poi_df, config)
    if output.empty:
        return output

    base_value = _numeric(output, "final_poi_value")
    weights = interest_weights_from_config(config)
    output["interest_fit"] = sum(float(weights[axis]) * _numeric(output, f"{axis}_score") for axis in INTEREST_AXES)
    output["park_bonus"] = (
        float(config.get("nature", "national_park_bonus", 0.80)) * output["is_national_park"].astype(float)
        + float(config.get("nature", "state_park_bonus", 0.45)) * output["is_state_park"].astype(float)
        + float(config.get("nature", "protected_area_bonus", 0.35)) * output["is_protected_area"].astype(float)
        + float(config.get("nature", "scenic_region_bonus", 0.30)) * output["is_scenic_viewpoint"].astype(float)
    )

    if not interest_enabled(config):
        output["interest_adjusted_value"] = base_value
        output["interest_delta"] = 0.0
        return output

    lambda_fit = float(config.get("interest", "lambda_fit", 0.65))
    lambda_park = float(config.get("interest", "lambda_park", 0.35))
    lambda_weather = float(config.get("interest", "lambda_weather", 0.30))
    lambda_season = float(config.get("interest", "lambda_season", 0.20))
    lambda_detour = float(config.get("interest", "lambda_detour", 0.006))
    weather_risk = _numeric(output, "weather_risk", 0.15).clip(0.0, 1.0)
    output["interest_adjusted_value"] = (
        base_value
        + lambda_fit * output["interest_fit"]
        + lambda_park * output["park_bonus"]
        - lambda_weather * _numeric(output, "weather_sensitivity") * weather_risk
        - lambda_season * _numeric(output, "seasonality_risk")
        - lambda_detour * _numeric(output, "detour_minutes")
    ).clip(lower=0.0)
    output["interest_delta"] = output["interest_adjusted_value"] - base_value
    output["reason_selected"] = output.apply(_reason_text, axis=1)
    return output


def _reason_text(row: pd.Series) -> str:
    reasons = []
    if bool(row.get("is_national_park", False)):
        reasons.append("national park priority")
    elif bool(row.get("is_state_park", False)):
        reasons.append("state park / reserve")
    elif bool(row.get("is_protected_area", False)):
        reasons.append("protected nature area")
    if float(row.get("scenic_score", 0.0) or 0.0) >= 0.6:
        reasons.append("scenic value")
    if float(row.get("hiking_score", 0.0) or 0.0) >= 0.6:
        reasons.append("hiking fit")
    if float(row.get("interest_delta", 0.0) or 0.0) > 0:
        reasons.append("interest bar boost")
    return "; ".join(reasons)


def curated_nature_fallback_pois(scenario_id: str | None = None) -> pd.DataFrame:
    scenario = get_scenario_definition(scenario_id)
    rows = []
    for row in scenario.curated_seed_pois:
        rows.append(
            {
                "yelp_rating": 0.0,
                "yelp_review_count": 0,
                "social_score": 0.0,
                "social_must_go": False,
                "must_go_weight": 0.0,
                "corridor_fit": 0.0,
                "detour_minutes": 0.0,
                "data_confidence": 0.35,
                "data_uncertainty": 0.65,
                "final_poi_value": float(row.get("source_score", 6.0)) / 10.0,
                "social_reason": "",
                **row,
            }
        )
    return infer_interest_attributes(pd.DataFrame(rows)) if rows else pd.DataFrame()


def _scenario_state_codes(scenario_id: str) -> str:
    if scenario_id.startswith("california"):
        return "CA"
    if scenario_id == "las_vegas_national_parks":
        return "NV,UT,AZ,CA"
    if scenario_id == "new_york_city_plus_nature":
        return "NY"
    return "CA"


def _parse_float(value: Any, default: float = np.nan) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _nps_cache_path(output_dir: Path, scenario_id: str) -> Path:
    cache_dir = output_dir.parent / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"nps_parks_{scenario_id}.json"


def _fetch_nps_parks(api_key: str, scenario_id: str) -> list[dict[str, Any]]:
    response = requests.get(
        "https://developer.nps.gov/api/v1/parks",
        params={
            "api_key": api_key,
            "stateCode": _scenario_state_codes(scenario_id),
            "limit": 500,
            "fields": "addresses,entranceFees,operatingHours,topics,activities",
        },
        timeout=20,
    )
    response.raise_for_status()
    payload = response.json()
    data = payload.get("data", [])
    return data if isinstance(data, list) else []


def _match_region_name(park_name: str, scenario_id: str) -> str:
    lowered = park_name.lower()
    for region in get_nature_region_definitions(scenario_id):
        if (
            region.name.lower().replace(" national park", "") in lowered
            or region.region_id.replace("_", " ") in lowered
        ):
            return region.name
    return park_name


def _nps_records_to_pois(records: list[dict[str, Any]], scenario_id: str) -> pd.DataFrame:
    rows = []
    for record in records:
        lat = _parse_float(record.get("latitude"))
        lon = _parse_float(record.get("longitude"))
        if np.isnan(lat) or np.isnan(lon):
            continue
        designation = str(record.get("designation", "national park") or "national park")
        full_name = str(record.get("fullName", record.get("name", "")) or "")
        activities = record.get("activities") or []
        topics = record.get("topics") or []
        fees = record.get("entranceFees") or []
        hours = record.get("operatingHours") or []
        rows.append(
            {
                "name": full_name,
                "city": _match_region_name(full_name, scenario_id),
                "latitude": lat,
                "longitude": lon,
                "category": designation,
                "park_type": designation.lower().replace(" ", "_"),
                "nature_region": _match_region_name(full_name, scenario_id),
                "source_list": "nps_api",
                "source_score": 9.6
                if "national park" in designation.lower() or "national park" in full_name.lower()
                else 8.8,
                "final_poi_value": 0.96
                if "national park" in designation.lower() or "national park" in full_name.lower()
                else 0.88,
                "data_confidence": 0.85,
                "data_uncertainty": 0.15,
                "nps_park_code": record.get("parkCode", ""),
                "nps_designation": designation,
                "nps_states": record.get("states", ""),
                "nps_activities": " | ".join(
                    str(item.get("name", "")) for item in activities if isinstance(item, dict)
                ),
                "nps_topics": " | ".join(str(item.get("name", "")) for item in topics if isinstance(item, dict)),
                "nps_entrance_fees": json.dumps(fees),
                "nps_operating_hours": json.dumps(hours),
                "nps_url": record.get("url", ""),
                "nps_description": record.get("description", ""),
                "nps_weather_info": record.get("weatherInfo", ""),
                "weather_sensitivity": 0.70,
                "seasonality_risk": 0.30,
            }
        )
    return infer_interest_attributes(pd.DataFrame(rows)) if rows else pd.DataFrame()


def load_nps_or_curated_nature_pois(output_dir: str | Path, config: TripConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load optional NPS data when configured, otherwise return curated seeds."""
    output_dir = Path(output_dir)
    scenario_id = str(config.get("trip", "scenario", "california_coast"))
    audit = {
        "audit_type": "nps_or_curated_nature",
        "scenario": scenario_id,
        "used_nps_api": False,
        "status": "nps_disabled_curated_fallback",
    }
    seeds = (
        curated_nature_fallback_pois(scenario_id)
        if bool(config.get("nature", "use_curated_nature_fallback", True))
        else pd.DataFrame()
    )
    if not bool(config.get("nature", "use_nps_api", False)):
        return seeds, pd.DataFrame([audit | {"rows": int(len(seeds)), "cache_path": ""}])

    env_name = str(config.get("nature", "nps_api_key_env", "NPS_API_KEY"))
    api_key = os.environ.get(env_name)
    cache_path = _nps_cache_path(output_dir, scenario_id)
    if not api_key:
        audit["status"] = f"missing_{env_name}_curated_fallback"
        return seeds, pd.DataFrame([audit | {"rows": int(len(seeds)), "cache_path": str(cache_path)}])

    try:
        records = _fetch_nps_parks(api_key, scenario_id)
        cache_path.write_text(json.dumps({"data": records}, indent=2), encoding="utf-8")
        pois = _nps_records_to_pois(records, scenario_id)
        if pois.empty and not seeds.empty:
            audit["status"] = "nps_api_error_curated_fallback"
            return seeds, pd.DataFrame([audit | {"rows": int(len(seeds)), "cache_path": str(cache_path)}])
        audit["status"] = "nps_api_success"
        audit["used_nps_api"] = True
        return pois, pd.DataFrame([audit | {"rows": int(len(pois)), "cache_path": str(cache_path)}])
    except Exception as exc:
        if cache_path.exists():
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                records = cached.get("data", []) if isinstance(cached, dict) else cached
                pois = _nps_records_to_pois(records if isinstance(records, list) else [], scenario_id)
                if not pois.empty:
                    audit["status"] = "nps_api_error_cache_fallback"
                    audit["error"] = type(exc).__name__
                    return pois, pd.DataFrame([audit | {"rows": int(len(pois)), "cache_path": str(cache_path)}])
            except Exception:
                pass
        audit["status"] = "nps_api_error_curated_fallback"
        audit["error"] = type(exc).__name__
        return seeds, pd.DataFrame([audit | {"rows": int(len(seeds)), "cache_path": str(cache_path)}])


def route_interest_metrics(route_df: pd.DataFrame, config: TripConfig) -> dict[str, float]:
    frame = (
        compute_interest_adjusted_values(route_df, config) if not route_df.empty else ensure_nature_columns(route_df)
    )
    if frame.empty:
        return {
            "interest_adjusted_utility": 0.0,
            "interest_delta": 0.0,
            "balance_score": 0.0,
            "balance_error": 1.0,
            "outdoor_weather_risk": 0.0,
            "seasonality_risk": 0.0,
        }
    weights = interest_weights_from_config(config)
    route_mix = {axis: float(_numeric(frame, f"{axis}_score").mean()) for axis in INTEREST_AXES}
    balance_error = float(sum(abs(route_mix[axis] - weights[axis]) for axis in INTEREST_AXES))
    weather_risk = _numeric(frame, "weather_risk", 0.15).clip(0.0, 1.0)
    return {
        "interest_adjusted_utility": float(_numeric(frame, "interest_adjusted_value").sum()),
        "interest_delta": float(_numeric(frame, "interest_delta").sum()),
        "nature_score": float(_numeric(frame, "nature_score").sum()),
        "scenic_score": float(_numeric(frame, "scenic_score").sum()),
        "national_parks_selected": int(frame["is_national_park"].astype(bool).sum()),
        "protected_parks_selected": int(
            (frame["is_state_park"].astype(bool) | frame["is_protected_area"].astype(bool)).sum()
        ),
        "outdoor_weather_risk": float((_numeric(frame, "weather_sensitivity") * weather_risk).sum()),
        "seasonality_risk": float(_numeric(frame, "seasonality_risk").sum()),
        "balance_error": balance_error,
        "balance_score": max(0.0, 1.0 - balance_error),
        **{f"route_mix_{axis}": route_mix[axis] for axis in INTEREST_AXES},
    }


def build_interest_bar_preview(enriched_df: pd.DataFrame, config: TripConfig) -> dict[str, Any]:
    frame = compute_interest_adjusted_values(enriched_df, config)
    weights = interest_weights_from_config(config)
    top_n = int(config.get("interest", "preview_top_n", 12))
    if frame.empty:
        rows = []
        route_mix = {axis: 0.0 for axis in INTEREST_AXES}
    else:
        preview = (
            frame.sort_values(["interest_delta", "interest_adjusted_value"], ascending=False).head(max(1, top_n)).copy()
        )
        route_mix = {axis: float(_numeric(preview, f"{axis}_score").mean()) for axis in INTEREST_AXES}
        rows = []
        for row in preview.itertuples(index=False):
            rows.append(
                {
                    "name": str(getattr(row, "name", "")),
                    "city": str(getattr(row, "city", "")),
                    "lat": float(getattr(row, "latitude", np.nan)),
                    "lon": float(getattr(row, "longitude", np.nan)),
                    "final_poi_value": float(getattr(row, "final_poi_value", 0.0) or 0.0),
                    "nature": float(getattr(row, "nature_score", 0.0) or 0.0),
                    "city_axis": float(getattr(row, "city_score", 0.0) or 0.0),
                    "culture": float(getattr(row, "culture_score", 0.0) or 0.0),
                    "history": float(getattr(row, "history_score", 0.0) or 0.0),
                    "park_bonus": float(getattr(row, "park_bonus", 0.0) or 0.0),
                    "weather_sensitivity": float(getattr(row, "weather_sensitivity", 0.0) or 0.0),
                    "weather_risk": float(getattr(row, "weather_risk", 0.15) or 0.15),
                    "seasonality_risk": float(getattr(row, "seasonality_risk", 0.0) or 0.0),
                    "detour_minutes": float(getattr(row, "detour_minutes", 0.0) or 0.0),
                    "source_list": str(getattr(row, "source_list", "")),
                    "park_type": str(getattr(row, "park_type", "")),
                    "nature_region": str(getattr(row, "nature_region", "")),
                    "interest_fit": float(getattr(row, "interest_fit", 0.0) or 0.0),
                    "interest_adjusted_value": float(getattr(row, "interest_adjusted_value", 0.0) or 0.0),
                    "interest_delta": float(getattr(row, "interest_delta", 0.0) or 0.0),
                }
            )
    return {
        "enabled": interest_enabled(config),
        "weights": weights,
        "lambdas": {
            "lambda_fit": float(config.get("interest", "lambda_fit", 0.65)),
            "lambda_park": float(config.get("interest", "lambda_park", 0.35)),
            "lambda_weather": float(config.get("interest", "lambda_weather", 0.30)),
            "lambda_season": float(config.get("interest", "lambda_season", 0.20)),
            "lambda_detour": float(config.get("interest", "lambda_detour", 0.006)),
        },
        "bar": [
            {"axis": axis, "weight": weights[axis], "percent": round(weights[axis] * 100.0, 1)}
            for axis in INTEREST_AXES
        ],
        "top_boosted_pois": rows,
        "route_mix": route_mix,
    }


def write_interest_catalog_artifacts(enriched_df: pd.DataFrame, output_dir: str | Path, config: TripConfig) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    preview = build_interest_bar_preview(enriched_df, config)
    (output_dir / "production_interest_bar_preview.json").write_text(json.dumps(preview, indent=2), encoding="utf-8")

    frame = compute_interest_adjusted_values(enriched_df, config)
    metrics = route_interest_metrics(frame.head(max(1, int(config.trip_days) * 3)), config)
    metrics.update(
        {
            "metric_scope": "top_catalog_preview",
            "total_utility": float(_numeric(frame, "final_poi_value").head(max(1, int(config.trip_days) * 3)).sum())
            if not frame.empty
            else 0.0,
            "candidate_rows": int(len(frame)),
            "interest_enabled": interest_enabled(config),
        }
    )
    pd.DataFrame([metrics]).to_csv(output_dir / "production_nature_metric_summary.csv", index=False)

    comparison_rows = []
    for preset in config.get("interest", "presets", {}).keys():
        preset_config = TripConfig(
            data={
                **config.to_dict(),
                "interest": {
                    **config.section("interest"),
                    "enabled": True,
                    "weights": preset_to_interest_weights(preset, config.get("interest", "presets", {})),
                    "mode": preset,
                },
            },
            source_path=config.source_path,
        )
        preset_frame = compute_interest_adjusted_values(enriched_df, preset_config)
        top = preset_frame.sort_values("interest_adjusted_value", ascending=False).head(
            max(1, int(config.trip_days) * 3)
        )
        comparison_rows.append(
            {
                "interest_profile": preset,
                "top_interest_adjusted_utility": float(_numeric(top, "interest_adjusted_value").sum())
                if not top.empty
                else 0.0,
                "top_interest_delta": float(_numeric(top, "interest_delta").sum()) if not top.empty else 0.0,
                "top_nature_score": float(_numeric(top, "nature_score").sum()) if not top.empty else 0.0,
                "top_scenic_score": float(_numeric(top, "scenic_score").sum()) if not top.empty else 0.0,
            }
        )
    pd.DataFrame(comparison_rows).to_csv(output_dir / "production_interest_profile_comparison.csv", index=False)
    pd.DataFrame(scenario_summary_rows()).to_csv(output_dir / "production_scenario_comparison.csv", index=False)


def write_nature_route_artifacts(route_df: pd.DataFrame, output_dir: str | Path, config: TripConfig) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = (
        compute_interest_adjusted_values(route_df, config) if not route_df.empty else ensure_nature_columns(route_df)
    )
    columns = [
        "method",
        "profile",
        "trip_days",
        "day",
        "stop_order",
        "attraction_name",
        "city",
        "category",
        *NATURE_POI_COLUMNS,
    ]
    for column in columns:
        if column not in frame.columns:
            frame[column] = ""
    frame[[column for column in columns if column in frame.columns]].to_csv(
        output_dir / "production_nature_route_stops.csv", index=False
    )
    metrics = route_interest_metrics(frame, config)
    metrics.update({"metric_scope": "selected_route_stops", "selected_attractions": int(len(frame))})
    if "why_not_selected" in frame.columns:
        metrics["why_not_selected"] = " | ".join(frame["why_not_selected"].dropna().astype(str).unique()[:8])
    pd.DataFrame([metrics]).to_csv(output_dir / "production_nature_metric_summary.csv", index=False)


# TODO contextual bandit over interest bars:
# context_t = [scenario, trip_days, weather, budget, current p]
# reward_t = route_utility_t - travel_penalty_t - weather_penalty_t + balance_score_t
# theta <- update(theta | context_t, p_t, reward_t)
#
# TODO pairwise learning-to-rank:
# P(route A preferred to B) = sigmoid(f_theta(A) - f_theta(B))
# loss = -log P(A > B)
#
# TODO nonlinear POI utility:
# u_i = f_theta(base_features_i, interest_vector_i, weather_i, scenario_i)
# Use only after labeled preference data exists.
#
# TODO Pareto route frontier:
# maximize utility
# subject to weather_risk <= epsilon_w
#            detour <= epsilon_d
#            balance_error <= epsilon_b
