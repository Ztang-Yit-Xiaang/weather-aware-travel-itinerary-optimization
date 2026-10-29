"""Run immutable synthetic response-aware repair evidence; never a publication run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from itinerary_system.benchmark.response_aware import SCENARIOS, compare_matched_pool, run_toy_scenario  # noqa: E402
from itinerary_system.research_artifacts import stable_content_hash  # noqa: E402


def source_hashes() -> dict[str, str]:
    files = sorted((ROOT / "src" / "itinerary_system").rglob("*.py"))
    files += [
        Path(__file__),
        ROOT / "pyproject.toml",
        ROOT / "tests/repair/test_response_aware.py",
        ROOT / "tests/repair/test_plan_consistency.py",
    ]
    return {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}


def semantic_result(record: dict) -> dict:
    """Stable outcome fingerprint excludes clocks, generated IDs and runtime."""
    return {
        method: {
            "sequence": row["plan"]["sequence"],
            "disruption": row["disruption"],
            "travel_minutes": row["travel_minutes"],
            "utility_retained": row["utility"],
            "modeled_failures": row["evaluation"]["modeled_failures"],
            "canonical_status": row["evaluation"]["certificate"]["evaluation_status"],
        }
        if row
        else None
        for method, row in record["strategies"].items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root", required=True, type=Path, help="A fresh output directory; existing evidence is refused."
    )
    args = parser.parse_args()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=False)
    frozen_sources = source_hashes()
    (output / "source_hashes.json").write_text(json.dumps(frozen_sources, indent=2), encoding="utf-8")
    summaries, results, perturbations = [], {}, []
    for scenario in SCENARIOS:
        record = run_toy_scenario(scenario, output / scenario.scenario_id)
        results[scenario.scenario_id] = record
        (output / f"{scenario.scenario_id}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        for method, row in record["strategies"].items():
            summaries.append(
                {
                    "scenario": scenario.scenario_id,
                    "method": method,
                    "model_feasible": bool(row and row["evaluation"]["eligible"]),
                    "weighted_diff": row["disruption"]["weighted_edit_cost"] if row else None,
                    "secondary": row["disruption"]["secondary_cost"] if row else None,
                    "outside": row["disruption"]["outside_neighborhood_cost"] if row else None,
                    "travel_minutes": row["travel_minutes"] if row else None,
                    "utility_retained": row["utility"] if row else None,
                    "modeled_hard_failures": len(row["evaluation"]["modeled_failures"]) if row else None,
                    "canonical_status": row["evaluation"]["certificate"]["evaluation_status"] if row else "NO_PLAN",
                    "publication_eligible": False,
                    "runtime_seconds": record["runtime_seconds"][
                        method if method != "response_aware" else "response_aware_total"
                    ],
                }
            )
        variants = []
        for delta in (-0.1, 0.1):
            variant = replace(
                scenario,
                scenario_id=f"{scenario.scenario_id}:{delta:+}",
                improved_edge_minutes=scenario.improved_edge_minutes + delta,
            )
            probe = run_toy_scenario(variant, output / "perturbations" / variant.scenario_id.replace(":", "_"))
            path = output / "perturbations" / f"{variant.scenario_id.replace(':', '_')}.json"
            path.write_text(json.dumps(probe, indent=2), encoding="utf-8")
            variants.append(probe)
        for method, baseline in record["strategies"].items():
            base_sequence = baseline["plan"]["sequence"] if baseline else None
            sequences = [
                variant["strategies"][method]["plan"]["sequence"] if variant["strategies"][method] else None
                for variant in variants
            ]
            perturbations.append(
                {
                    "scenario": scenario.scenario_id,
                    "method": method,
                    "edge_deltas_minutes": [-0.1, 0.1],
                    "base_sequence": base_sequence,
                    "perturbed_sequences": sequences,
                    "changed_from_base_count": sum(sequence != base_sequence for sequence in sequences),
                    "feasibility_switch_count": sum(
                        (sequence is None) != (base_sequence is None) for sequence in sequences
                    ),
                    "both_infeasible": base_sequence is None and all(sequence is None for sequence in sequences),
                }
            )
    with (output / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    (output / "stability.json").write_text(json.dumps(perturbations, indent=2), encoding="utf-8")
    matched = compare_matched_pool()
    (output / "matched_pool.json").write_text(json.dumps(matched, indent=2), encoding="utf-8")
    if source_hashes() != frozen_sources:
        raise RuntimeError("source changed during experiment; evidence is not accepted")
    manifest = {
        "schema_version": "response-aware-synthetic-benchmark-v1",
        "python": platform.python_version(),
        "evidence_mode": "synthetic_graph_only",
        "publication_eligible": False,
        "base_scenario_count": len(SCENARIOS),
        "perturbation_scenario_count": len(SCENARIOS) * 2,
        "strategy_rows": len(SCENARIOS) * 3 * 3,
        "source_unchanged": True,
        "result_fingerprint": stable_content_hash({key: semantic_result(value) for key, value in results.items()}),
        "matched_pool_fingerprint": stable_content_hash(
            {
                case: {
                    basis: {"sequence": row["sequence"], "cost": row["diff"]["weighted_edit_cost"]}
                    for basis, row in record["outcomes"].items()
                }
                for case, record in matched["cases"].items()
            }
        ),
        "inputs": {key: value["input_hash"] for key, value in results.items()},
        "output_sha256": {
            path.relative_to(output).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(output.rglob("*"))
            if path.is_file()
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        json.dumps({key: value for key, value in manifest.items() if key not in {"inputs", "output_sha256"}}, indent=2)
    )


if __name__ == "__main__":
    main()
