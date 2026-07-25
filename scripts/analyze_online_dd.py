from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


Z95 = 1.96


def read_rows(path: Path) -> list[dict[str, float]]:
    with path.open(newline="") as handle:
        return [{key: float(value) for key, value in row.items()} for row in csv.DictReader(handle)]


def aggregate(rows: list[dict[str, float]], min_return: float, max_return: float) -> list[dict[str, float]]:
    scale = max_return - min_return
    if scale <= 0:
        raise ValueError("max_return must be greater than min_return")
    groups: dict[float, list[dict[str, float]]] = {}
    for row in rows:
        groups.setdefault(float(row["num_params"]), []).append(row)
    summary = []
    for num_params, group in sorted(groups.items()):
        test = np.array([row["test_return"] for row in group], dtype=float)
        train = np.array([row["train_return"] for row in group], dtype=float)
        normalized_test = (test - min_return) / scale
        normalized_train = (train - min_return) / scale
        row: dict[str, float] = {
            "num_params": num_params,
            "width": float(group[0].get("width", float("nan"))),
            "run_count": float(len(group)),
            "train_fit_mean": float(normalized_train.mean()),
            "train_fit_std": float(normalized_train.std(ddof=1)) if len(train) > 1 else 0.0,
            "test_return_mean": float(normalized_test.mean()),
            "test_return_std": float(normalized_test.std(ddof=1)) if len(test) > 1 else 0.0,
            "test_return_sem": float(normalized_test.std(ddof=1) / math.sqrt(len(test))) if len(test) > 1 else 0.0,
        }
        row["test_return_ci95"] = Z95 * row["test_return_sem"]
        for name in ("train_entropy", "test_entropy", "train_coverage", "test_coverage"):
            values = np.array([row_data[name] for row_data in group if name in row_data and not math.isnan(row_data[name])])
            if values.size:
                row[f"{name}_mean"] = float(values.mean())
                row[f"{name}_std"] = float(values.std(ddof=1)) if values.size > 1 else 0.0
        summary.append(row)
    return summary


def pooled_uncertainty(a: dict[str, float], b: dict[str, float]) -> float:
    return Z95 * math.sqrt(a["test_return_sem"] ** 2 + b["test_return_sem"] ** 2)


def find_candidates(
    summary: list[dict[str, float]],
    fit_threshold: float,
    practical_effect: float,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for peak in range(1, len(summary) - 2):
        for dip in range(peak + 1, len(summary) - 1):
            for recovery in range(dip + 1, len(summary)):
                left_rise = summary[peak]["test_return_mean"] - summary[peak - 1]["test_return_mean"]
                drop = summary[peak]["test_return_mean"] - summary[dip]["test_return_mean"]
                recovery_gain = summary[recovery]["test_return_mean"] - summary[dip]["test_return_mean"]
                rise_supported = left_rise > pooled_uncertainty(summary[peak - 1], summary[peak])
                drop_supported = drop > pooled_uncertainty(summary[peak], summary[dip])
                recovery_supported = recovery_gain > pooled_uncertainty(summary[dip], summary[recovery])
                persistent_recovery = all(
                    tail["test_return_mean"] >= summary[dip]["test_return_mean"]
                    for tail in summary[recovery:]
                )
                item = {
                    "peak_width": summary[peak]["width"],
                    "dip_width": summary[dip]["width"],
                    "recovery_width": summary[recovery]["width"],
                    "initial_rise": left_rise,
                    "drop": drop,
                    "recovery_gain": recovery_gain,
                    "fit_at_dip": summary[dip]["train_fit_mean"],
                    "rise_supported_by_95ci": rise_supported,
                    "drop_supported_by_95ci": drop_supported,
                    "recovery_supported_by_95ci": recovery_supported,
                    "persistent_recovery": persistent_recovery,
                }
                item["passes"] = bool(
                    item["initial_rise"] >= practical_effect
                    and item["drop"] >= practical_effect
                    and item["recovery_gain"] >= practical_effect
                    and item["fit_at_dip"] >= fit_threshold
                    and rise_supported
                    and drop_supported
                    and recovery_supported
                    and persistent_recovery
                )
                candidates.append(item)
    return candidates


def write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(summary: list[dict[str, float]], path: Path) -> None:
    params = [row["num_params"] for row in summary]
    means = [row["test_return_mean"] for row in summary]
    errors = [row["test_return_ci95"] for row in summary]
    train = [row["train_fit_mean"] for row in summary]
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    axes[0].errorbar(params, train, marker="o", label="train fit")
    axes[0].errorbar(params, means, yerr=errors, marker="o", label="test return")
    axes[0].set_ylabel("Normalized return")
    axes[0].legend()
    axes[1].errorbar(params, means, yerr=errors, marker="o", color="tab:orange")
    axes[1].set_xlabel("Number of policy parameters")
    axes[1].set_ylabel("Test return")
    axes[1].set_xscale("log")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a fixed online-RL capacity sweep.")
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--min-return", type=float, required=True)
    parser.add_argument("--max-return", type=float, required=True)
    parser.add_argument("--fit-threshold", type=float, default=0.95)
    parser.add_argument("--practical-effect", type=float, default=0.10)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(args.metrics)
    summary = aggregate(rows, args.min_return, args.max_return)
    candidates = find_candidates(summary, args.fit_threshold, args.practical_effect)
    write_csv(args.out_dir / "aggregate.csv", summary)
    plot(summary, args.out_dir / "curve.png")
    analysis = {
        "acceptance_criterion": {
            "fit_threshold": args.fit_threshold,
            "practical_effect": args.practical_effect,
            "uncertainty": "each rise, drop, and recovery must exceed pooled 95% normal-approximation uncertainty",
            "persistent_recovery": "all capacities from recovery through the end must stay at or above the dip mean",
        },
        "run_count_per_capacity": {str(row["width"]): int(row["run_count"]) for row in summary},
        "candidate_count": len(candidates),
        "passed": any(item["passes"] for item in candidates),
        "passing_candidates": [item for item in candidates if item["passes"]],
        "all_candidates": candidates,
    }
    with (args.out_dir / "analysis.json").open("w") as handle:
        json.dump(analysis, handle, indent=2)
    print(json.dumps({"passed": analysis["passed"], "passing_candidates": analysis["passing_candidates"]}, indent=2))


if __name__ == "__main__":
    main()
