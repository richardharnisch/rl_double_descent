from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


Z95 = 1.96


def load(paths: list[str]) -> dict[int, list[tuple[float, float]]]:
    values: dict[int, list[tuple[float, float]]] = {}
    for path in paths:
        with open(path, newline="") as handle:
            for row in csv.DictReader(handle):
                episode = int(float(row["episode"]))
                values.setdefault(episode, []).append(
                    (float(row["train_return"]), float(row["test_return"]))
                )
    return values


def summarize(values: dict[int, list[tuple[float, float]]], min_return: float, max_return: float) -> list[dict[str, float]]:
    scale = max_return - min_return
    if scale <= 0:
        raise ValueError("max_return must be greater than min_return")
    rows = []
    for episode in sorted(values):
        train_raw = np.asarray([pair[0] for pair in values[episode]], dtype=float)
        test_raw = np.asarray([pair[1] for pair in values[episode]], dtype=float)
        normalized = (test_raw - min_return) / scale
        train_normalized = (train_raw - min_return) / scale
        sem = float(normalized.std(ddof=1) / math.sqrt(len(normalized))) if len(test_raw) > 1 else 0.0
        rows.append({
            "episode": float(episode),
            "run_count": float(len(test_raw)),
            "train_fit_mean": float(train_normalized.mean()),
            "test_return_mean": float(test_raw.mean()),
            "test_return_std": float(test_raw.std(ddof=1)) if len(test_raw) > 1 else 0.0,
            "test_fit_mean": float(normalized.mean()),
            "test_fit_sem": sem,
            "test_fit_ci95": Z95 * sem,
        })
    return rows


def candidates(rows: list[dict[str, float]], fit_threshold: float, practical_effect: float) -> list[dict[str, object]]:
    found: list[dict[str, object]] = []
    for peak in range(1, len(rows) - 2):
        for dip in range(peak + 1, len(rows) - 1):
            for recovery in range(dip + 1, len(rows)):
                initial_rise = rows[peak]["test_fit_mean"] - rows[peak - 1]["test_fit_mean"]
                drop = rows[peak]["test_fit_mean"] - rows[dip]["test_fit_mean"]
                recovery_gain = rows[recovery]["test_fit_mean"] - rows[dip]["test_fit_mean"]
                rise_unc = Z95 * math.sqrt(rows[peak - 1]["test_fit_sem"] ** 2 + rows[peak]["test_fit_sem"] ** 2)
                drop_unc = Z95 * math.sqrt(rows[peak]["test_fit_sem"] ** 2 + rows[dip]["test_fit_sem"] ** 2)
                recovery_unc = Z95 * math.sqrt(rows[dip]["test_fit_sem"] ** 2 + rows[recovery]["test_fit_sem"] ** 2)
                persistent = all(row["test_fit_mean"] >= rows[dip]["test_fit_mean"] for row in rows[recovery:])
                item = {
                    "peak_episode": rows[peak]["episode"],
                    "dip_episode": rows[dip]["episode"],
                    "recovery_episode": rows[recovery]["episode"],
                    "initial_rise": initial_rise,
                    "drop": drop,
                    "recovery_gain": recovery_gain,
                    "fit_at_dip": rows[dip]["train_fit_mean"],
                    "persistent_recovery": persistent,
                    "passes": bool(
                        initial_rise >= practical_effect
                        and drop >= practical_effect
                        and recovery_gain >= practical_effect
                        and rows[dip]["train_fit_mean"] >= fit_threshold
                        and initial_rise > rise_unc
                        and drop > drop_unc
                        and recovery_gain > recovery_unc
                        and persistent
                    ),
                }
                found.append(item)
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze long-run online RL checkpoints.")
    parser.add_argument("--periodic-glob", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--min-return", type=float, required=True)
    parser.add_argument("--max-return", type=float, required=True)
    parser.add_argument("--fit-threshold", type=float, default=0.95)
    parser.add_argument("--practical-effect", type=float, default=0.10)
    args = parser.parse_args()
    paths = sorted(glob.glob(args.periodic_glob))
    if not paths:
        raise FileNotFoundError(f"No periodic files matched {args.periodic_glob}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = summarize(load(paths), args.min_return, args.max_return)
    with (args.out_dir / "aggregate.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar([row["episode"] for row in rows], [row["test_fit_mean"] for row in rows], yerr=[row["test_fit_ci95"] for row in rows], marker="o")
    ax.set_xlabel("Training episodes")
    ax.set_ylabel("Normalized test return")
    fig.tight_layout()
    fig.savefig(args.out_dir / "curve.png", dpi=160)
    plt.close(fig)
    found = candidates(rows, args.fit_threshold, args.practical_effect)
    result = {
        "acceptance_criterion": {
            "fit_threshold": args.fit_threshold,
            "practical_effect": args.practical_effect,
            "uncertainty": "each rise, drop, and recovery must exceed pooled 95% normal-approximation uncertainty",
            "persistent_recovery": "all later checkpoints stay at or above the dip mean",
        },
        "source_files": paths,
        "candidate_count": len(found),
        "passed": any(item["passes"] for item in found),
        "passing_candidates": [item for item in found if item["passes"]],
        "all_candidates": found,
    }
    with (args.out_dir / "analysis.json").open("w") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps({"passed": result["passed"], "passing_candidates": result["passing_candidates"]}, indent=2))


if __name__ == "__main__":
    main()
