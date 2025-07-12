#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_bwt_old.py

Compute the backward transfer (BWT) of the *old* method.

Directory layout expected:

<dataset_path>/
    <seed_1>/
        task_0.hdf5
        task_1.hdf5
        ...
    <seed_2>/
        task_0.hdf5
        ...

Each task_i.hdf5 must contain:
    • "test_predictions"            – ndarray (N,) with model predictions
    • "test_labels_for_predictions" – ndarray (N,) with ground-truth labels
"""

import argparse
import os
import sys
from typing import List

import h5py
import numpy as np
import pandas as pd

print("Modules loaded")


# ----------------------------------------------------------------------------- #
#                              Command-line interface                           #
# ----------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute backward transfer (BWT) for the old method."
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Root path that contains one sub-folder per seed.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        required=True,
        help="Seed sub-folders to include in the analysis.",
    )
    parser.add_argument(
        "--num_tasks",
        type=int,
        required=True,
        help="Number of continual-learning tasks.",
    )
    parser.add_argument(
        "--num_classes_task0",
        type=int,
        default=50,
        help="Number of classes introduced in task 0 (default: 50).",
    )
    parser.add_argument(
        "--num_classes_per_task",
        type=int,
        default=10,
        help="Number of new classes introduced by each subsequent task "
        "(default: 10).",
    )
    parser.add_argument(
        "--file_prefix",
        type=str,
        default="task_",
        help='File prefix for task files (default: "task_").',
    )
    parser.add_argument(
        "--file_suffix",
        type=str,
        default=".hdf5",
        help='File suffix for task files (default: ".hdf5").',
    )
    return parser.parse_args()


# ----------------------------------------------------------------------------- #
#                              Helper functions                                 #
# ----------------------------------------------------------------------------- #
def total_num_classes(t: int, c0: int, ct: int) -> int:
    """Total number of classes after *t* tasks."""
    return c0 + (t - 1) * ct


def class_ranges(t: int, c0: int, ct: int) -> List[range]:
    """Return a list of class ranges introduced by each task."""
    ranges: List[range] = []
    for task_id in range(t):
        if task_id == 0:
            ranges.append(range(0, c0))
        else:
            start = c0 + (task_id - 1) * ct
            ranges.append(range(start, start + ct))
    return ranges


def load_preds_labels(path: str):
    """Load predictions and labels from a single HDF5 file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    with h5py.File(path, "r") as f:
        preds = np.asarray(f["test_predictions"])
        labels = np.asarray(f["test_labels_for_predictions"])
    return preds, labels


def per_class_accuracy(
    preds: np.ndarray, labels: np.ndarray, num_classes: int
) -> np.ndarray:
    """Return an array of per-class accuracies (NaN when the class is absent)."""
    acc = np.full(num_classes, np.nan, dtype=np.float32)
    for cls in np.unique(labels):
        mask = labels == cls
        acc[cls] = np.sum(preds[mask] == cls) / mask.sum()
    return acc


def compute_bwt(acc_matrix: np.ndarray, ranges: List[range]) -> float:
    """
    BWT = 1/(T-1) Σ_{t=0}^{T-2} ( acc_T-1(t) – acc_t(t) ).
    """
    final_acc = acc_matrix[-1]
    deltas = [
        np.nanmean(final_acc[r]) - np.nanmean(acc_matrix[t][r])
        for t, r in enumerate(ranges[:-1])
    ]
    return float(np.nanmean(deltas) * 100.0)


def process_seed(
    seed_dir: str,
    args: argparse.Namespace,
    num_classes: int,
    ranges: List[range],
) -> float:
    """Build the [T × C] accuracy matrix for one seed and return its BWT."""
    acc_matrix = np.full((args.num_tasks, num_classes), np.nan, dtype=np.float32)

    for task_id in range(args.num_tasks):
        fname = f"{args.file_prefix}{task_id}{args.file_suffix}"
        path = os.path.join(args.dataset_path, seed_dir, fname)
        preds, labels = load_preds_labels(path)
        acc_matrix[task_id] = per_class_accuracy(preds, labels, num_classes)

    return compute_bwt(acc_matrix, ranges)


# ----------------------------------------------------------------------------- #
#                                   Main                                        #
# ----------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()

    num_classes = total_num_classes(
        args.num_tasks, args.num_classes_task0, args.num_classes_per_task
    )
    ranges = class_ranges(
        args.num_tasks, args.num_classes_task0, args.num_classes_per_task
    )

    results = []
    for seed in args.seeds:
        print(f"\n>>> Processing seed: {seed}")
        try:
            bwt = process_seed(seed, args, num_classes, ranges)
            print(f"    BWT = {bwt:.3f}")
            results.append({"seed": seed, "bwt": bwt})
        except Exception as e:
            print(f"    [ERROR] {e}", file=sys.stderr)

    if not results:
        print("No valid results – exiting.")
        sys.exit(1)

    # Aggregate statistics
    df = pd.DataFrame(results)
    df["mean_bwt"] = df["bwt"].mean()
    df["std_bwt"] = df["bwt"].std(ddof=0)  # population std for completeness

    # Console summary
    print("\n=== Summary ===")
    for _, row in df.iterrows():
        print(f"{row['seed']}: {row['bwt']:.3f}")
    print(f"Mean BWT: {df['mean_bwt'].iloc[0]:.3f}")
    print(f"Std  BWT: {df['std_bwt'].iloc[0]:.3f}")


if __name__ == "__main__":
    main()
