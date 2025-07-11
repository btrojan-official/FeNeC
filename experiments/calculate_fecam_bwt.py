#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute BWT from the scalar accuracies stored in FeCAM *.hdf5 result files.

The only information we read from every task file is:
    f['info'].attrs['accuracy']   # a single float

Author: <your-name>
"""

import argparse
import glob
import os
from pathlib import Path
import statistics
import h5py
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Compute backward transfer (BWT) from FeCAM result folders.")
    p.add_argument("--dataset_dir", required=True, type=str,
                   help="Directory that contains seed_xxx sub-folders with task_k.hdf5 files.")
    p.add_argument("--num_classes_task_0", required=True, type=int,
                   help="Number of classes in the initial task (kept for API compatibility).")
    p.add_argument("--num_classes_per_task", required=True, type=int,
                   help="Number of classes introduced by every subsequent task (kept for API compatibility).")
    return p.parse_args()


def read_accuracies_for_seed(seed_path: Path):
    """Return a list of accuracies ordered by task index for the given seed directory."""
    task_files = sorted(seed_path.glob("task_*.hdf5"),
                        key=lambda p: int(p.stem.split("_")[1]))  # sort by task number
    accuracies = []
    for tf in task_files:
        with h5py.File(tf, "r") as f:
            acc = f["info"].attrs["accuracy"]
            # attrs may come back as numpy scalar, bytes, etc.  Convert to float.
            accuracies.append(float(acc))
    return accuracies


def compute_bwt(accuracies, num_classes_task_0, num_classes_per_task):
    """
    Replicates the original per-class formula, but with task-level accuracies:

        BWT = mean_i ( acc_final(task_i) – acc_at_training_of_task_i )

    We exclude the last task because acc_final(last) == acc_at_training_of_last.
    """
    num_tasks = len(accuracies)
    if num_tasks < 2:
        raise ValueError("Need at least two tasks to compute BWT.")

    bwt_terms = []
    final_acc = accuracies[-1]

    # task 0 (the 'big' task)
    bwt_terms.append(final_acc - accuracies[0])

    # the incremental tasks 1 … N-2
    for i in range(1, num_tasks - 1):
        bwt_terms.append(final_acc - accuracies[i])

    return np.mean(bwt_terms) * 100.0  # convert to percentage points


def main():
    args = parse_args()
    root = Path(args.dataset_dir).expanduser()

    seed_dirs = sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("seed_"))
    if not seed_dirs:
        raise FileNotFoundError(f"No seed_* directories found under {root}")

    seed_to_bwt = {}

    for sd in seed_dirs:
        accuracies = read_accuracies_for_seed(sd)
        bwt = compute_bwt(
            accuracies,
            num_classes_task_0=args.num_classes_task_0,
            num_classes_per_task=args.num_classes_per_task,
        )
        seed_to_bwt[sd.name] = bwt

    # ----- reporting -----
    print("Backward Transfer (BWT) per seed [in percentage points]:\n")
    for seed, bwt in seed_to_bwt.items():
        print(f"  {seed:<10}: {bwt:7.3f}")

    bwt_values = list(seed_to_bwt.values())
    mean_bwt = statistics.mean(bwt_values)
    std_bwt = statistics.stdev(bwt_values) if len(bwt_values) > 1 else 0.0

    print("\nSummary:")
    print(f"  Mean BWT : {mean_bwt:7.3f}")
    print(f"  Std  BWT : {std_bwt:7.3f}")


if __name__ == "__main__":
    main()
