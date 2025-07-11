"""
Compute backward transfer (BWT) for a FeCAM-style experiment.

Usage
-----
python compute_bwt.py \
    --dataset_dir /shared/results/kksiazek/FeCAM/ImageNet-R \
    --num_classes_task_0 50 \
    --num_classes_per_task 10
"""

import argparse
import os
import re
import h5py
import numpy as np


def load_preds_and_labels(h5_path):
    """Return predictions and labels saved in a task_X.hdf5 file."""
    with h5py.File(h5_path, "r") as f:
        preds = f["test_predictions"][()]          # (N,)
        labels = f["test_labels_for_predictions"][()]  # (N,)
    return preds.squeeze(), labels.squeeze()


def class_range_for_task(tid, n_cls_task0, n_cls_per_task):
    """Return the numpy array with class ids that belong to task `tid`."""
    if tid == 0:
        return np.arange(0, n_cls_task0)
    start = n_cls_task0 + (tid - 1) * n_cls_per_task
    return np.arange(start, start + n_cls_per_task)


def accuracy_on_subset(preds, labels, cls_subset):
    """Compute accuracy restricted to a given subset of classes."""
    mask = np.isin(labels, cls_subset)
    if mask.sum() == 0:         # no samples from that subset in the array
        return np.nan
    return (preds[mask] == labels[mask]).mean()


def compute_bwt_for_seed(seed_path, n_cls_task0, n_cls_per_task):
    """Compute BWT for a single seed directory."""
    task_files = [
        f for f in os.listdir(seed_path) if re.match(r"task_\d+\.hdf5", f)
    ]
    task_ids = sorted(int(re.findall(r"\d+", f)[0]) for f in task_files)
    n_tasks = len(task_ids)
    if n_tasks < 2:
        raise RuntimeError(f"Seed {seed_path} has fewer than two tasks – can't compute BWT")

    preds, labels = {}, {}
    for tid in task_ids:
        p, l = load_preds_and_labels(os.path.join(seed_path, f"task_{tid}.hdf5"))
        preds[tid], labels[tid] = p.astype(np.int64), l.astype(np.int64)

    acc_initial = {}
    for tid in task_ids:
        cls_subset = class_range_for_task(tid, n_cls_task0, n_cls_per_task)
        acc_initial[tid] = accuracy_on_subset(preds[tid], labels[tid], cls_subset)

    last_tid = task_ids[-1]
    acc_final = {}
    for tid in task_ids[:-1]:           # we skip the final task when averaging
        cls_subset = class_range_for_task(tid, n_cls_task0, n_cls_per_task)
        acc_final[tid] = accuracy_on_subset(preds[last_tid], labels[last_tid], cls_subset)
  
    deltas = [
        acc_final[tid] - acc_initial[tid]
        for tid in task_ids[:-1]
        if not (np.isnan(acc_final[tid]) or np.isnan(acc_initial[tid]))
    ]
    return float(np.mean(deltas) * 100)   # convert to percentage



def main():
    parser = argparse.ArgumentParser(description="Compute Backward Transfer (BWT).")
    parser.add_argument("--dataset_dir", required=True,
                        help="Path that contains seed_xxxx folders.")
    parser.add_argument("--num_classes_task_0", type=int, required=True,
                        help="Number of classes in the very first task.")
    parser.add_argument("--num_classes_per_task", type=int, required=True,
                        help="Number of classes added by every subsequent task.")
    args = parser.parse_args()

    seed_dirs = sorted(
        d for d in os.listdir(args.dataset_dir)
        if os.path.isdir(os.path.join(args.dataset_dir, d)) and d.startswith("seed_")
    )
    if not seed_dirs:
        raise RuntimeError(f"No seed_* directories found inside {args.dataset_dir}")

    bwt_values = []
    print("# Backward Transfer (higher = better)\n")
    for sd in seed_dirs:
        seed_path = os.path.join(args.dataset_dir, sd)
        bwt = compute_bwt_for_seed(
            seed_path,
            args.num_classes_task_0,
            args.num_classes_per_task,
        )
        bwt_values.append(bwt)
        print(f"Seed {sd}: {bwt:+.4f} %")

    mean_bwt = np.mean(bwt_values)
    std_bwt = np.std(bwt_values)
    print("\n---------------------------")
    print(f"Mean  BWT: {mean_bwt:+.4f} %")
    print(f"Std   BWT: {std_bwt:.4f}")


if __name__ == "__main__":
    main()
