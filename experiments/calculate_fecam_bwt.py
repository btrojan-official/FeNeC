#!/usr/bin/env python3
"""
Obliczanie BWT wyłącznie z gotowych accuracy zapisanych w atrybucie
f['info'].attrs['accuracy'] w plikach task_X.hdf5.

Uruchomienie:
python compute_bwt_from_attr.py \
    --dataset_dir /shared/results/kksiazek/FeCAM/ImageNet-R \
    --num_tasks 10        # (opcjonalnie) liczba zadań jeżeli chcemy dodatkową weryfikację
"""

import argparse
import os
import re
import h5py
import numpy as np


# -----------------------------------------------------------------------------#
# Pomocnicze                                                                   #
# -----------------------------------------------------------------------------#
def read_accuracy_vector(h5_path):
    """
    Zwróć listę/lub tablicę dokładności zapisanych w atrybucie
    f['info'].attrs['accuracy'].
    """
    with h5py.File(h5_path, "r") as f:
        acc = f["info"].attrs["accuracy"]
    return np.asarray(acc, dtype=float)


def compute_bwt(seed_path):
    """
    Oblicz BWT dla pojedynczego seeda na podstawie wektorów accuracy.
    """
    # ------------------- 1. Zbierz i posortuj pliki ---------------------------
    task_files = [
        f for f in os.listdir(seed_path) if re.match(r"task_\d+\.hdf5", f)
    ]
    task_ids = sorted(int(re.findall(r"\d+", f)[0]) for f in task_files)
    n_tasks = len(task_ids)
    if n_tasks < 2:
        raise RuntimeError(f"{seed_path}: mniej niż dwa zadania – nie liczymy BWT.")

    # ------------------- 2. Wczytaj accuracy dla każdego tasku ---------------
    acc_vectors = {}
    for tid in task_ids:
        path = os.path.join(seed_path, f"task_{tid}.hdf5")
        acc_vectors[tid] = read_accuracy_vector(path)

        # szybka sanity-check: długość wektora powinna wynosić tid+1
        if len(acc_vectors[tid]) != tid + 1:
            raise ValueError(
                f"{path}: spodziewano się {tid+1} wartości accuracy, "
                f"znaleziono {len(acc_vectors[tid])}"
            )

    # ------------------- 3. BWT ----------------------------------------------
    last_vec = acc_vectors[task_ids[-1]]         # accuracy po ostatnim tasku
    deltas = []
    for tid in task_ids[:-1]:                    # pomijamy sam ostatni task
        r_ii  = acc_vectors[tid][tid]            # accuracy tasku i zaraz po nim
        r_ti  = last_vec[tid]                    # accuracy tego tasku po T-1
        deltas.append(r_ti - r_ii)

    return float(np.mean(deltas) * 100.0)        # %


# -----------------------------------------------------------------------------#
# Główna część                                                                 #
# -----------------------------------------------------------------------------#
def main():
    parser = argparse.ArgumentParser(description="BWT z atrybutu accuracy.")
    parser.add_argument("--dataset_dir", required=True,
                        help="Katalog zawierający foldery seed_*")
    parser.add_argument("--num_tasks", type=int, default=None,
                        help="(opcjonalnie) oczekiwana liczba zadań – sprawdzenie spójności")
    args = parser.parse_args()

    # ------ znajdź foldery seed_* --------------------------------------------
    seed_dirs = sorted(
        d for d in os.listdir(args.dataset_dir)
        if os.path.isdir(os.path.join(args.dataset_dir, d)) and d.startswith("seed_")
    )
    if not seed_dirs:
        raise RuntimeError(f"W {args.dataset_dir} nie znaleziono folderów seed_*")

    # ------ licz BWT per seed -------------------------------------------------
    bwt_list = []
    print("# BWT (tylko z atrybutu 'accuracy')\n")
    for sd in seed_dirs:
        path = os.path.join(args.dataset_dir, sd)
        bwt = compute_bwt(path)
        bwt_list.append(bwt)
        print(f"{sd}: {bwt:+.4f} %")

        if args.num_tasks is not None:
            # krótka kontrola: czy w seedzie jest tyle samo plików ile zadeklarowano
            if len(os.listdir(path)) < args.num_tasks:
                print(f"  ⚠  UWAGA: spodziewano {args.num_tasks} tasków, "
                      f"znaleziono {len(os.listdir(path))}")

    # ------ agregacja ---------------------------------------------------------
    mean_bwt = np.mean(bwt_list)
    std_bwt  = np.std(bwt_list)
    print("\n---------------------------")
    print(f"Średni BWT: {mean_bwt:+.4f} %")
    print(f"Std. BWT : {std_bwt:.4f}")


if __name__ == "__main__":
    main()
