import os
import argparse
import h5py
import numpy as np

def calculate_bwt(acc_matrix, num_tasks, num_classes_task_0, num_classes_per_task):
    """
    acc_matrix: shape (num_tasks, num_classes)
    """
    acc_matrix = np.asarray(acc_matrix)
    if acc_matrix.ndim == 0:
        raise ValueError("acc_matrix is a scalar, not an array. Check input data.")
    if acc_matrix.ndim == 1:
        # Only one task, shape (num_classes,)
        acc_matrix = acc_matrix[None, :]
    if acc_matrix.shape[0] != num_tasks:
        raise ValueError(f"acc_matrix shape {acc_matrix.shape} does not match num_tasks={num_tasks}")

    bwt_list = []
    # Task 0
    bwt_list.append(
        np.mean(acc_matrix[-1][:num_classes_task_0]) - np.mean(acc_matrix[0][:num_classes_task_0])
    )
    # Subsequent tasks
    for i in range(1, num_tasks-1):
        start = num_classes_task_0 + (i-1)*num_classes_per_task
        end = num_classes_task_0 + i*num_classes_per_task
        bwt_list.append(
            np.mean(acc_matrix[-1][start:end]) - np.mean(acc_matrix[i][start:end])
        )
    return np.mean(bwt_list) * 100

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to directory containing seed dirs with task_{i}.hdf5 files")
    parser.add_argument("--num_classes_task_0", type=int, required=True)
    parser.add_argument("--num_classes_per_task", type=int, required=True)
    args = parser.parse_args()

    dataset_path = args.dataset_path
    num_classes_task_0 = args.num_classes_task_0
    num_classes_per_task = args.num_classes_per_task

    seed_dirs = [os.path.join(dataset_path, d) for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    seed_dirs.sort()

    bwt_values = []

    for seed_dir in seed_dirs:
        # Find all task_{i}.hdf5 files and sort by task number
        task_files = [f for f in os.listdir(seed_dir) if f.startswith("task_") and f.endswith(".hdf5")]
        task_files = sorted(task_files, key=lambda x: int(x.split("_")[1].split(".")[0]))
        num_tasks = len(task_files)

        # Read per-task per-class accuracy from each file
        acc_matrix = []
        for task_file in task_files:
            with h5py.File(os.path.join(seed_dir, task_file), "r") as f:
                if "info" in f and "accuracy" in f["info"].attrs:
                    acc = f["info"].attrs["accuracy"]
                    acc = np.array(acc)
                    if acc.ndim == 0:
                        # Scalar, make it shape (1,)
                        acc = np.array([acc])
                    acc_matrix.append(acc)
                else:
                    raise ValueError(f"Missing accuracy in {os.path.join(seed_dir, task_file)}")
        acc_matrix = np.array(acc_matrix)
        # acc_matrix shape: (num_tasks, num_classes)
        bwt = calculate_bwt(acc_matrix, num_tasks, num_classes_task_0, num_classes_per_task)
        print(f"BWT for seed {os.path.basename(seed_dir)}: {bwt}")
        bwt_values.append(bwt)

    bwt_values = np.array(bwt_values)
    print(f"\nMean BWT: {np.mean(bwt_values)}")
    print(f"Std BWT: {np.std(bwt_values)}")

if __name__ == "__main__":
    main()
