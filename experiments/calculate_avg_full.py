print("Script started")

import argparse
import csv
import os
import sys

import pandas as pd
import torch

from model import FeNeC
from utils.other import GradKNNDataloader

print("Loaded modules")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, required=True, help="Base path to the dataset"
    )
    parser.add_argument(
        "--name1",
        type=str,
        required=True,
        help="Name of the first run subfolder or identifier",
    )
    parser.add_argument(
        "--name2",
        type=str,
        required=True,
        help="Name of the second run subfolder or identifier",
    )
    parser.add_argument(
        "--name3",
        type=str,
        required=True,
        help="Name of the third run subfolder or identifier",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["resnet", "vit"],
        required=True,
        help="Model architecture (resnet or vit)",
    )
    parser.add_argument(
        "--best_hyperparameters_file",
        type=str,
        required=True,
        help="Path to the CSV file containing the best hyperparameters",
    )
    parser.add_argument(
        "--num_of_tasks",
        type=int,
        required=True,
        help="Number of tasks to train/test on",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to the output CSV file where results will be stored",
    )
    parser.add_argument(
        "--sufix", type=str, default="", help="Optional suffix for the dataset files"
    )
    return parser.parse_args()


def build_config_from_row(row, model):
    """
    Builds a config dictionary from a CSV row for the given model.
    For simplicity, only the resnet scenario is detailed here according to example columns.
    Extend or adjust for 'vit' if needed (with appropriate parameter columns).
    """
    if model == "resnet":
        return {
            "metric": "mahalanobis",
            "weight": "distance",
            "use_tukey": True,
            "tukey_lambda": float(row["tukey_lambda"].replace(",", ".")),
            "num_of_shrinkages": 2,
            "shrinkage_alpha_0": float(row["shrinkage_alpha_0"].replace(",", ".")),
            "shrinkage_alpha_1": float(row["shrinkage_alpha_1"].replace(",", ".")),
            "norm_in_mahalanobis": True,
            "knn_k": int(row["knn_k"]),
            "use_kmeans": True,
            "kmeans_k": int(row["kmeans_k"]),
            "sklearn_seed": 42,
            "use_logits_mode_0": False,
        }
    elif model == "vit":
        return {
            "metric": "mahalanobis",
            "weight": "distance",
            "use_tukey": False,
            "tukey_lambda": 1,
            "num_of_shrinkages": 1,
            "shrinkage_alpha_0": float(row["shrinkage_alpha_0"].replace(",", ".")),
            "shrinkage_alpha_1": float(row["shrinkage_alpha_1"].replace(",", ".")),
            "norm_in_mahalanobis": False,
            "knn_k": int(row["knn_k"]),
            "use_kmeans": True,
            "kmeans_k": int(row["kmeans_k"]),
            "sklearn_seed": 42,
            "use_logits_mode_0": False,
        }

    else:
        raise NotImplementedError(
            "Model configuration for this scenario is not defined."
        )


def run_evaluation(dataset_path, config, num_of_tasks, name, sufix):
    """
    Loads the dataset from 'dataset_path', instantiates a GradKNN model using 'config',
    trains across 'num_of_tasks', and returns the accuracy on each task and the average accuracy.
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    data_loader = GradKNNDataloader(
        num_tasks=num_of_tasks,
        dataset_name=name,
        dataset_path=dataset_path,
        load_covariances=True,
        load_prototypes=False,
        sufix=sufix,
    )
    model = FeNeC(config, device=device)

    accuracies = []
    for task_idx in range(num_of_tasks):
        X_train, y_train, X_test, y_test, _, _ = data_loader.get_data(task_idx)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        correct = torch.sum((y_test.flatten().to(device) == predictions).int())
        accuracy = (correct / X_test.shape[0] * 100).item()
        accuracies.append(accuracy)

    return accuracies


def main():
    args = parse_args()

    df = pd.read_csv(args.best_hyperparameters_file, delimiter=";")

    results = []
    for _, row in df.iterrows():
        print(f"Processing row")
        config = build_config_from_row(row, args.model)

        accs_1 = run_evaluation(
            args.dataset, config, args.num_of_tasks, args.name1, args.sufix
        )
        accs_2 = run_evaluation(
            args.dataset, config, args.num_of_tasks, args.name2, args.sufix
        )
        accs_3 = run_evaluation(
            args.dataset, config, args.num_of_tasks, args.name3, args.sufix
        )

        avg_last_task_acc = (accs_1[-1] + accs_2[-1] + accs_3[-1]) / 3.0
        avg_of_avg_acc = (
            sum(accs_1) / len(accs_1)
            + sum(accs_2) / len(accs_2)
            + sum(accs_3) / len(accs_3)
        ) / 3.0

        new_row = row.to_dict()
        for i in range(args.num_of_tasks):
            new_row[f"accuracy_task_{i+1}_name1"] = accs_1[i]
            new_row[f"accuracy_task_{i+1}_name2"] = accs_2[i]
            new_row[f"accuracy_task_{i+1}_name3"] = accs_3[i]
        new_row["last_task_accuracy_name1"] = accs_1[-1]
        new_row["last_task_accuracy_name2"] = accs_2[-1]
        new_row["last_task_accuracy_name3"] = accs_3[-1]
        new_row["average_last_task_accuracy"] = avg_last_task_acc
        new_row["average_accuracy_name1"] = sum(accs_1) / len(accs_1)
        new_row["average_accuracy_name2"] = sum(accs_2) / len(accs_2)
        new_row["average_accuracy_name3"] = sum(accs_3) / len(accs_3)
        new_row["average_of_average_accuracy"] = avg_of_avg_acc
        results.append(new_row)

    results_df = pd.DataFrame(results)
    results_df.to_csv(args.output_file, index=False, sep=";")


if __name__ == "__main__":
    main()
