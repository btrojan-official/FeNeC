import argparse
import csv
import os
import sys

import pandas as pd
import torch
from model import GradKNN
from utils.other import GradKNNDataloader

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True,
                        help="Base path to the dataset")
    parser.add_argument("--name1", type=str, required=True,
                        help="Name of the first run subfolder or identifier")
    parser.add_argument("--name2", type=str, required=True,
                        help="Name of the second run subfolder or identifier")
    parser.add_argument("--name3", type=str, required=True,
                        help="Name of the third run subfolder or identifier")
    parser.add_argument("--model", type=str, choices=["resnet", "vit"], required=True,
                        help="Model architecture (resnet or vit)")
    parser.add_argument("--best_hyperparameters_file", type=str, required=True,
                        help="Path to the CSV file containing the best hyperparameters")
    parser.add_argument("--num_of_tasks", type=int, required=True,
                        help="Number of tasks to train/test on")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Path to the output CSV file where results will be stored")
    return parser.parse_args()

def build_config_from_row(row, model):
    """
    Builds a config dictionary from a CSV row for the given model.
    For simplicity, only the resnet scenario is detailed here according to example columns.
    Extend or adjust for 'vit' if needed (with appropriate parameter columns).
    """
    if model == "resnet":
        # Use columns as found in your CSV (like 'knn_k', 'kmeans_k', etc.)
        return {
            "metric": "mahalanobis",
            "weight": "distance",
            "use_tukey": True,
            "tukey_lambda": float(row['tukey_lambda'].replace(',', '.')),
            "num_of_shrinkages": 2,
            "shrinkage_alpha_0": float(row['shrinkage_alpha_0'].replace(',', '.')),
            "shrinkage_alpha_1": float(row['shrinkage_alpha_1'].replace(',', '.')),
            "norm_in_mahalanobis": True,
            "knn_k": int(row['knn_k']),
            "use_kmeans": True,
            "kmeans_k": int(row['kmeans_k']),
            "sklearn_seed": 42,
            "use_logits_mode_0": False
        }
    else:  # For vit or other, adjust parameters as needed
        raise NotImplementedError("Model configuration for this scenario is not defined.")

def run_evaluation(dataset_path, config, num_of_tasks, name):
    """
    Loads the dataset from 'dataset_path', instantiates a GradKNN model using 'config',
    trains across 'num_of_tasks', and returns the accuracy on each task and the average accuracy.
    """
    # Decide on device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    device = torch.device("mps")

    data_loader = GradKNNDataloader(
        num_tasks=num_of_tasks,
        dataset_name=name,
        dataset_path=dataset_path,
        load_covariances=True,
        load_prototypes=False
    )
    model = GradKNN(config, device=device)

    accuracies = []
    # Train on all tasks
    for task_idx in range(num_of_tasks):
        X_train, y_train, X_test, y_test, _, _ = data_loader.get_data(task_idx)
        model.fit(X_train, y_train)

        # Evaluate on the current task
        predictions = model.predict(X_test)
        correct = torch.sum((y_test.flatten().to(device) == predictions).int())
        accuracy = (correct / X_test.shape[0] * 100).item()
        accuracies.append(accuracy)

    # Return last task accuracy and average accuracy across all tasks
    last_task_accuracy = accuracies[-1]
    avg_accuracy = sum(accuracies) / len(accuracies)
    return last_task_accuracy, avg_accuracy

def main():
    args = parse_args()

    # Read the CSV containing best hyperparameters
    df = pd.read_csv(args.best_hyperparameters_file, delimiter=';')

    results = []
    for _, row in df.iterrows():
        # Build config from row
        config = build_config_from_row(row, args.model)

        # Evaluate on 3 runs/data subfolders
        last_acc_1, avg_acc_1 = run_evaluation(args.dataset, config, args.num_of_tasks, args.name1)
        last_acc_2, avg_acc_2 = run_evaluation(args.dataset, config, args.num_of_tasks, args.name2)
        last_acc_3, avg_acc_3 = run_evaluation(args.dataset, config, args.num_of_tasks, args.name3)

        # Compute average of last task accuracies and average of average accuracies
        avg_last_task_acc = (last_acc_1 + last_acc_2 + last_acc_3) / 3.0
        avg_of_avg_acc = (avg_acc_1 + avg_acc_2 + avg_acc_3) / 3.0

        # Copy the original row data and append new columns
        new_row = row.to_dict()
        new_row['last_task_accuracy_name1'] = last_acc_1
        new_row['last_task_accuracy_name2'] = last_acc_2
        new_row['last_task_accuracy_name3'] = last_acc_3
        new_row['average_last_task_accuracy'] = avg_last_task_acc
        new_row['average_accuracy_name1'] = avg_acc_1
        new_row['average_accuracy_name2'] = avg_acc_2
        new_row['average_accuracy_name3'] = avg_acc_3
        new_row['average_of_average_accuracy'] = avg_of_avg_acc
        results.append(new_row)

    # Save augmented results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(args.output_file, index=False, sep=';')

if __name__ == "__main__":
    main()
