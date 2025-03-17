#!/usr/bin/env python3

import argparse
import csv
import json
import os
import copy
import torch
import concurrent.futures
import optuna
import torch
# Samplers (import as needed)
from optuna.samplers import GPSampler, QMCSampler, TPESampler
from optuna.visualization import (plot_contour, plot_optimization_history,
                                  plot_slice)

from configs.config import \
    config  # if you have a base config, though here we'll generate one dynamically
# Your modules
from model import FeNeC
from utils.other import GradKNNDataloader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--num_of_trials", type=int, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--sampler", type=str, choices=["QMCSampler", "TPESampler", "GPSampler"], required=True)
    parser.add_argument("--result_dir", type=str, required=True, help="Directory to save the result plots")
    return parser.parse_args()

def get_config(trial):
    return {
        "metric": "mahalanobis",
        "weight": "distance",
        "use_tukey": True,
        "tukey_lambda": trial.suggest_float("tukey_lambda", 0.3, 0.9),
        "num_of_shrinkages": 2,
        "shrinkage_alpha_0": trial.suggest_float("shrinkage_alpha_0", 0.5, 2),
        "shrinkage_alpha_1": trial.suggest_float("shrinkage_alpha_1", 0.5, 2),
        "norm_in_mahalanobis": True,
        
        # For 2 other resnet representations, k_kmeans remained high and k_knn low, hence the narrowing of values
        "knn_k": trial.suggest_int("knn_k", 1, 20),
        "use_kmeans": True,
        "kmeans_k": trial.suggest_int("kmeans_k", 20, 50),
        "sklearn_seed": 42,
        "use_logits_mode_0": False,
    }


def merge_models(model0, model1, trial_config, merged_device):
    """
    Merge the states of two models into one.
    - Concatenates the accumulated training data (X_train, y_train).
    - For mahalanobis metric, concatenates the covMatrices.
    - If using logits, averages the learned parameters.
    """
    # Create a new model instance on merged_device
    merged_model = FeNeC(trial_config, device=merged_device)
    
    # Merge training data: assume both models have non-None X_train and y_train.
    merged_model.X_train = torch.cat([
        model0.X_train.to(merged_device),
        model1.X_train.to(merged_device)
    ], dim=0)
    
    merged_model.y_train = torch.cat([
        model0.y_train.to(merged_device),
        model1.y_train.to(merged_device)
    ], dim=0)
    
    # Merge covMatrices if needed.
    if merged_model.metric == "mahalanobis" and (model0.covMatrices is not None and model1.covMatrices is not None):
        merged_model.covMatrices = torch.cat([
            model0.covMatrices.to(merged_device),
            model1.covMatrices.to(merged_device)
        ], dim=0)
    
    # (Optional) Set current_task to the sum of tasks from both models.
    merged_model.current_task = model0.current_task + model1.current_task

    return merged_model


def main():
    args = parse_args()

    # Select optuna sampler
    if args.sampler == "QMCSampler":
        sampler = QMCSampler()
    elif args.sampler == "TPESampler":
        sampler = TPESampler()
    elif args.sampler == "GPSampler":
        sampler = GPSampler()

    # Decide on device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Prepare data loader once (this is optional; you can also do it in the objective if needed)
    data_loader = GradKNNDataloader(
        num_tasks=6, 
        dataset_name=args.dataset,
        load_covariances=True, 
        load_prototypes=False,
        dataset_path="/shared/sets/datasets/ImageNet-Subset"
    )

    def train_on_tasks(model, task_indices):
        """
        Runs the model's fit() method on a subset of tasks.
        Each call updates the model's internal state (e.g., accumulating training data).
        """
        for i in task_indices:
            X_train, y_train, X_test, y_test, covariances, prototypes = data_loader.get_data(i)
            # Move data to the model's device.
            model.fit(X_train.to(model.device), y_train.to(model.device))
        return model

    def objective(trial):
        try:
            # Sample config (if your get_config saves the config in the model, you might want to pass it to the merged model)
            trial_config = get_config(trial)
            
            # Define two devices.
            device0 = torch.device("cuda:0")
            device1 = torch.device("cuda:1")
            
            # Create two model instances (they share the same config) on separate GPUs.
            model0 = FeNeC(trial_config, device=device0)
            model1 = FeNeC(trial_config, device=device1)
            
            # Split your 6 tasks between the two GPUs.
            # For example: tasks 0-2 on GPU0, tasks 3-5 on GPU1.
            tasks0 = range(0, 3)
            tasks1 = range(3, 6)
            
            # Run training concurrently on the two GPUs.
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                future0 = executor.submit(train_on_tasks, model0, tasks0)
                future1 = executor.submit(train_on_tasks, model1, tasks1)
                model0_trained = future0.result()
                model1_trained = future1.result()
            
            # Merge the models into a single instance on device0.
            merged_model = merge_models(model0_trained, model1_trained, trial_config, merged_device=device0)
            
            # Evaluate on the test set from the last task.
            X_train, y_train, X_test, y_test, covariances, prototypes = data_loader.get_data(5)
            predictions = merged_model.predict(X_test.to(merged_model.device))
            
            correct = torch.sum((y_test.flatten().to(merged_model.device) == predictions).int())
            acc = (correct.item() / X_test.shape[0]) * 100
            return acc

        except Exception as e:
            print(f"Error during trial: {e}")
            return 0.4

    # Create study (tell optuna we want to maximize accuracy)
    db_name = f"optuna_resnet_imnetsubset.db"
    study = optuna.create_study(direction="maximize", sampler=sampler, storage=f"sqlite:///{db_name}")
    study.optimize(objective, n_trials=args.num_of_trials)

    # Save all results to a CSV file
    trials = study.trials
    with open(args.output_file, "w", newline='') as csvfile:
        fieldnames = list(trials[0].params.keys()) + ['last_task_accuracy']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for trial in trials:
            row = {'last_task_accuracy': trial.value}
            row.update(trial.params)
            writer.writerow(row)

    print(f"Best accuracy: {study.best_trial.value}")
    print(f"Best params: {study.best_trial.params}")

    # Create result directory if it does not exist
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    # Generate and save plots
    slice_fig = plot_slice(study)
    slice_fig.write_image(f"{args.result_dir}/plot_slice.png")

    optimization_history_fig = plot_optimization_history(study)
    optimization_history_fig.write_image(f"{args.result_dir}/plot_optimization_history.png")

    contour_fig = plot_contour(study)
    contour_fig.write_image(f"{args.result_dir}/plot_contour.png")

if __name__ == "__main__":
    main()
