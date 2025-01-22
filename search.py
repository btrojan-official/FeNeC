#!/usr/bin/env python3

import argparse
import json
import torch
import optuna
import csv
import os
from optuna.visualization import plot_slice, plot_optimization_history, plot_contour

# Samplers (import as needed)
from optuna.samplers import QMCSampler, TPESampler, GPSampler

# Your modules
from model import Knn_Kmeans_Logits
from configs.config import config  # if you have a base config, though here we'll generate one dynamically
from utils.other import GradKNNDataloader

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, choices=["vit", "resnet"], required=True)
    parser.add_argument("--use_logits", type=lambda x: x.lower() == 'true', required=True)
    parser.add_argument("--num_of_trials", type=int, required=True)
    parser.add_argument("--num_of_tasks", type=int, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--sampler", type=str, choices=["QMCSampler", "TPESampler", "GPSampler"], required=True)
    parser.add_argument("--result_dir", type=str, required=True, help="Directory to save the result plots")
    return parser.parse_args()

def get_config(trial, model: str, use_logits: bool):
    """
    Build a config dictionary based on the combination of (model, use_logits),
    sampling the parameters from optuna.
    """

    if model == "vit" and not use_logits:
        # Config dla use_logits=false, model="vit"
        return {
            "metric": "mahalanobis",
            "weight": trial.suggest_categorical("weight", ["uniform", "distance"]),
            "tukey_lambda": 1,
            "num_of_shrinkages": 1,
            "shrinkage_alpha_0": trial.suggest_float("shrinkage_alpha_0", 0, 10),
            "shrinkage_alpha_1": 0,
            "norm_in_mahalanobis": False,
            "knn_k": trial.suggest_int("knn_k", 1, 20),
            "use_kmeans": True,
            "kmeans_k": trial.suggest_int("kmeans_k", 1, 20),
            "sklearn_seed": 42,
            "use_logits_mode_0": False,
        }

    elif model == "resnet" and not use_logits:
        # Config dla use_logits=false, model="resnet"
        return {
            "metric": "mahalanobis",
            "weight": trial.suggest_categorical("weight", ["uniform", "distance"]),
            "tukey_lambda": trial.suggest_float("tukey_lambda", 0, 1),
            "num_of_shrinkages": 2,
            "shrinkage_alpha_0": trial.suggest_float("shrinkage_alpha_0", 0, 10),
            "shrinkage_alpha_1": trial.suggest_float("shrinkage_alpha_1", 0, 10),
            "norm_in_mahalanobis": True,
            "knn_k": trial.suggest_int("knn_k", 1, 20),
            "use_kmeans": True,
            "kmeans_k": trial.suggest_int("kmeans_k", 1, 20),
            "sklearn_seed": 42,
            "use_logits_mode_0": False,
        }

    elif model == "vit" and use_logits:
        # Config dla use_logits=true, model="vit"
        return {
            "metric": "mahalanobis",
            "weight": trial.suggest_categorical("weight", ["uniform", "distance"]),
            "tukey_lambda": 1,
            "num_of_shrinkages": 1,
            "shrinkage_alpha_0": trial.suggest_float("shrinkage_alpha_0", 0, 10),
            "shrinkage_alpha_1": 0,
            "norm_in_mahalanobis": False,
            "knn_k": 1,  # fixed
            "use_kmeans": True,
            "kmeans_k": trial.suggest_int("kmeans_k", 1, 20),
            "sklearn_seed": 42,
            "use_logits_mode_0": True,
            "logits_n_samples": trial.suggest_int("logits_n_samples", 1, 20),
            "logits_train_epochs": trial.suggest_int("logits_train_epochs", 10, 50),
            "logits_batch_size": trial.suggest_int("logits_batch_size", 10, 100),
            "logits_learning_rate": trial.suggest_float("logits_learning_rate", 1e-4, 1e-2, log=True),
            "logits_regularization_strength": trial.suggest_float("logits_regularization_strength", 1, 1e4, log=True),
            "logits_patience": 5,
        }

    elif model == "resnet" and use_logits:
        # Config dla use_logits=true, model="resnet"
        return {
            "metric": "mahalanobis",
            "weight": trial.suggest_categorical("weight", ["uniform", "distance"]),
            "tukey_lambda": trial.suggest_float("tukey_lambda", 0, 1),
            "num_of_shrinkages": 2,
            "shrinkage_alpha_0": trial.suggest_float("shrinkage_alpha_0", 0, 10),
            "shrinkage_alpha_1": trial.suggest_float("shrinkage_alpha_1", 0, 10),
            "norm_in_mahalanobis": True,
            "knn_k": 1,  # fixed
            "use_kmeans": True,
            "kmeans_k": trial.suggest_int("kmeans_k", 1, 20),
            "sklearn_seed": 42,
            "use_logits_mode_0": True,
            "logits_n_samples": trial.suggest_int("logits_n_samples", 1, 20),
            "logits_train_epochs": trial.suggest_int("logits_train_epochs", 10, 50),
            "logits_batch_size": trial.suggest_int("logits_batch_size", 10, 100),
            "logits_learning_rate": trial.suggest_float("logits_learning_rate", 1e-4, 1e-2, log=True),
            "logits_regularization_strength": trial.suggest_float("logits_regularization_strength", 1, 1e4, log=True),
            "logits_patience": 5,
        }

    else:
        raise ValueError("Invalid combination of model and use_logits")

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
        num_tasks=args.num_of_tasks, 
        dataset_name=args.dataset,
        load_covariances=True, 
        load_prototypes=False
    )

    def objective(trial):
        # Sample config
        trial_config = get_config(trial, args.model, args.use_logits)

        # Create model
        model = Knn_Kmeans_Logits(trial_config, device=device)
        
        for i in range(args.num_of_tasks):
            X_train, y_train, X_test, y_test, covariances, prototypes = data_loader.get_data(i)
            model.fit(X_train, y_train)

        # Evaluate on the last task
        predictions = model.predict(X_test)
        acc = (torch.sum((y_test.flatten().to(device) == predictions).int()) 
               / X_test.shape[0] * 100).item()

        # We want to maximize accuracy, so the objective is the negative loss.
        # However, if we specify direction="maximize", we can just return `acc`.
        return acc

    # Create study (tell optuna we want to maximize accuracy)
    study = optuna.create_study(direction="maximize", sampler=sampler)
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
