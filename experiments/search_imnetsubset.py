import argparse
import concurrent.futures
import copy
import csv
import json
import os

import optuna
import torch
from optuna.samplers import GPSampler, QMCSampler, TPESampler
from optuna.visualization import plot_contour, plot_optimization_history, plot_slice

from model import FeNeC
from utils.other import GradKNNDataloader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--num_of_trials", type=int, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument(
        "--sampler",
        type=str,
        choices=["QMCSampler", "TPESampler", "GPSampler"],
        required=True,
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        required=True,
        help="Directory to save the result plots",
    )
    parser.add_argument(
        "--configs_json",
        type=str,
        required=False,
        help="JSON string: lista obiektów config do kategorycznego przeszukania przez Optunę",
    )
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
    merged_model = FeNeC(trial_config, device=merged_device)

    merged_model.X_train = torch.cat(
        [model0.X_train.to(merged_device), model1.X_train.to(merged_device)], dim=0
    )
    merged_model.y_train = torch.cat(
        [model0.y_train.to(merged_device), model1.y_train.to(merged_device)], dim=0
    )

    if (
        merged_model.metric == "mahalanobis"
        and model0.covMatrices is not None
        and model1.covMatrices is not None
    ):
        merged_model.covMatrices = torch.cat(
            [
                model0.covMatrices.to(merged_device),
                model1.covMatrices.to(merged_device),
            ],
            dim=0,
        )

    merged_model.current_task = model0.current_task + model1.current_task
    return merged_model


def main():
    args = parse_args()

    if args.configs_json:
        configs_list = json.loads(args.configs_json)
        if not isinstance(configs_list, list):
            raise ValueError("--configs_json should be a JSON list of dicts")
    else:
        configs_list = None

    if args.sampler == "QMCSampler":
        sampler = QMCSampler()
    elif args.sampler == "TPESampler":
        sampler = TPESampler()
    elif args.sampler == "GPSampler":
        sampler = GPSampler()

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    data_loader = GradKNNDataloader(
        num_tasks=6,
        dataset_name=args.dataset,
        load_covariances=True,
        load_prototypes=False,
        dataset_path="/shared/sets/datasets/ImageNet-Subset",
    )

    def train_on_tasks(model, task_indices):
        for i in task_indices:
            X_train, y_train, X_test, y_test, covariances, prototypes = (
                data_loader.get_data(i)
            )
            model.fit(X_train.to(model.device), y_train.to(model.device))
        return model

    def objective(trial):
        try:
            if configs_list is not None:
                idx = trial.suggest_int("config_index", 0, len(configs_list) - 1)
                trial_config = configs_list[idx]
            else:
                trial_config = get_config(trial)

            device0 = torch.device("cuda:0")
            device1 = torch.device("cuda:1")

            model0 = FeNeC(trial_config, device=device0)
            model1 = FeNeC(trial_config, device=device1)

            tasks0 = range(0, 3)
            tasks1 = range(3, 6)

            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                future0 = executor.submit(train_on_tasks, model0, tasks0)
                future1 = executor.submit(train_on_tasks, model1, tasks1)
                model0_trained = future0.result()
                model1_trained = future1.result()

            merged_model = merge_models(
                model0_trained, model1_trained, trial_config, merged_device=device0
            )

            X_train, y_train, X_test, y_test, covariances, prototypes = (
                data_loader.get_data(5)
            )
            predictions = merged_model.predict(X_test.to(merged_model.device))

            correct = torch.sum(
                (y_test.flatten().to(merged_model.device) == predictions).int()
            )
            acc = (correct.item() / X_test.shape[0]) * 100
            return acc

        except Exception as e:
            print(f"Error during trial: {e}")
            return 0.4

    db_name = "optuna_resnet_imnetsubset.db"
    study = optuna.create_study(
        direction="maximize", sampler=sampler, storage=f"sqlite:///{db_name}"
    )
    study.optimize(objective, n_trials=args.num_of_trials)

    trials = study.trials
    with open(args.output_file, "w", newline="") as csvfile:
        fieldnames = list(trials[0].params.keys()) + ["last_task_accuracy"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for trial in trials:
            row = {"last_task_accuracy": trial.value}
            row.update(trial.params)
            writer.writerow(row)

    print(f"Best accuracy: {study.best_trial.value}")
    print(f"Best params: {study.best_trial.params}")

    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    slice_fig = plot_slice(study)
    slice_fig.write_image(f"{args.result_dir}/plot_slice.png")

    optimization_history_fig = plot_optimization_history(study)
    optimization_history_fig.write_image(
        f"{args.result_dir}/plot_optimization_history.png"
    )

    contour_fig = plot_contour(study)
    contour_fig.write_image(f"{args.result_dir}/plot_contour.png")


if __name__ == "__main__":
    main()
