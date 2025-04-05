import argparse
import json
import os

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert CSV hyperparameter files to JSON configs."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing input CSV files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save output JSON files.",
    )
    return parser.parse_args()


def build_config_from_row(row, model):
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
        raise ValueError(f"Unsupported model type: {model}")


def main():
    args = parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    for filename in os.listdir(args.input_dir):
        if filename.endswith(".csv"):
            input_filepath = os.path.join(args.input_dir, filename)
            df = pd.read_csv(input_filepath, delimiter=";")

            # Determine model type from presence of 'tukey_lambda' column
            model = "resnet" if "tukey_lambda" in df.columns else "vit"

            # Find the row with the best average accuracy
            df["average_of_average_accuracy"] = df["average_of_average_accuracy"].apply(
                lambda x: float(str(x).replace(",", "."))
            )
            best_row = df.loc[df["average_of_average_accuracy"].idxmax()]

            config = build_config_from_row(best_row, model)

            output_filename = filename.replace(".csv", ".json")
            output_filepath = os.path.join(args.output_dir, output_filename)

            with open(output_filepath, "w") as json_file:
                json.dump(config, json_file, separators=(",", ":"))

            print(f"Saved best config to {output_filepath}")


if __name__ == "__main__":
    main()
