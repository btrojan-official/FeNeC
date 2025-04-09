import os
import sys

import pandas as pd


def get_top_parameters(csv_path, output_folder):
    df = pd.read_csv(csv_path, delimiter=";")
    df = df.sort_values(by="last_task_accuracy", ascending=False)
    df = df.drop_duplicates(subset="last_task_accuracy")
    df = df.head(5)
    original_name = os.path.basename(csv_path)
    output_file = os.path.join(output_folder, f"BEST_{original_name}")
    df.to_csv(output_file, index=False, sep=";")


if __name__ == "__main__":
    csv_path = sys.argv[1]
    output_folder = sys.argv[2]

    get_top_parameters(csv_path, output_folder)
