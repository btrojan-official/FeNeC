import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_method_scores(df: pd.DataFrame, title: str) -> None:
    """
    Plots the average accuracy per task for different methods with error bands (std deviation).

    Args:
        df (pd.DataFrame): A DataFrame in wide format with method names as rows and task accuracies as columns.
                        Must contain a column 'method_name' and one column per task.
        title (str): Title of the plot and also the filename (PDF) used to save the plot.

    Behavior:
        - Converts the input DataFrame to long format for seaborn plotting.
        - Groups by method and task to compute mean and standard deviation of accuracy.
        - Plots accuracy curves with error bands for methods: 'FeCAM', 'FeNeC', and 'FeNeC-Log'.
        - Saves the plot as a PDF in the 'plots/' directory.
        - Displays the plot using matplotlib.
    """

    markers = ["d", "o", "*"]
    linestyles = ["solid", "solid", "solid"]
    alpha = 0.6

    df_long = df.melt(id_vars=["method_name"], var_name="task", value_name="accuracy")
    df_long["task"] = df_long["task"].astype(int)

    grouped = (
        df_long.groupby(["method_name", "task"])
        .agg({"accuracy": ["mean", "std"]})
        .reset_index()
    )
    grouped.columns = ["method_name", "task", "mean_accuracy", "std_accuracy"]

    sns.set_context("paper", font_scale=1.6)
    sns.set_style("whitegrid")

    plt.figure(figsize=(6, 4))

    methods_names = ["FeCAM", "FeNeC", "FeNeC-Log"]

    for i, method in enumerate(methods_names):
        method_data = grouped[grouped["method_name"] == method]
        plt.plot(
            method_data["task"],
            method_data["mean_accuracy"],
            marker=markers[i],
            linestyle=linestyles[i],
            label=method,
            linewidth=2,
            markersize=8,
            alpha=alpha,
        )
        plt.fill_between(
            method_data["task"],
            method_data["mean_accuracy"] - method_data["std_accuracy"],
            method_data["mean_accuracy"] + method_data["std_accuracy"],
            alpha=0.2,
        )

    method = grouped["method_name"].unique()[0]
    method_data = grouped[grouped["method_name"] == method]
    plt.xticks(method_data["task"])

    plt.xlabel("Task number", fontsize=19.2)
    plt.ylabel("Accuracy (%)", fontsize=19.2)
    plt.title(title, fontsize=22.4)
    plt.legend(title="Methods", fontsize=14.5)
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    if os.path.exists(os.path.join("plots")) == False:
        os.makedirs(os.path.join("plots"))
    plt.savefig(
        os.path.join("plots", title + ".pdf"), format="pdf", bbox_inches="tight"
    )

    plt.show()
