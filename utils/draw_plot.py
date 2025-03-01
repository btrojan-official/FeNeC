import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

def plot_method_scores(df: pd.DataFrame, title: str) -> None:
    """
    Plots the mean and standard deviation of method scores across multiple runs for a single task.
    
    Parameters:
        df (pd.DataFrame): A dataframe containing method names and their scores across different tasks.
    """

    df_long = df.melt(id_vars=["method_name"], var_name="task", value_name="accuracy")
    df_long["task"] = df_long["task"].astype(int)
    
    grouped = df_long.groupby(["method_name", "task"]).agg({
        "accuracy": ["mean", "std"]
    }).reset_index()
    grouped.columns = ["method_name", "task", "mean_accuracy", "std_accuracy"]
    
    sns.set_context("paper", font_scale=1.6)
    sns.set_style("whitegrid")

    plt.figure(figsize=(7, 5))
    
    for method in grouped["method_name"].unique():
        method_data = grouped[grouped["method_name"] == method]
        plt.plot(method_data["task"], method_data["mean_accuracy"], marker='o', label=method)
        plt.fill_between(method_data["task"], 
                         method_data["mean_accuracy"] - method_data["std_accuracy"],
                         method_data["mean_accuracy"] + method_data["std_accuracy"],
                         alpha=0.2)
    
    plt.xlabel("Task number", fontsize=19.2)
    plt.ylabel("Accuracy", fontsize=19.2)
    plt.title(title, fontsize=22.4)
    plt.legend(title="Methods")
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    if os.path.exists(os.path.join("hyperparameters_selection", "plots")) == False:
        os.makedirs(os.path.join("hyperparameters_selection", "plots"))
    plt.savefig(os.path.join("hyperparameters_selection", "plots", title + ".pdf"), format="pdf", bbox_inches="tight")
    
    # Show plot
    plt.show()
