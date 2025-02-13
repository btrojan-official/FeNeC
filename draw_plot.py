import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def plot_method_scores(df: pd.DataFrame) -> None:
    """
    Plots the mean and standard deviation of method scores across multiple runs for a single task.
    
    Parameters:
        df (pd.DataFrame): A dataframe containing method names and their scores across different tasks.
    """

    df_long = df.melt(id_vars=["method_name"], var_name="task", value_name="accuracy")
    
    grouped = df_long.groupby(["method_name", "task"]).agg({
        "accuracy": ["mean", "std"]
    }).reset_index()
    grouped.columns = ["method_name", "task", "mean_accuracy", "std_accuracy"]
    
    plt.figure(figsize=(6, 4))
    
    for method in grouped["method_name"].unique():
        method_data = grouped[grouped["method_name"] == method]
        plt.plot(method_data["task"], method_data["mean_accuracy"], marker='o', label=method)
        plt.fill_between(method_data["task"], 
                         method_data["mean_accuracy"] - method_data["std_accuracy"],
                         method_data["mean_accuracy"] + method_data["std_accuracy"],
                         alpha=0.2)
    
    plt.xlabel("Task number")
    plt.ylabel("Accuracy")
    plt.title("Method Performance Across Tasks")
    plt.xticks(rotation=45)
    plt.legend(title="Methods")
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Show plot
    plt.show()
