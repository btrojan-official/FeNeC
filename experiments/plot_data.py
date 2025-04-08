import argparse
import json
import os

import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.random_projection import SparseRandomProjection

from model import FeNeC
from utils.other import GradKNNDataloader

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config", type=str, required=True, help="JSON string of model configuration"
)
parser.add_argument(
    "--selected_classes",
    type=str,
    required=True,
    help="Comma-separated list of selected class indices",
)
parser.add_argument("--num_tasks", type=int, required=True, help="Number of tasks")
parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name")
parser.add_argument("--dataset_path", type=str, required=True, help="Path to dataset")
parser.add_argument(
    "--output_dir", type=str, required=True, help="Directory to save the plot"
)
parser.add_argument(
    "--plot_centroids",
    type=lambda x: x.lower() == "true",
    required=True,
    help="Whether to plot centroids (true/false)",
)
parser.add_argument(
    "--dim_reduction_method",
    type=str,
    choices=["PCA", "TSNE", "SRP"],
    required=True,
    help="Dimensionality reduction method",
)
args = parser.parse_args()

data_loader = GradKNNDataloader(
    num_tasks=args.num_tasks,
    dataset_name=args.dataset_name,
    dataset_path=args.dataset_path,
    load_covariances=False,
    load_prototypes=False,
    sufix="",
)

config = json.loads(args.config)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FeNeC(config, device=device)

X_all_tasks = []
y_all_tasks = []

for task_idx in range(args.num_tasks):
    X_train, y_train, _, _, _, _ = data_loader.get_data(task_idx)
    model.fit(X_train.to(device), y_train.to(device))
    X_all_tasks.append(X_train)
    y_all_tasks.append(y_train)

X_all_tasks = torch.cat(X_all_tasks, dim=0)
y_all_tasks = torch.cat(y_all_tasks, dim=0)

selected_classes = [int(cls.strip()) for cls in args.selected_classes.split(",")]

X_selected = []
y_selected = []

for cls in selected_classes:
    mask = y_all_tasks == cls
    if torch.any(mask):
        X_selected.append(X_all_tasks[mask])
        y_selected.append(y_all_tasks[mask])

if X_selected:
    X_selected = torch.cat(X_selected, dim=0)
    y_selected = torch.cat(y_selected, dim=0)
else:
    # If none of the selected classes appear in the data, just exit early
    print("No samples found for the selected classes.")
    exit(0)

centroids = model.X_train.cpu().detach()
centroid_labels = model.y_train.cpu().detach()

centroids_filtered = []
centroid_labels_filtered = []
for cls in selected_classes:
    mask = centroid_labels == cls
    centroids_filtered.append(centroids[mask])
    centroid_labels_filtered.append(centroid_labels[mask])

centroids_filtered = torch.cat(centroids_filtered, dim=0)
centroid_labels_filtered = torch.cat(centroid_labels_filtered, dim=0)

X_combined = torch.cat([X_selected.cpu(), centroids_filtered], dim=0)

if args.dim_reduction_method == "PCA":
    reducer = PCA(n_components=2)
elif args.dim_reduction_method == "TSNE":
    reducer = TSNE(n_components=2)
elif args.dim_reduction_method == "SRP":
    reducer = SparseRandomProjection(n_components=2)

X_reduced_combined = reducer.fit_transform(X_combined)

X_reduced_data = X_reduced_combined[: X_selected.shape[0]]
X_reduced_centroids = X_reduced_combined[X_selected.shape[0] :]

plt.figure(figsize=(10, 8))
import colorsys
import random

colors = [
    colorsys.hsv_to_rgb(i / len(selected_classes), 1.0, 1.0)
    for i in range(len(selected_classes))
]
colors = [f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}" for r, g, b in colors]

for i, cls in enumerate(selected_classes):
    mask = y_selected == cls
    plt.scatter(
        X_reduced_data[mask, 0],
        X_reduced_data[mask, 1],
        c=colors[i % len(colors)],
        label=f"Class {cls}",
        s=10,
        alpha=0.5,
    )

if args.plot_centroids:
    for i, cls in enumerate(selected_classes):
        mask = centroid_labels_filtered == cls
        plt.scatter(
            X_reduced_centroids[mask, 0],
            X_reduced_centroids[mask, 1],
            c=colors[i % len(colors)],
            edgecolors="black",
            marker="X",
            s=50,
            linewidths=1.5,
        )

plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.title(f"{args.dim_reduction_method} of Selected Classes with Model Centroids")
plt.legend()
plt.grid(True)

os.makedirs(args.output_dir, exist_ok=True)
output_path = os.path.join(
    args.output_dir,
    f"{args.dataset_name}_{args.dim_reduction_method.lower()}_clusters.png",
)
plt.savefig(output_path)
print(f"Plot saved to {output_path}")
