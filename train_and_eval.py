import torch
import numpy as np

from model import FeNeC

from configs.imagenet_subset_resnet import config
from utils.loader import FeNeCDataLoader

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")


# For simplicity, you can pass extracted features from backbone with shapes as below to the model:
# - X_train (num_train_samples, num_features)
# - y_train (num_train_samples)
# - X_test (num_test_samples, num_features)
# - y_test (num_test_samples)
# 
# Here we are using our dataloader, but it requires the datasets to be stored in
# the given data directory and the directory with dataset name (ex. "./data/tinyimagenet") Each task data need to be stored
# in task_0.hdf5 files were 0 is number of task. In each of hdf5 files there are some keys specified. 
# Look at utils/loader.py file for more information.

num_tasks = 10
num_classes = 100
accuracies = []

data_loader = FeNeCDataLoader(
    num_tasks=num_tasks,
    dataset_name="seed_1993",
    load_covariances=True,
    load_prototypes=False,
    dataset_path="/shared/sets/datasets/ViT_CIFAR/",
)
model = FeNeC(config, device=device)

per_class_accuracy_matrix = np.zeros((num_tasks, num_classes))

for task_id in range(num_tasks):
    X_train, y_train, X_test, y_test, covariances, prototypes = data_loader.get_data(task_id)

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # Overall accuracy
    accuracy = (
        torch.sum((y_test.flatten().to(device) == predictions).int()) / X_test.shape[0]
    ).item()
    accuracies.append(accuracy)
    print(f"Task {task_id} - Accuracy: {accuracy}")

    # Per-class accuracy
    y_test_cpu = y_test.flatten().cpu()
    preds_cpu = predictions.cpu()

    for cls in range(num_classes):
        mask = y_test_cpu == cls
        if mask.sum().item() == 0:
            acc = np.nan  # No samples for this class
        else:
            correct = (preds_cpu[mask] == cls).sum().item()
            acc = correct / mask.sum().item()

        per_class_accuracy_matrix[task_id, cls] = acc

# Print average accuracy
print(f"\nAverage incremental accuracy: {sum(accuracies)/len(accuracies)}")

