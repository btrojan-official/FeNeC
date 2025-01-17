import torch
import torchvision
import torchvision.transforms as transforms

import h5py

def _get_single_class_examples(X_train, y_train, class_number, device):
    y_train = y_train.view(-1).to(device)

    indices = (y_train == class_number).nonzero(as_tuple=True)[0].to(device)

    return X_train[indices]

def load_data(task_num=0, dataset_name="resnet",load_covariances=False, load_prototypes=False):
    current_file = f"./data/{dataset_name}/task_{task_num}.hdf5"

    with h5py.File(current_file, "r") as f:
        X_train = f["X_train"][:]
        y_train = f["y_train"][:]
        covariances = f["covariances"][:]
        X_test = f["X_test"][:]
        y_test = f["y_test"][:]
        test_predictions = f["test_predictions"][:]
        task_id = f["info"].attrs["task"]
        accuracy = f["info"].attrs["accuracy"]
        if load_prototypes:
            prototypes = f["prototypes"][:]

        print(f"Accuracy: {accuracy}, Task ID: {task_id}")

        if load_covariances and load_prototypes:
            return torch.tensor(X_train), torch.tensor(y_train), torch.tensor(X_test), torch.tensor(y_test), torch.tensor(covariances), torch.tensor(prototypes)
        if load_covariances:
            return torch.tensor(X_train), torch.tensor(y_train), torch.tensor(X_test), torch.tensor(y_test), torch.tensor(covariances)
        if load_prototypes:
            return torch.tensor(X_train), torch.tensor(y_train), torch.tensor(X_test), torch.tensor(y_test), torch.tensor(prototypes)
        return torch.tensor(X_train), torch.tensor(y_train), torch.tensor(X_test), torch.tensor(y_test)