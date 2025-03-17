import h5py
import torch
import torchvision
import torchvision.transforms as transforms

class FeNeCDataLoader:
    def __init__(self, num_tasks=6, dataset_name="resnet", load_covariances=False, load_prototypes=False, dataset_path="./data", sufix=""):
        """
        This DataLoader class loads all tasks at once so they can be reused multiple times
        without re-reading from disk.
        """
        self.dataset_name = dataset_name
        self.num_tasks = num_tasks
        self.load_covariances = load_covariances
        self.load_prototypes = load_prototypes
        self.sufix = sufix
        self.data = []

        # Load all tasks into memory during initialization
        for task_num in range(self.num_tasks):
            current_file = f"{dataset_path}/{self.dataset_name}/task_{task_num}{self.sufix}.hdf5"
            with h5py.File(current_file, "r") as f:
                X_train = f["X_train"][:]
                y_train = f["y_train"][:]
                covariances = f["covariances"][:]
                X_test = f["X_test"][:]
                y_test = f["y_test"][:]

                # If prototypes are needed, load them; otherwise set None
                if self.load_prototypes:
                    prototypes = f["prototypes"][:]
                else:
                    prototypes = None

                # Store loaded data in a list of dictionaries
                self.data.append({
                    "X_train": torch.tensor(X_train),
                    "y_train": torch.tensor(y_train),
                    "X_test": torch.tensor(X_test),
                    "y_test": torch.tensor(y_test),
                    "covariances": torch.tensor(covariances),
                    "prototypes": torch.tensor(prototypes) if prototypes is not None else None
                })

    def get_data(self, task_num):
        """
        Returns a tuple (X_train, y_train, X_test, y_test, covariances, prototypes) 
        for the specified task number.
        """
        X_train = self.data[task_num]["X_train"]
        y_train = self.data[task_num]["y_train"]
        X_test = self.data[task_num]["X_test"]
        y_test = self.data[task_num]["y_test"]

        # Retrieve covariances and prototypes only if they were loaded
        covariances = self.data[task_num]["covariances"] if self.load_covariances else None
        prototypes = self.data[task_num]["prototypes"] if self.load_prototypes else None

        return X_train, y_train, X_test, y_test, covariances, prototypes
      
    def get_data(self, task_num):
        """
        Returns a tuple (X_train, y_train, X_test, y_test, covariances, prototypes) 
        for the specified task number.
        """
        X_train = self.data[task_num]["X_train"]
        y_train = self.data[task_num]["y_train"]
        X_test = self.data[task_num]["X_test"]
        y_test = self.data[task_num]["y_test"]

        # Retrieve covariances and prototypes only if they were loaded
        covariances = self.data[task_num]["covariances"] if self.load_covariances else None
        prototypes = self.data[task_num]["prototypes"] if self.load_prototypes else None

        return X_train, y_train, X_test, y_test, covariances, prototypes