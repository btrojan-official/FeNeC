import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from utils.cov_matrices_operations import (calc_single_covariance,
                                           matrix_shrinkage,
                                           normalize_covariance_matrix,
                                           tukeys_transformation)
from utils.metric_functions import euclidean, mahalanobis
from utils.other import (get_single_class_examples,
                         get_smallest_values_per_class)


class FeNeC:
    def __init__(self, config, device="cpu"):
        """
        Initializes the FeNeC continual learning model.

        Args:
            config (dict): Configuration dictionary containing model parameters 
                        such as metric type, KNN settings, shrinkage factors, 
                        and whether to use logits or not.
            device (str): Device to run the model on ('cpu', 'cuda' or 'mps').

        Returns:
            None
        """

        self.device = device
        self.current_task = -1

        self.metric = config["metric"]
        self.weight = config["weight"]

        self.knn_k = config["knn_k"]

        self.num_of_shrinkages = config["num_of_shrinkages"]
        self.shrinkage_alpha_0 = config["shrinkage_alpha_0"]
        self.shrinkage_alpha_1 = config["shrinkage_alpha_1"]
        self.norm_in_mahalanobis = config["norm_in_mahalanobis"]

        self.use_tukey = config["use_tukey"]
        if not self.use_tukey:
            self.tukey_lambda = 1
        else:
            self.tukey_lambda = config["tukey_lambda"]

        self.use_kmeans = config["use_kmeans"]
        self.kmeans_k = config["kmeans_k"]
        self.sklearn_seed = config["sklearn_seed"]

        self.X_train = None
        self.y_train = None
        self.covMatrices = None

        self.use_logits = config["use_logits"]
        if self.use_logits:
            self.train_only_task_0 = config["train_only_task_0"]
            self.logits_n_samples = config["logits_n_samples"]
            self.logits_train_epochs = config["logits_train_epochs"]
            self.logits_batch_size = config["logits_batch_size"]
            self.logits_learning_rate = config["logits_learning_rate"]
            self.logits_regularization_strength = config[
                "logits_regularization_strength"
            ]
            self.logits_patience = config["logits_patience"]

            self.parameters = torch.nn.ParameterDict(
                {
                    "a": torch.nn.Parameter(torch.randn(1, device=self.device)),
                    "b": torch.nn.Parameter(torch.randn(1, device=self.device)),
                }
            )

    def fit(self, X_train, y_train):
        """
        Fits the model to the current task's training data.

        Args:
            X_train (torch.Tensor): Feature tensor of the current task. (num_train_samples, num_features)
            y_train (torch.Tensor): Label tensor of the current task. (num_train_samples)

        Returns:
            None

        Notes:
            - Computes and stores centroids (optionally using KMeans).
            - Applies optional Tukey transformation.
            - Calculates and stores class-wise covariance matrices if using Mahalanobis distance.
            - Optionally trains a logits-based predictor.
        """

        self.current_task += 1

        if self.X_train is None or self.y_train is None:
            if self.metric == "mahalanobis":
                self.covMatrices = (
                    self._calc_covariances(
                        tukeys_transformation(X_train, self.tukey_lambda), y_train
                    )
                    .float()
                    .to(self.device)
                )
        else:
            if self.metric == "mahalanobis":
                self.covMatrices = (
                    torch.cat(
                        (
                            self.covMatrices,
                            self._calc_covariances(
                                tukeys_transformation(X_train, self.tukey_lambda),
                                y_train,
                            ),
                        )
                    )
                    .float()
                    .to(self.device)
                )
        if self.use_kmeans:
            uniqes = torch.unique(y_train, sorted=True).to(self.device)

            for i in uniqes:
                single_class_examples = get_single_class_examples(
                    X_train.to(self.device), y_train.to(self.device), i, self.device
                )
                if i == uniqes[0]:
                    new_X_train = self._kmeans(single_class_examples).to(self.device)
                    new_y_train = torch.full((self.kmeans_k,), i.item()).to(self.device)
                else:
                    new_X_train = torch.cat(
                        (
                            new_X_train,
                            self._kmeans(single_class_examples).to(self.device),
                        )
                    )
                    new_y_train = torch.cat(
                        (
                            new_y_train,
                            torch.full((self.kmeans_k,), i.item()).to(self.device),
                        )
                    )

            # Tukey transformation with lambda < 0 can't handle negative values
            if self.use_tukey:
                print(
                    "Warning!!! All values smaller than 0 were set to 0, bacuse tukey transformation can't handle negative values!"
                )
                new_X_train[new_X_train < 0] = 0

            X_train_centroids = new_X_train.float()
            y_train_centroids = new_y_train

        if self.X_train is None or self.y_train is None:
            self.X_train = X_train_centroids.to(self.device)
            self.y_train = y_train_centroids.to(self.device)

            if self.use_logits:
                self._train_logits(X_train, y_train)
        else:
            self.X_train = torch.cat((self.X_train, X_train_centroids.to(self.device)))
            self.y_train = torch.cat((self.y_train, y_train_centroids.to(self.device)))

            if self.use_logits and not self.train_only_task_0:
                self._train_logits(X_train, y_train)

    def predict(self, X_test):
        """
        Predicts class labels for given test samples.

        Args:
            X_test (torch.Tensor): Test samples to classify. (num_test_samples, num_features)

        Returns:
            torch.Tensor: Predicted class labels.

        Notes:
            - Uses either logits-based prediction or KNN with majority voting depending on configuration.
        """

        if self.use_logits:
            return self._predict_with_logits(X_test.to(self.device))
        return self._predict_with_majority_voting(X_test.to(self.device))

    def _predict_with_majority_voting(self, X_test):

        if self.metric == "euclidean":
            distances = euclidean(self.X_train, X_test, self.device)
        elif self.metric == "mahalanobis":
            distances = mahalanobis(
                self.X_train,
                self.y_train,
                X_test,
                self.covMatrices,
                self.tukey_lambda,
                self.device,
                self.norm_in_mahalanobis,
            )

        _, knn_indices = torch.topk(
            distances, self.knn_k, largest=False, dim=1, sorted=True
        )

        nearest_neighbours_matrix = self.y_train[knn_indices].squeeze()

        if self.knn_k == 1:
            return nearest_neighbours_matrix

        if len(nearest_neighbours_matrix.size()) < 2:
            nearest_neighbours_matrix = nearest_neighbours_matrix.unsqueeze(0)

        batch_size, _ = nearest_neighbours_matrix.shape

        number_of_classes = torch.max(self.y_train) + 1

        counts = torch.zeros(batch_size, number_of_classes, dtype=torch.float).to(
            self.device
        )

        if self.weight == "uniform":
            weights_matrix = torch.ones_like(
                nearest_neighbours_matrix, dtype=torch.float
            ).to(self.device)
        elif self.weight == "distance":
            weights_matrix = 1 / torch.gather(distances, 1, knn_indices).to(self.device)

        counts.scatter_add_(
            dim=1, index=nearest_neighbours_matrix, src=(weights_matrix)
        )

        most_frequent = torch.argmax(counts, dim=1)

        def is_draw(tensor):
            sorted_tensor, _ = tensor.sort(dim=0, descending=True)

            max_values = sorted_tensor[0]
            second_max_values = sorted_tensor[1]
            return max_values == second_max_values

        for i, line in enumerate(counts):
            if is_draw(line):
                most_frequent[i] = nearest_neighbours_matrix[i][0]

        return most_frequent

    def _calc_covariances(self, X_train, y_train):
        X_train = X_train.to(self.device)
        y_train = y_train.to(self.device)

        classes_list = torch.unique(y_train, sorted=True).to(self.device)

        for i in classes_list:
            cov = calc_single_covariance(X_train, y_train, i, self.device)

            for _ in range(self.num_of_shrinkages):
                cov = matrix_shrinkage(
                    cov, self.shrinkage_alpha_0, self.shrinkage_alpha_1, self.device
                )

            cov = normalize_covariance_matrix(cov)

            if i == classes_list[0]:
                covariances = cov.clone().detach()
            else:
                covariances = torch.cat((covariances, cov.clone().detach()))

        return covariances

    def _kmeans(self, X_train):
        kmeans = KMeans(n_clusters=self.kmeans_k, random_state=self.sklearn_seed)
        kmeans.fit(X_train.cpu().numpy())

        cluster_centers = kmeans.cluster_centers_

        cluster_centers_tensor = torch.tensor(cluster_centers, dtype=X_train.dtype)

        return cluster_centers_tensor

    def _train_logits(self, X_train, y_train):

        prev_task_params = {
            "a": self.parameters["a"].clone().item(),
            "b": self.parameters["b"].clone().item(),
        }

        if self.metric == "euclidean":
            distances = euclidean(self.X_train, X_train, self.device)
        elif self.metric == "mahalanobis":
            distances = mahalanobis(
                self.X_train,
                self.y_train,
                X_train,
                self.covMatrices,
                self.tukey_lambda,
                self.device,
                self.norm_in_mahalanobis,
            )

        closest_distances = get_smallest_values_per_class(
            distances, self.y_train, self.logits_n_samples
        )

        train_data, val_data, train_labels, val_labels = train_test_split(
            closest_distances, y_train, test_size=0.2, random_state=self.sklearn_seed
        )

        trainloader = DataLoader(
            TensorDataset(train_data, train_labels),
            batch_size=self.logits_batch_size,
            shuffle=True,
        )
        valloader = DataLoader(
            TensorDataset(val_data, val_labels), batch_size=self.logits_batch_size
        )

        optimizer = torch.optim.Adam(
            self.parameters.parameters(), lr=self.logits_learning_rate
        )
        criterion = torch.nn.CrossEntropyLoss()

        self._epochs_without_improvement = 0
        self._best_val_loss = float("inf")

        for epoch in range(self.logits_train_epochs):
            running_loss, regularizaion_loss = self._logits_training_loop(
                trainloader, prev_task_params, optimizer, criterion
            )
            val_loss, accuracy = self._logits_validation_loop(valloader, criterion)

            print(
                f"Epoch [{epoch+1}|{self.logits_train_epochs}] Loss: {sum(running_loss) / len(running_loss)} (in this regularization: {sum(regularizaion_loss) / len(regularizaion_loss)}) ||| Validation Loss: {sum(val_loss) / len(val_loss)}, Val accuracy: {accuracy}"
            )

            if self.early_stopping(val_loss):
                print(f"Early stopping at epoch {epoch+1}")
                break

    def early_stopping(self, val_loss):
        val_loss_value = sum(val_loss) / len(val_loss)

        if val_loss_value < self._best_val_loss:
            self._best_val_loss = val_loss_value
            self._epochs_without_improvement = 0
        else:
            self._epochs_without_improvement += 1

        if self._epochs_without_improvement == self.logits_patience:
            return True
        return False

    def _logits_training_loop(
        self, trainloader, prev_task_params, optimizer, criterion
    ):

        running_loss = []
        regularizaion_loss = []

        for data, target in trainloader:
            optimizer.zero_grad()

            output = self._calculate_logits(data)

            loss = criterion(output, target.to(self.device).long())
            reg_loss = self._calc_logits_regularization(prev_task_params)
            loss += reg_loss * self.logits_regularization_strength

            loss.backward()
            optimizer.step()

            running_loss.append(loss.item())
            regularizaion_loss.append(
                reg_loss.item() * self.logits_regularization_strength
            )

        return running_loss, regularizaion_loss

    def _logits_validation_loop(self, valloader, criterion):
        val_loss = []
        accuracy = []

        with torch.no_grad():
            for data, target in valloader:
                output = self._calculate_logits(data)
                val_loss.append(criterion(output, target.to(self.device).long()).item())
                accuracy.append(
                    [
                        self._count_correct_guesses(
                            output, target.to(self.device).long()
                        ),
                        len(target),
                    ]
                )

        acc = torch.sum(
            torch.tensor(accuracy, dtype=torch.float)[:, 0], dim=0
        ) / torch.sum(torch.tensor(accuracy, dtype=torch.float)[:, 1], dim=0)

        return val_loss, acc

    def _calc_logits_regularization(self, prev_task_params):
        # Applies L2 regularization

        if self.current_task == 0:
            return torch.tensor(0.0, device=self.device)

        regularizaion_loss = torch.tensor(0.0, device=self.device)

        for param_key in self.parameters:
            regularizaion_loss += torch.sum(
                (self.parameters[param_key] - prev_task_params[param_key]) ** 2
            )

        return regularizaion_loss

    def _calculate_logits(self, data):
        """
        Predict class probabilities using logits.

        Args:
            data (torch.Tensor): Tensor of shape (batch_size, num_classes, n_closest_samples) containing input data.

        Returns:
            torch.Tensor: Tensor of shape (batch_size, num_classes) containing class probabilities.
        """

        data_transformed = self.parameters["a"] + self.parameters["b"] * torch.log(
            data + 1e-10
        )
        data_activated = F.leaky_relu(data_transformed, negative_slope=0.01)
        logits = data_activated.sum(dim=-1)

        return logits

    def _predict_with_logits(self, X_test):
        if self.metric == "euclidean":
            distances = euclidean(self.X_train, X_test, self.device)
        elif self.metric == "mahalanobis":
            distances = mahalanobis(
                self.X_train,
                self.y_train,
                X_test,
                self.covMatrices,
                self.tukey_lambda,
                self.device,
                self.norm_in_mahalanobis,
            )

        closest_distances = get_smallest_values_per_class(
            distances, self.y_train, self.logits_n_samples
        )

        logits = self._calculate_logits(closest_distances)
        prediction = logits.argmax(dim=1)

        return prediction

    def _count_correct_guesses(self, logits, targets):

        predicted_classes = torch.argmax(logits, dim=1)

        correct_guesses = (predicted_classes == targets).sum().item()

        return correct_guesses
