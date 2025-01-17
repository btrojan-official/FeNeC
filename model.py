import torch
import torch.nn.functional as F

from sklearn.cluster import KMeans

from utils.metric_functions import _euclidean, _mahalanobis
from utils.covMatrices_operations import _tukeys_transformation, _matrix_shrinkage, _normalize_covariance_matrix, _calc_single_covariance
from utils.other import _get_single_class_examples

class Knn_Kmeans_Logits:
    def __init__(self, config, device="cpu"):
        
        self.device = device

        self.metric = config["metric"]
        self.weight = config["weight"] 

        self.knn_k = config["knn_k"]

        self.num_of_shrinkages = config["num_of_shrinkages"]
        self.shrinkage_alpha_0 = config["shrinkage_alpha_0"]
        self.shrinkage_alpha_1 = config["shrinkage_alpha_1"]
        self.norm_in_mahalanobis = config["norm_in_mahalanobis"]

        self.tukey_lambda = config["tukey_lambda"]

        self.use_kmeans = config["use_kmeans"]
        self.kmeans_k = config["kmeans_k"]
        self.kmeans_seed = config["kmeans_seed"]

        self.X_train = None
        self.y_train = None
        self.covMatrices = None

    def fit(self, X_train, y_train):

        if self.X_train is None or self.y_train is None:
            if self.metric == "mahalanobis":
                self.covMatrices = self._calc_covariances(_tukeys_transformation(X_train, self.tukey_lambda), y_train).float().to(self.device)
        else:
            if self.metric == "mahalanobis":
                self.covMatrices = torch.cat((self.covMatrices, self._calc_covariances(_tukeys_transformation(X_train,self.tukey_lambda), y_train))).float().to(self.device)
        if self.use_kmeans:
            uniqes = torch.unique(y_train, sorted=True).to(self.device)

            for i in uniqes:
                single_class_examples = _get_single_class_examples(X_train.to(self.device), y_train.to(self.device), i, self.device)
                if i == uniqes[0]: 
                    new_X_train = self._kmeans(single_class_examples).to(self.device)
                    new_y_train = torch.full((self.kmeans_k,), i.item()).to(self.device)
                else:
                    new_X_train = torch.cat((new_X_train, self._kmeans(single_class_examples).to(self.device)))
                    new_y_train = torch.cat((new_y_train, torch.full((self.kmeans_k,), i.item()).to(self.device)))

            X_train = new_X_train.float()
            y_train = new_y_train

        if self.X_train is None or self.y_train is None:
            self.X_train = X_train.to(self.device)
            self.y_train = y_train.to(self.device)
        else:
            self.X_train = torch.cat((self.X_train, X_train.to(self.device)))
            self.y_train = torch.cat((self.y_train, y_train.to(self.device)))
    
    def predict(self, X_test):

        X_test = X_test.to(self.device)

        if self.metric == "euclidean":
            distances = _euclidean(self.X_train, X_test, self.device)
        elif self.metric == "mahalanobis":
            distances = _mahalanobis(self.X_train, self.y_train, X_test, self.covMatrices, self.tukey_lambda, self.device, self.norm_in_mahalanobis)

        _, knn_indices = torch.topk(distances, self.knn_k, largest=False, dim=1, sorted=True)

        nearest_neighbours_matrix = self.y_train[knn_indices].squeeze()

        if self.knn_k == 1:
            return nearest_neighbours_matrix

        if len(nearest_neighbours_matrix.size()) < 2:
            nearest_neighbours_matrix = nearest_neighbours_matrix.unsqueeze(0)

        batch_size, _ = nearest_neighbours_matrix.shape

        number_of_classes = torch.max(self.y_train) + 1

        counts = torch.zeros(batch_size, number_of_classes, dtype=torch.float).to(self.device)

        if self.weight == "uniform": weights_matrix = torch.ones_like(nearest_neighbours_matrix, dtype=torch.float).to(self.device)
        elif self.weight == "distance": weights_matrix = 1 / torch.gather(distances, 1, knn_indices).to(self.device)

        counts.scatter_add_(dim=1, index=nearest_neighbours_matrix, src=(weights_matrix))            

        most_frequent = torch.argmax(counts, dim=1)

        def is_draw(tensor):
            sorted_tensor, _ = tensor.sort(dim=0, descending=True)

            max_values = sorted_tensor[0]
            second_max_values = sorted_tensor[1]
            return max_values == second_max_values

        for i,line in enumerate(counts):
            if is_draw(line):
                most_frequent[i] = nearest_neighbours_matrix[i][0]

        return most_frequent

    def _calc_covariances(self, X_train, y_train):
        X_train = X_train.to(self.device)
        y_train = y_train.to(self.device)

        classes_list = torch.unique(y_train, sorted=True).to(self.device)

        for i in classes_list:
            cov = _calc_single_covariance(X_train, y_train, i, self.device)

            for _ in range(self.num_of_shrinkages):
                cov = _matrix_shrinkage(cov, self.shrinkage_alpha_0, self.shrinkage_alpha_1, self.device)
                
            cov = _normalize_covariance_matrix(cov)

            if i == classes_list[0]:
                covariances = cov.clone().detach()
            else:
                covariances = torch.cat((covariances, cov.clone().detach()))
        
        return covariances

    def _kmeans(self, X_train):
        kmeans = KMeans(n_clusters=self.kmeans_k, random_state=self.kmeans_seed)
        kmeans.fit(X_train.cpu().numpy())

        cluster_centers = kmeans.cluster_centers_

        cluster_centers_tensor = torch.tensor(cluster_centers, dtype=X_train.dtype)

        return cluster_centers_tensor
