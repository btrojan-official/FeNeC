config = {
    "metric": "mahalanobis",  # ["euclidean", "mahalanobis"]
    "weight": "uniform",  # ["uniform", "distance"]
    "use_tukey": False,
    "tukey_lambda": 1.0,
    "num_of_shrinkages": 1,
    "shrinkage_alpha_0": 6.122437601,
    "shrinkage_alpha_1": 8.101230677,
    "norm_in_mahalanobis": False,
    "knn_k": 1,
    "use_kmeans": True,  # if this is set to false, our method acts like a normal knn algorithm
    "kmeans_k": 26,
    "sklearn_seed": 42,
    "use_logits": False,
    "train_only_task_0": True,
    "logits_n_samples": 3,
    "logits_train_epochs": 10,
    "logits_batch_size": 64,
    "logits_learning_rate": 0.003,
    "logits_regularization_strength": 0,
    "logits_patience": 10,
}
