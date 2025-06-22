config = {
    "metric": "mahalanobis",  # ["euclidean", "mahalanobis"]
    "weight": "uniform",  # ["uniform", "distance"]
    "use_tukey": True,
    "tukey_lambda": 0.377,
    "num_of_shrinkages": 2,
    "shrinkage_alpha_0": 0.886,
    "shrinkage_alpha_1": 0.896,
    "norm_in_mahalanobis": True,
    "knn_k": 1,
    "use_kmeans": True,  # if this is set to false, our method acts like a normal knn algorithm
    "kmeans_k": 47,
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
