config = {
    "metric": "mahalanobis", # ["euclidean", "mahalanobis"]
    "weight": "uniform", # ["uniform", "distance"]

    "use_tukey": True,
    "tukey_lambda": 0.5,

    "num_of_shrinkages": 2,
    "shrinkage_alpha_0": 1,
    "shrinkage_alpha_1": 1,

    "norm_in_mahalanobis": True,

    "knn_k": 1,

    "use_kmeans": True,
    "kmeans_k": 1,
    "sklearn_seed": 42,

    "use_logits_mode_0": True,
    "train_only_task_0": True,
    "logits_n_samples": 3,
    "logits_train_epochs":20,
    "logits_batch_size": 64,
    "logits_learning_rate": 1e-3,
    "logits_regularization_strength": 1e+3,
    "logits_patience": 10,
}