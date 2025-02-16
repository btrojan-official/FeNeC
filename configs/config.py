config = {
    "metric": "mahalanobis", # ["euclidean", "mahalanobis"]
    "weight": "uniform", # ["uniform", "distance"]

    "use_tukey": True,
    "tukey_lambda": 0.40846134166276027,

    "num_of_shrinkages": 2,
    "shrinkage_alpha_0": 1.146773712678215,
    "shrinkage_alpha_1": 1.146773712678215,

    "norm_in_mahalanobis": True,

    "knn_k": 6,

    "use_kmeans": True,
    "kmeans_k": 20,
    "sklearn_seed": 11,

    "use_logits_mode_0": False,
    "train_only_task_0": True,
    "logits_n_samples": 2,
    "logits_train_epochs":10,
    "logits_batch_size": 64,
    "logits_learning_rate": 0.003029956502423998,
    "logits_regularization_strength": 0,
    "logits_patience": 10,
}