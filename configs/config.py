config = {
    "metric": "mahalanobis", # ["euclidean", "mahalanobis"]
    "weight": "uniform", # ["uniform", "distance"]

    "tukey_lambda": 0.4,
    "num_of_shrinkages": 2,
    "shrinkage_alpha_0": 0.85,
    "shrinkage_alpha_1": 1.55,

    "norm_in_mahalanobis": True,

    "knn_k": 3,

    "use_kmeans": True,
    "kmeans_k": 5,
    "sklearn_seed": 42,

    "use_logits_mode_0": False,
    "logits_n_samples": 6,
    "logits_train_epochs":50,
    "logits_batch_size": 64,
    "logits_learning_rate": 1e-3,
    "logits_regularization_strength": 0,
    "logits_patience": 5,
}