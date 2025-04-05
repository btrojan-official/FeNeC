import torch


def get_single_class_examples(X_train, y_train, class_number, device):
    y_train = y_train.view(-1).to(device)

    indices = (y_train == class_number).nonzero(as_tuple=True)[0].to(device)

    return X_train[indices]


def get_smallest_values_per_class(data, labels, n_smallest):
    """
    Extract n_smallest values for each class per row in the data tensor.

    Args:
        data (torch.Tensor): Tensor of shape (num_test_samples, num_classes*num_centroids_per_class) containing data values.
        labels (torch.Tensor): Tensor of shape (num_classes*num_centroids_per_class,) containing labels for each column.
        n_smallest (int): Number of smallest values to extract per class (distances to closes centroids).

    Returns:
        torch.Tensor: Tensor of shape (num_test_samples, num_classes, n_smallest) containing the smallest values per class.
    """
    class_list = torch.unique(labels, sorted=True).to(data.device)
    num_classes = len(class_list)

    # Initialize output tensor
    result = torch.full(
        (data.size(0), num_classes, n_smallest), float("inf"), device=data.device
    )

    for cls in range(num_classes):
        # Mask columns belonging to the current class
        mask = (labels == class_list[cls]).to(data.device)
        class_values = data[:, mask]

        if class_values.size(1) > 0:
            smallest_values, _ = torch.topk(
                class_values,
                k=min(n_smallest, class_values.size(1)),
                dim=1,
                largest=False,
            )
            result[:, cls, : smallest_values.size(1)] = smallest_values

    return result
