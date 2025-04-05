import torch

from utils.other import get_single_class_examples


def calc_single_covariance(X_train, y_train, class_number, device):
    """
    Calculates the covariance matrix for a specific class from the training data.

    Args:
        X_train (torch.Tensor): The input data of shape (num_samples, num_features).
        y_train (torch.Tensor): The labels of shape (num_samples,).
        class_number (int): The class label for which covariance is calculated.
        device (torch.device): The device (CPU/GPU) to store the output tensor.

    Returns:
        torch.Tensor: The covariance matrix of the selected class with shape (num_features, num_features).
    """

    single_class_examples = get_single_class_examples(
        X_train, y_train, class_number, device
    )

    return torch.cov(single_class_examples.T).to(device)


def matrix_shrinkage(cov_matrix, alpha_0, alpha_1, device):
    """
    Applies shrinkage to the covariance matrix to improve its stability.

    Args:
        cov_matrix (torch.Tensor): The covariance matrix to be shrinked, must be square (num_features x num_features).
        alpha_0 (float): Shrinkage parameter for the diagonal elements.
        alpha_1 (float): Shrinkage parameter for the off-diagonal elements.
        device (torch.device): The device (CPU/GPU) to store the output tensor.

    Returns:
        torch.Tensor: The shrinked covariance matrix with the same shape as the input covariance matrix.
    
    Raises:
        AssertionError: If the covariance matrix is not square.
    """

    assert (
        cov_matrix.shape[0] == cov_matrix.shape[1]
    ), "Covariance matrix must be square"

    diag_mean = torch.mean(torch.diag(cov_matrix))
    off_diag = cov_matrix.clone()
    off_diag.fill_diagonal_(0.0)
    I = torch.eye(cov_matrix.shape[0], device=cov_matrix.device)
    mask = I == 0.0  # off_diag != 0.0 #
    off_diag_mean = (off_diag * mask).sum() / mask.sum()
    shrinkaged_cov_matrix = (
        cov_matrix + (alpha_0 * diag_mean * I) + (alpha_1 * off_diag_mean * (1 - I))
    )

    return shrinkaged_cov_matrix.clone().detach().to(device)


def normalize_covariance_matrix(cov_matrix):
    """
    Normalizes the covariance matrix by dividing it by the outer product of its diagonal elements.

    Args:
        cov_matrix (torch.Tensor): The covariance matrix to be normalized, shape (num_features, num_features).

    Returns:
        torch.Tensor: The normalized covariance matrix with the same shape as the input covariance matrix.
    """

    diag_elements = torch.sqrt(torch.diag(cov_matrix))

    outer_diag = torch.outer(diag_elements, diag_elements)

    normalized_cov_matrix = cov_matrix / outer_diag

    return normalized_cov_matrix


def tukeys_transformation(x, tukey_lambda):
    """
    Applies Tukey's transformation to the input tensor.

    Args:
        x (torch.Tensor): The input data tensor.
        tukey_lambda (float): The transformation exponent. If 0, applies a logarithmic transformation.

    Returns:
        torch.Tensor: The transformed tensor.
    """
    
    if tukey_lambda != 0:
        return torch.pow(x, tukey_lambda)
    else:
        return torch.log(x)
