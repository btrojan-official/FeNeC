import torch
from utils.other import _get_single_class_examples

def _calc_single_covariance(X_train, y_train, class_number, device):

    single_class_examples = _get_single_class_examples(X_train, y_train, class_number, device)
    
    return torch.cov(single_class_examples.T).to(device)

def _matrix_shrinkage(cov_matrix, alpha_0, alpha_1, device):

    assert cov_matrix.shape[0] == cov_matrix.shape[1], "Covariance matrix must be square"

    diag_mean = torch.mean(torch.diag(cov_matrix))
    off_diag = cov_matrix.clone()
    off_diag.fill_diagonal_(0.0)
    I = torch.eye(cov_matrix.shape[0], device=cov_matrix.device)
    mask = I == 0.0 # off_diag != 0.0 # 
    off_diag_mean = (off_diag*mask).sum() / mask.sum()
    shrinkaged_cov_matrix = cov_matrix + (alpha_0 * diag_mean * I) + (alpha_1 * off_diag_mean * (1 - I))

    return shrinkaged_cov_matrix.clone().detach().to(device)

def _normalize_covariance_matrix(cov_matrix):

    diag_elements = torch.sqrt(torch.diag(cov_matrix))
    
    outer_diag = torch.outer(diag_elements, diag_elements)

    normalized_cov_matrix = cov_matrix / outer_diag
    
    return normalized_cov_matrix

def _tukeys_transformation(x, tukey_lambda):
    if tukey_lambda != 0:
        return torch.pow(x, tukey_lambda)
    else:
        return torch.log(x)