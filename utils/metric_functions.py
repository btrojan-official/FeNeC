import torch
import torch.nn.functional as F

from utils.covMatrices_operations import _tukeys_transformation


def _euclidean(X_train, X_test, device, training_batch_size=10000):
        
    X_test = X_test.clone().to(device)
    test_squared_norms = torch.sum(X_test ** 2, dim=1).unsqueeze(1)

    for i in range(0, X_train.shape[0], training_batch_size): # nie wiem czy przy batch_size które nie jest dzielnikiem X_train.shape[0] weźmie wszystkie przykłady pod uwagę
        X_train_batch = X_train[i:i + training_batch_size, :].clone().to(device)

        train_squared_norms = torch.sum(X_train_batch ** 2, dim=1).unsqueeze(0)

        dot_product = torch.mm(X_test, X_train_batch.t())

        dists_squared = test_squared_norms + train_squared_norms - 2 * dot_product

        dists_squared = torch.clamp(dists_squared, min=0.0)

        if i == 0:
            dists = torch.sqrt(dists_squared)
        else:
            dists = torch.cat((dists, torch.sqrt(dists_squared)), dim=1)

    return dists

def _mahalanobis(X_train, y_train, X_test, covMatrices, tukey_lambda, device, norm_in_mahalanobis=True, batch_size=16):
    EPSILON = 1e-8

    X_test = X_test.to(device)
    X_test = (X_test.T / (torch.linalg.norm(X_test.T, axis=0) + EPSILON)).T

    X_test = _tukeys_transformation(X_test, tukey_lambda)
    curr_X_train = _tukeys_transformation(X_train.clone().detach(), tukey_lambda)

    f_num = covMatrices.shape[1]
    num_classes = covMatrices.shape[0] // f_num

    cov_inv_list = []
    for i in range(num_classes):  
        cov_inv = torch.linalg.pinv(covMatrices[i * f_num:(i + 1) * f_num, :]).to(device)
        cov_inv_list.append(cov_inv)
    cov_inv_stack = torch.stack(cov_inv_list)

    mahalanobis_distances = []

    for start_idx in range(0, curr_X_train.shape[0], batch_size):
        end_idx = min(start_idx + batch_size, curr_X_train.shape[0])
        X_train_batch = curr_X_train[start_idx:end_idx].to(device)
        cov_inv_per_batch = cov_inv_stack[y_train[start_idx:end_idx]]
        X_test_exp = X_test.unsqueeze(0).repeat(end_idx - start_idx, 1, 1)
        if norm_in_mahalanobis:
            diff = (F.normalize(X_test_exp, p=2, dim=-1) - F.normalize(X_train_batch.unsqueeze(1), p=2, dim=-1)).float()
        else:
            diff = X_test_exp - X_train_batch.unsqueeze(1)
        batch_distances = torch.sqrt(torch.sum(diff @ cov_inv_per_batch * diff, dim=2))
        mahalanobis_distances.append(batch_distances)

    return torch.cat(mahalanobis_distances, dim=0).T.to(device)