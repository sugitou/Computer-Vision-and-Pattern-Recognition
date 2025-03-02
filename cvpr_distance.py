import numpy as np

def cvpr_distance(F1, F2, metric, inv_cov_mat=None):
    if metric == 'Euclidean':
        dst = np.linalg.norm(F1 - F2)
    elif metric == 'Mahalanobis':
        diff = F1 - F2
        mdist_squared = diff.T @ inv_cov_mat @ diff
        dst = np.sqrt(mdist_squared)
    elif metric == 'Manhattan':
        dst = np.sum(np.abs(F1 - F2))
    else:
        raise ValueError('Invalid metric')
    return dst