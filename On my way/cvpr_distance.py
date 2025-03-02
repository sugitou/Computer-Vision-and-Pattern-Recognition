import numpy as np

def cvpr_distance(F1, F2, metric, inv_cov_mat=None):
    if metric == 'Euclidean':
        dst = np.linalg.norm(F1 - F2)
    elif metric == 'Mahalanobis':
        # cov_mat = np.cov(F1)
        # inv_cov_mat = np.linalg.inv(cov_mat)
        diff = F1 - F2
        # dst = np.sqrt(np.dot(np.dot(diff, inv_cov_mat), diff))
        mdist_squared = diff.T @ inv_cov_mat @ diff
        dst = np.sqrt(mdist_squared)
    elif metric == 'Cosine':
        dst = 1 - np.dot(F1, F2) / (np.linalg.norm(F1) * np.linalg.norm(F2))
    elif metric == 'Manhattan':
        dst = np.sum(np.abs(F1 - F2))
    else:
        raise ValueError('Invalid metric')
    return dst