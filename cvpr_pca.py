import numpy as np

def pca_dimen_reduc(features, k=0):
    # Perform PCA on all descriptors
    mean_F = np.mean(features, axis=0)
    centered_F = features - mean_F
    # Compute the covariance matrix
    cov_mat = np.cov(centered_F, rowvar=False)
    # Decompose the covariance matrix into eigenvectors and eigenvalues
    val, vct = np.linalg.eig(cov_mat)
    # Sort the eigenvalues and eigenvectors
    sorted_indices = np.argsort(val)[::-1]
    # sorted_val = val[sorted_indices]
    sorted_vct = vct[:, sorted_indices]
    # Select the top k eigenvectors
    projection_matrix = sorted_vct[:, :k]
    # Project the data onto the new space
    pca_result = np.dot(centered_F, projection_matrix)
    
    return pca_result