import numpy as np


def mds(data, n_components=2):
    """
    Apply multidimensional scaling (aka Principal Coordinates Analysis)
    :param data: nxn square distance matrix
    :param n_components: number of components for projection
    :return: projected output of shape (n_components, n)
    """
    print("center")
    # Center distance matrix
    _center(data)
    print("eigen")
    # Make a list of (eigenvalue, eigenvector) tuples
    eig_val_cov, eig_vec_cov = np.linalg.eig(data)
    print("pairs")
    origin_eig_pairs = [
        (np.abs(eig_val_cov[i]), eig_vec_cov[:, i]) for i in range(len(eig_val_cov))
    ]

    # Select n_components eigenvectors with largest eigenvalues, obtain subspace transform matrix
    sorted_eig_pairs = np.array(
        sorted(origin_eig_pairs, key=lambda x: x[0], reverse=True))
    matrix_w = np.hstack(
        [sorted_eig_pairs[i, 1].reshape(data.shape[1], 1)
         for i in range(n_components)]
    )
    embeddings = np.dot(data, matrix_w)
    
    cr = np.asarray([eig_pair[0] / np.sum(np.abs(eig_val_cov)) for eig_pair in sorted_eig_pairs])
    # Return samples in new subspace
    return embeddings, matrix_w, cr

def _center(K):
    """
    Method to center the distance matrix
    :param K: numpy array of shape mxm
    :return: numpy array of shape mxm
    """
    n_samples = K.shape[0]

    # Mean for each row/column
    meanrows = np.sum(K, axis=0) / n_samples
    meancols = (np.sum(K, axis=1)/n_samples)[:, np.newaxis]

    # Mean across all rows (entire matrix)
    meanall = meanrows.sum() / n_samples

    K -= meanrows
    K -= meancols
    K += meanall
    return K
