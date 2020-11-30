"""
Utility functions for dimensionality reduction using PCA and Truncated SVD. Taken from work done in HW 5.
"""
import numpy.linalg as la
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
plt.rcParams.update({'font.size': 22})


def get_k_principal_components(X, k):
    """
    Calculate the matrix P of k principal components such that P_k = U_kS_k
    :param X: feature matrix
    :type X: pandas dataframe or numpy array
    :param k: number of principal components to keep
    :type k: int
    :return low_dim_X: k-subspace approximation of full feature matrix X
    """
    U, S, V = la.svd(X, full_matrices=False)
    sigma = np.diag(S)

    low_dim_X = U[:, :k] @ sigma[:k, :k] @ V[:k, :]

    return low_dim_X


def get_k_subspace_aprox(X, k):
    """
    Calculate the best subspace approximation using the truncated SVD.
    :param X: feature matrix
    :type X: pandas dataframe or numpy array
    :param k: number of principal components to keep
    :type k: int
    """
    U, S, V = la.svd(X, full_matrices=False)

    # Calculate pseudo-inverse 
    S_mat = np.diag(S)
    S_mat[k + 1:] = 0

    X_reconstructed = U @ S_mat @ V

    return X_reconstructed


def plot_spectrum(X):
    """
    Plot the spectrum (magnitude of all singular values)  of feature matrix X.
    """
    U, S, V = la.svd(X, full_matrices=False)
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.plot(range(S.shape[0]), S, 'b.', markersize=12)
    ax.set_title('Spectrum of X')
    ax.set_xlabel('Singular value number')
    ax.set_ylabel('Singular value magnitude')


def corr_pc_features(X, prin_comps):
    """ 
    Calculate correlation between original features X and principal components prin_comps.
    :param X: feature matrix
    :prin_comps: principal component vectors
    :return results: pd.DataFrame with correlations between vectors and principal components.
    """
    col_names = ["PC" + str(i+1) for i in range(prin_comps.shape[0])]
    results = pd.DataFrame(columns=col_names)

    for i in X:
        vec = X.loc[:, i].values
        row = []
        for pc in prin_comps:
            row.append(np.corrcoef(pc,vec)[0][1])

        results = results.append({col_names[j]:r for j,r in enumerate(row)}, 
                                 ignore_index=True)
    
    results = results.rename(index={i:feature for i, feature in enumerate(X.columns.values)})

    return results

def train_test_split_w(X,Y, train_size=.80, random_state=22):
    """
    Wrapper function for sklearn's train/test split function.
    """
    X_train, X_test, y_train, y_test = train_test_split(X,Y, train_size=train_size, random_state=random_state)
    return  X_train, X_test, y_train, y_test


def least_squares_classifier(X_train, X_test, y_train):
    """
    Perform standard least squares to find the optimal weights and use weightts to predict
    values using test set. Round to nearest -1, 1 to get labels.

    :param X_train: feature matrix training set 
    :param X_test: feature matrix test set
    :param y_train: training set labels
    :return y_p: predicted labels vector
    """

    # Get weights and make predictions
    w_hat = la.inv(X_train.T @ X_train) @ X_train.T @ y_train
    y_p = np.sign(X_test @ w_hat)

    return y_p

