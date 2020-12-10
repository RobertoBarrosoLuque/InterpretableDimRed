"""
Utility functions for dimensionality reduction using PCA and Truncated SVD. Taken from work done in HW 5.
"""
import numpy.linalg as la
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
plt.rcParams.update({'font.size': 22})
import seaborn as sns


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

def plot_feature_pca_corr(X, princip_comp, n):
    """
    Plot bargraphs to visualize the features with the largest correlations to the principal component vector.
    """

    results = corr_pc_features(X, princip_comp)

    fig, axes = plt.subplots(figsize=(18, 10))

    for i, cols in enumerate(results.columns):

        results = corr_pc_features(X, princip_comp)

        for i, cols in enumerate(results.columns):

            sortd = results.loc[:,[cols]].sort_values(by=cols, ascending=False)
            sortd.iloc[:n,:].plot(kind="bar", ax=axes)


    fig.suptitle('Correlation between PCA vectors and original features (top {})'.format(n), size=15)
    plt.show()


def normalize(dataset, features): 
    '''
    Normalizes continuous variables in a training and testing dataset using
    the mean and the standard deviation from solely the training set. Uses
    the population estimates with zero degrees of freedom.
    
    Inputs: 
        - train_df: (pandas dataframe) a dataframe containing the training set
        - test_df: (pandas dataframe) a dataframe containing the testing set
        - features: (lst) a list of column names that should be normalized
        
    Returns: 
        - train_norm_df: (pandas dataframe) a dataframe containing the training
                          set with the features normalized
        - test_norm_df: (pandas dataframe) a dataframe containing the testing
                         set with the features normalized
    '''
    data_norm = dataset.copy()
    
    for col in features: 
        col_mean = dataset[col].mean()
        col_std = dataset[col].std(ddof=0)
        data_norm[col] = (dataset[col] - col_mean) / col_std
        
    return data_norm


def dummy_cat_vars(df, features, separator="_"):
    '''
    One-hot encodes categorical variables in a data set

    Inputs: 
        - df: (pandas df) either a training or testing df
        - features: (lst) the categorical features to encode as dummies
        - separator: (str) the string to connect the feature name as a prefix
                      on the feature value, defaults to "_"

    Returns: 
        - df: (pandas df) a modified dataframe with dummy variables included
    '''
    df = pd.get_dummies(df, columns=features, prefix_sep=separator)
    df.columns = df.columns.str.lower()
    df.columns = df.columns.str.replace(" ", "_")

    return df


def squared_error(y_hat, true_y):
    '''
    Calculates the squared error on a prediction problem
    
    Inputs: 
        y_hat (1-D numpy array): a vector of predictions from the classifier
        true_y (1-D numpy array): a vector of true labels from the classifier
    
    Returns:
        error_rate (float): the number of errors divided by number of samples
    '''
    num_y = len(true_y)
    error_rate = np.sum((y_hat - true_y)**2) / num_y
    
    return error_rate


def c_and_g_algo(PC_vector):
    '''
    Runs the following algorithm as described by Chipman and Gu. Uses squared 
    distance as a measure of closeness rather than angle closeness. 
    
    Takes the non-zero elements of the PC direction vector. 
    For K = 1 through all the non zero elements,
    Takes the k elements with the largest absolute value and sets them equal to
        plus or minus one / square root of k
    Sets all the other elements equal to zero
    Finds the vector with k elements closest to the original PC direction vector
    
    Inputs: 
        PC_vector (array) One principle component direction vector
        
    Returns: 
        closest (array) One interpretable component direction vector
        factor (float): the norm of the closest component direction vector
    '''
    closest = np.zeros((len(PC_vector)))
    factor = 1
    p = np.count_nonzero(PC_vector)
    
    signs = np.sign(PC_vector)   
    in_order = np.sort(np.abs(PC_vector.copy()))
    
    for k in range(1, p):
        magnitude = np.abs(PC_vector)
        largest = in_order[-k:]
        
        for element in largest:
            magnitude[magnitude == element] = 1
        magnitude[magnitude != 1] = 0
        
        interp = magnitude * signs 
        interp_factor = np.sqrt(k)
        
        distance = squared_error((inter / interp_factor), PC_vector)
        prev_distance = squared_error(closest, PC_vector)
        
        if distance < prev_distance:
            closest = interp
            factor = interp_factor
            
    return closest, factor