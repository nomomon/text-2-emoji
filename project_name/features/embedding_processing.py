import pandas as pd
import numpy as np
from sklearn.decomposition import PCA


def balance_data(features, target, n_samples=None):
    """
    Balance a dataset by taking 
    # TODO: add description

    Args:
        features (np.array): The features
        target (np.array): The target
        n_samples (int): The number of samples to take from the majority class (default: median)

    Returns:
        features_resampled, target_resampled (np.array, np.array): The balanced features and target, respectively
    """
    df = pd.DataFrame(features)
    df["target"] = target

    if n_samples is None:
        n_samples = df.target.value_counts().median().astype(int)
    
    # Balance dataset
    df = df.groupby("target").apply(lambda x: x.sample(min(len(x), n_samples)))
    
    features_resampled = df.drop("target", axis=1).values
    target_resampled = df.target.values

    return features_resampled, target_resampled


def reduce_dimensions_pca(features, n_dimensions):
    """
    Reduce the number of dimensions using PCA (for now at least)

    Args:
        features (np.array): The features
        n_dimensions (int): The number of dimensions to reduce to

    Returns:
        features_reduced (np.array): The reduced np.array of features `shape == (-1, n_dimensions)`
    """

    # Reduce dimensions
    pca = PCA(n_components=n_dimensions)
    features_reduced = pca.fit_transform(features)

    return features_reduced
