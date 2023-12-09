import pandas as pd
from sklearn.decomposition import PCA
from umap import UMAP


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


def reduce_dimensions(features, n_dimensions, technique="pca"):
    """
    Reduce the number of dimensions using PCA (for now at least)

    Args:
        features (np.array): The features
        n_dimensions (int): The number of dimensions to reduce to

    Returns:
        features_reduced (np.array): The reduced np.array of features `shape == (-1, n_dimensions)`
    """

    # Reduce dimensions
    if technique == "pca":
        pca = PCA(n_components=n_dimensions)
        features_reduced = pca.fit_transform(features)
    elif technique == "umap":
        reducer = UMAP(n_components=n_dimensions, n_neighbors=10, n_jobs=-1)
        features_reduced = reducer.fit_transform(features)
    else:
        raise ValueError(f"Unknown dimensionality reduction technique: {technique}")

    return features_reduced
