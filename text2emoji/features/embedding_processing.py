import pandas as pd
from sklearn.decomposition import PCA
from umap import UMAP
from imblearn.over_sampling import RandomOverSampler


def balance_data(features, target, balance_technique, n_samples=None):
    """
    Balance a dataset by either undersampling or oversampling the majority class\
    If no technique is specified, the dataset is not balanced

    Args:
        features (np.array): The features
        target (np.array): The target
        n_samples (int): The number of samples to take from the majority class (default: median)

    Returns:
        features_resampled, target_resampled (np.array, np.array): The balanced features and target, respectively
    """
    df = pd.DataFrame(features)
    df["target"] = target

    if balance_technique == "none":

        return features, target

    elif balance_technique == "undersample":

        if n_samples is None:
            n_samples = df.target.value_counts().median().astype(int)

        # Balance dataset
        df = df.groupby("target").apply(lambda x: x.sample(min(len(x), n_samples)))

        features_resampled = df.drop("target", axis=1).values
        target_resampled = df.target.values

        return features_resampled, target_resampled

    elif balance_technique == "oversample":
        oversampler = RandomOverSampler()
        features_resampled, target_resampled = oversampler.fit_resample(features, target)
        return features_resampled, target_resampled

    else:
        raise ValueError(f"Unknown balancing technique: {balance_technique}")


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
    elif technique == "none":
        features_reduced = features
    else:
        raise ValueError(f"Unknown dimensionality reduction technique: {technique}")

    return features_reduced
