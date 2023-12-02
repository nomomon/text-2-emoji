import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.decomposition import PCA


def balance_dataframe(df):
    """
    Balance a dataframe by undersampling the majority class

    Args:
        df (dataframe): The dataframe to balance

    Returns:
        dataframe: The balanced dataframe
    """

    # Separate features and labels
    features = df.drop(columns=["label"])
    labels = df["label"]

    # Balance dataset
    rus = RandomUnderSampler()
    features_resampled, labels_resampled = rus.fit_resample(features, labels)

    # Combine features and labels
    balanced_dataframe = pd.DataFrame(features_resampled)
    balanced_dataframe["label"] = labels_resampled

    return balanced_dataframe


def reduce_dimensions_pca(df, n_dimensions):
    """
    Reduce the number of dimensions of a dataframe using PCA (for now at least)

    Args:
        df (dataframe): The dataframe to reduce
        n_dimensions (int): The number of dimensions to reduce to

    Returns:
        dataframe: The reduced dataframe
    """

    # Separate features and labels
    features = df.drop(columns=["label"])
    labels = df["label"]

    # Reduce dimensions
    pca = PCA(n_components=n_dimensions)
    features_reduced = pca.fit_transform(features)

    # Combine features and labels
    reduced_dataframe = pd.DataFrame(features_reduced)
    reduced_dataframe["label"] = labels

    return reduced_dataframe
