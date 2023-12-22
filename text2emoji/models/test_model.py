import torch
import numpy as np
import pandas as pd

from text2emoji.features.embedding_processing import reduce_dimensions
from text2emoji.models.nn_classifier import get_performance


def test_best_model(type):

    # Load best model
    best_model = torch.load("out/best_model.pt")

    # Load test embeddings
    test_features = np.load(f"data/gold/test_{type}_features.npy")
    test_labels = np.load(f"data/gold/test_{type}_target.npy")

    # Load the number of dimensions and technique used for dimensionality reduction
    df = pd.read_csv("out/grid_search_results.csv")

    # dimensionality_reduction ,n_dimensions, get these fields from the df
    reduction_technique = df["dimensionality_reduction"].iloc[0]
    n_dimensions = df["n_dimensions"].iloc[0]

    # Reduce dimensions
    test_embeddings = reduce_dimensions(
        test_features, n_dimensions, reduction_technique
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    best_model.to(device)

    # Get accuracy
    accuracy, loss = get_performance(
        best_model, torch.tensor(test_embeddings).to(device), torch.tensor(test_labels).to(device)
    )

    print(f"Accuracy: {accuracy:.2f}")

    # Find the most frequent label
    unique_labels, counts = np.unique(test_labels, return_counts=True)
    most_frequent_label = unique_labels[np.argmax(counts)]

    # Find the accuracy of a model that always predicts the most frequent label
    baseline_accuracy = (test_labels == most_frequent_label).mean()
    print(f"Baseline accuracy: {baseline_accuracy}")


if __name__ == "__main__":
    test_best_model("word2vec")