import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from text2emoji.features.embedding_processing import reduce_dimensions
from text2emoji.models.bootstrap import bootstrap
from text2emoji.models.nn_classifier import get_probabilities
from sklearn.metrics import (
    top_k_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

def eval_best_model(type, eval_set="valid"):
    # Load evaluation embeddings
    eval_features = np.load(f"data/gold/{eval_set}_{type}_features.npy")
    eval_labels = np.load(f"data/gold/{eval_set}_{type}_target.npy")

    # Load the number of dimensions and technique used for dimensionality reduction
    df = pd.read_csv("out/grid_search_results.csv")

    # dimensionality_reduction ,n_dimensions, get these fields from the df
    reduction_technique = df["dimensionality_reduction"].iloc[0]
    n_dimensions = df["n_dimensions"].iloc[0]

    # Reduce dimensions
    eval_embeddings = reduce_dimensions( 
        eval_features, n_dimensions, reduction_technique
    )


    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load best model
    
    best_model = torch.load("out/best_model.pt", map_location=torch.device('cpu'))
    best_model.to(device)

    # Get predictions of the model
    probabilties = get_probabilities(
        best_model, torch.tensor(eval_embeddings).to(device)
    )
    
    # Get baseline accuracy (most frequent label)
    most_freq_labels = np.full(len(eval_labels), np.argmax(np.bincount(eval_labels)))
    most_freq_one_hot = np.zeros((len(eval_labels), len(np.unique(eval_labels))))
    most_freq_one_hot[np.arange(len(eval_labels)), most_freq_labels] = 1
    
    # Get baseline accuracy (random, with same distribution as labels)
    random_labels = np.random.choice(eval_labels, len(eval_labels))
    random_one_hot = np.zeros((len(eval_labels), len(np.unique(eval_labels))))
    random_one_hot[np.arange(len(eval_labels)), random_labels] = 1

    results = pd.DataFrame(columns=["accuracy", "top_3_accuracy", "top_5_accuracy", "macro f1_score", "macro precision", "macro recall"])
    results.loc["most_freq"] = [
        np.mean(most_freq_labels == eval_labels),
        top_k_accuracy_score(eval_labels, most_freq_one_hot, k=3),
        top_k_accuracy_score(eval_labels, most_freq_one_hot, k=5),
        f1_score(eval_labels, most_freq_labels, average="macro"),
        precision_score(eval_labels, most_freq_labels, average="macro"),
        recall_score(eval_labels, most_freq_labels, average="macro"),
    ]
    results.loc["random"] = [
        np.mean(random_labels == eval_labels),
        top_k_accuracy_score(eval_labels, random_one_hot, k=3),
        top_k_accuracy_score(eval_labels, random_one_hot, k=5),
        f1_score(eval_labels, random_labels, average="macro"),
        precision_score(eval_labels, random_labels, average="macro"),
        recall_score(eval_labels, random_labels, average="macro"),
    ]
    results.loc["model"] = [
        np.mean(probabilties.argmax(axis=1) == eval_labels),
        top_k_accuracy_score(eval_labels, probabilties, k=3),
        top_k_accuracy_score(eval_labels, probabilties, k=5),
        f1_score(eval_labels, probabilties.argmax(axis=1), average="macro"),
        precision_score(eval_labels, probabilties.argmax(axis=1), average="macro"),
        recall_score(eval_labels, probabilties.argmax(axis=1), average="macro"),
    ]

    print("Evaluated on", eval_set, "set")
    print(results.to_latex(float_format="%.3f"))

    conf_matrix = confusion_matrix(eval_labels, probabilties.argmax(axis=1), normalize="true")
    plt.figure(figsize=(10, 10))
    plt.imshow(conf_matrix, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Normalized confusion matrix for {eval_set} set")
    labels = ["❤", "😍", "😂", "💕", "🔥", "😊", "😎", "✨", "💙", "😘", "📷", "🇺🇸", "☀", "💜", "😉", "💯", "😁", "🎄", "📸", "😜"]
    plt.xticks(range(len(labels)), labels, rotation=90)
    plt.yticks(range(len(labels)), labels)
    plt.colorbar()
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.annotate(
                f"{conf_matrix[i, j]:.2f}",
                xy=(j, i),
                horizontalalignment="center",
                verticalalignment="center",
                c = "white" if conf_matrix[i, j] > 0.5 else "black"
            )
    plt.tight_layout()
    plt.savefig(f"out/confusion_matrix_{eval_set}.eps")
    plt.savefig(f"out/confusion_matrix_{eval_set}.png")

    signif = 0.05
    p_val_most_freq = bootstrap(probabilties.argmax(axis=1), most_freq_labels, eval_labels)
    p_val_random = bootstrap(probabilties.argmax(axis=1), random_labels, eval_labels)

    print(
f"""
Bootstrap test for model vs most frequent
    - H0: model acc == most frequent acc
    - H1: model acc >= most frequent acc + 2 * observed_delta

    p-value: {p_val_most_freq:.5f}
    significance level: {signif}
    p-value < significance level: {p_val_most_freq < signif}
    {"reject H0" if p_val_most_freq < signif else "do not reject H0"}


Bootstrap test for model vs random
    - H0: model acc == random acc
    - H1: model acc >= random acc + 2 * observed_delta

    p-value: {p_val_random:.5f}
    significance level: {signif}
    p-value < significance level: {p_val_random < signif}
    {"reject H0" if p_val_random < signif else "do not reject H0"}
"""
    )

