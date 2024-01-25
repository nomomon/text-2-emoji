import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm

from sklearn.metrics import (
    top_k_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

from text2emoji.models.bootstrap import bootstrap


def eval_best_model(type, eval_set="valid"):
    # Load data
    prefix = "unfrozen_transformer_" if type == "unfrozen_bert" else ""
    df = pd.read_csv(f"data/silver/{prefix}{eval_set}.csv")
    
    eval_labels = df["label"].values
    eval_text = df["text"].values

    probabilities = []
    for text in tqdm(eval_text):
        api_url = "http://127.0.0.1:8000/get_emoji"
        url_text = text.replace(" ", "%20")
        usr_params = {"text": url_text, "embedding_type": type}
        try:
            data = requests.get(api_url, params=usr_params).json()
        except Exception as e:
            print(e)
            return None

        df_data = pd.json_normalize(data["results"])
        df_data = df_data.sort_values(by="label", ascending=True)
        probabilities.append(df_data["probability"].values)
    probabilties = np.array(probabilities)

    # Get baseline accuracy (most frequent label)
    most_freq_labels = np.full(len(eval_labels), np.argmax(np.bincount(eval_labels)))
    most_freq_one_hot = np.zeros((len(eval_labels), len(np.unique(eval_labels))))
    most_freq_one_hot[np.arange(len(eval_labels)), most_freq_labels] = 1
    
    # Get baseline accuracy (random, with same distribution as labels)
    random_labels = np.random.choice(eval_labels, len(eval_labels))
    random_one_hot = np.zeros((len(eval_labels), len(np.unique(eval_labels))))
    random_one_hot[np.arange(len(eval_labels)), random_labels] = 1

    results = pd.DataFrame(columns=[
        "accuracy", "top_3_accuracy", "top_5_accuracy", 
        "macro f1_score", "macro precision", "macro recall"
    ])
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

    with open(f"out/{type}/results_{eval_set}.txt", "w") as f:
        f.write(results.to_string())

    conf_matrix = confusion_matrix(eval_labels, probabilties.argmax(axis=1), normalize="true")
    plt.figure(figsize=(10, 10))
    plt.imshow(conf_matrix, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Normalized confusion matrix for {eval_set} set ({type})")
    labels = ["❤", "😍", "😂", "💕", "🔥", "😊", "😎", "✨", "💙", "😘", "📷", "🇺🇸", "☀", "💜", "😉", "💯", "😁", "🎄", "📸", "😜"]
    plt.xticks(range(len(labels)), labels)
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
    plt.savefig(f"out/{type}/confusion_matrix_{eval_set}.png")

    signif = 0.05
    p_val_most_freq = bootstrap(probabilties.argmax(axis=1), most_freq_labels, eval_labels)
    p_val_random = bootstrap(probabilties.argmax(axis=1), random_labels, eval_labels)

    results = \
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

    with open(f"out/{type}/bootstrap_{eval_set}.txt", "w") as f:
        f.write(results)