from text2emoji.models.grid_search_model import GridSearchModel
from text2emoji.models.grid_search_transformer import TransformerGridSearch
from text2emoji.models.eval_model import eval_best_model


def classifier_model(model_type):
    """
    Trains a classifier model which operates on the embeddings.

    Args:
        model_type (string): The type of classifier model to train.

    Returns:
        GridSearchModel: A grid search model.
    """

    hyperparameters = {
        "dimensionality_reduction": ["none"],
        "n_dimensions": [1],
        "balancing_technique": ["none", "undersample", "oversample"],
        "n_layers": [0, 1, 2, 3],
        "n_neurons": [10, 50, 100],
        "optimizer_type": ["adam"],
        "learning_rate": [1e-1, 1e-2, 1e-3],
        "epochs": [80, 100, 125, 150, 175, 200],
    }

    return GridSearchModel(hyperparameters, model_type)


def transformer_model(model_type):
    """
    Trains a transformer model which operates on the raw text.

    Args:
        model_type (string): The type of transformer model to train.

    Returns:
        TransformerGridSearch: A variant of the grid search model for transformers.
    """

    # Be careful with too many hyperparameters as there might not be enough memory
    hyperparameters = {
        "learning_rate": [1e-5, 2e-5, 5e-5],
        "dropout": [0.1, 0.3],
    }

    return TransformerGridSearch(hyperparameters, model_type)


def find_best_model(model_type):
    """
    Finds the best model for the given model type.

    Args:
        model_type (string): The type of model to train.
    """

    if model_type in ["word2vec", "mobert"]:
        model = classifier_model(model_type)
    else:
        model = transformer_model(model_type)

    model.run()
    print(model.get_best_hyperparameters())
    model.plot_loss_curve()
    model.save_results()


# Supported model types
MODEL_TYPES = ["word2vec", "mobert", "unfrozen_bert"]

if __name__ == "__main__":

    model_type = "unfrozen_bert"
    find_best_model(model_type)
    # eval_best_model(model_type, "valid")
