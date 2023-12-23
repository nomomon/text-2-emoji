from text2emoji.models.grid_search_model import GridSearchModel
from text2emoji.models.eval_model import eval_best_model


def find_best_model(model_type):

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

    grid_search_model = GridSearchModel(hyperparameters, model_type)
    grid_search_model.run()
    print(grid_search_model.get_best_hyperparameters())

    grid_search_model.plot_loss_curve()
    grid_search_model.save_results()


if __name__ == "__main__":
    model_type = "word2vec"
    find_best_model(model_type)
    # eval_best_model(model_type, "valid")
