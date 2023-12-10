from text2emoji.models.grid_search_model import GridSearchModel
from text2emoji.models.test_model import test_best_model


def find_best_model():
    hyperparameters = {
        "dimensionality_reduction": ["pca"],
        "n_dimensions": [5, 10, 25, 50, 100],
        "n_layers": [1, 3, 5, 10],
        "n_neurons": [100, 200],
        "optimizer_type": ["adam", "sgd"],
        "learning_rate": [0.01, 0.001],
        "epochs": [1, 2, 5, 10, 20],
    }

    grid_search_model = GridSearchModel(hyperparameters, "word2vec")
    grid_search_model.run()
    print(grid_search_model.get_best_hyperparameters())

    grid_search_model.plot_loss_curve()
    grid_search_model.save_results()


if __name__ == "__main__":
    find_best_model()
    test_best_model("word2vec")
