from text2emoji.models.grid_search_model import GridSearchModel


def find_best_model():
    hyperparameters = {
        "dimensionality_reduction": ["pca"],
        "n_dimensions": [10, 25, 50],
        "n_layers": [1, 2],
        "n_neurons": [20, 50],
        "optimizer_type": ["adam", "sgd"],
        "learning_rate": [0.01, 0.001],
        "epochs": [5, 10, 20],
    }

    grid_search_model = GridSearchModel(hyperparameters, "word2vec")
    grid_search_model.run()
    print(grid_search_model.get_best_hyperparameters())

    grid_search_model.save_results()


if __name__ == "__main__":

    find_best_model()
