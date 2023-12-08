from project_name.models.grid_search_model import GridSearchModel

if __name__ == "__main__":

    hyperparameters = {
        "n_dimensions": [10, 50],
        "n_layers": [1, 2],
        "n_neurons": [10, 50],
        "optimizer_type": ["adam", "sgd"],
        "learning_rate": [0.01, 0.001],
        "epochs": [5, 10, 20],
    }

    grid_search_model = GridSearchModel(hyperparameters, "word2vec")
    grid_search_model.run()
    print(grid_search_model.get_best_hyperparameters())
