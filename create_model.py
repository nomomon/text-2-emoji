from text2emoji.models.grid_search_model import GridSearchModel


def find_best_model():
    hyperparameters = {
        "dimensionality_reduction": ["pca"],
        "n_dimensions": [100],
        "n_layers": [5],
        "n_neurons": [100],
        "optimizer_type": ["adam"],
        "learning_rate": [0.005],
        "epochs": [400],
    }

    grid_search_model = GridSearchModel(hyperparameters, "word2vec")
    grid_search_model.run()
    print(grid_search_model.get_best_hyperparameters())

    grid_search_model.plot_loss_curve()
    grid_search_model.save_results()


if __name__ == "__main__":

    find_best_model()
