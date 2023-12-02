import pandas as pd
import itertools
from project_name.features.embedding_processing import balance_dataframe, reduce_dimensions_pca
from project_name.models.nn_classifier import get_model, get_optimizer, train_model
from tqdm import tqdm


class GridSearchModel():
    """
    A model that performs a grid search over a set of hyperparameters using a neural network
    """

    # Use list for keys
    hyperparameters_keys = ["n_dimensions", "n_layers", "n_neurons", "optimizer_type", "learning_rate", "epochs"]

    def __init__(self, hyperparameters, embedding_type):
        """
        Initialize the model, load data and create results dataframe
        """

        # Load data, change to real embeddings later
        self.train_data = pd.read_csv(f"data/gold/train_{embedding_type}_embeddings.csv")
        self.val_data = pd.read_csv(f"data/gold/val_{embedding_type}_embeddings.csv")

        # Create results dataframe
        self.results = pd.DataFrame(columns=self.hyperparameters_keys + ["accuracy"])

        self.hyperparameters = hyperparameters

        # Assert if keys are in hyperparameters
        assert set(self.hyperparameters.keys()) == set(self.hyperparameters_keys)

    def add_result(self, hyperparameter_combination, accuracy):
        """
        Adds the result of a hyperparameter combination

        Args:
            hyperparameter_combination (tuple): The hyperparameter combination
            accuracy (float): The accuracy of the model with the hyperparameters
        """

        # Create a dictionary of hyperparameters and accuracy
        hyperparameters = dict(zip(self.hyperparameters.keys(), hyperparameter_combination))
        hyperparameters["accuracy"] = accuracy

        # Add results to dataframe, use concat
        self.results = pd.concat([self.results, pd.DataFrame(hyperparameters, index=[0])])

    def run(self, verbose=True):
        """
        Run the grid search by exhaustively iterating over all hyperparameter combinations
        """

        # Create all combinations of hyperparameters
        hyperparameter_combinations = list(itertools.product(*self.hyperparameters.values()))

        if verbose:
            hyperparameter_combinations = tqdm(hyperparameter_combinations)

        # Iterate over all combinations
        for hyperparameter_combination in hyperparameter_combinations:

            # Unpack hyperparameters
            n_dimensions, n_layers, n_neurons, optimizer_type, learning_rate, epochs = hyperparameter_combination

            # Balance and reduce dimensions of data
            reduced_train_data = reduce_dimensions_pca(self.train_data, n_dimensions)
            balanced_train_data = balance_dataframe(reduced_train_data)
            reduced_val_data = reduce_dimensions_pca(self.val_data, n_dimensions)

            # Create model
            model = get_model(n_dimensions, n_layers, n_neurons)
            optimizer = get_optimizer(model, optimizer_type, learning_rate)

            # Train model
            accuracy = train_model(model, optimizer, epochs, balanced_train_data, reduced_val_data)

            # Save results
            self.add_result(hyperparameter_combination, accuracy)

        # Save results to csv
        self.results.to_csv("out/grid_search_results.csv", index=False)

    def get_best_hyperparameters(self):
        """
        Get the best hyperparameter combination

        Returns:
            tuple: The best hyperparameter combination
        """

        # Get index of best result
        best_index = self.results["accuracy"].idxmax()

        # Get best combination
        best_combination = self.results.iloc[best_index]

        return best_combination
