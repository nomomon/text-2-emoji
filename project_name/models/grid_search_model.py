import pandas as pd
import numpy as np
from tqdm import tqdm

import itertools

from project_name.features.embedding_processing import balance_data, reduce_dimensions_pca
from project_name.models.nn_classifier import get_model, get_optimizer, train_model


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

        # Load data
        self.train_features = np.load(f"data/gold/train_{embedding_type}_features.npy")
        self.valid_features = np.load(f"data/gold/valid_{embedding_type}_features.npy")
        self.train_target = np.load(f"data/gold/train_{embedding_type}_target.npy")
        self.valid_target = np.load(f"data/gold/valid_{embedding_type}_target.npy")

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
        hyperparameters = pd.Series(hyperparameters)

        # Add results to dataframe
        self.results.loc[len(self.results)] = hyperparameters

    def run(self, verbose=True):
        """
        Run the grid search by exhaustively iterating over all hyperparameter combinations
        """

        # Create all combinations of hyperparameters
        hyperparameter_combinations = list(itertools.product(*self.hyperparameters.values()))

        if verbose:
            hyperparameter_combinations = tqdm(hyperparameter_combinations, desc="Hyperparameter search")

        # Iterate over all combinations
        for hyperparameter_combination in hyperparameter_combinations:

            # Unpack hyperparameters
            n_dimensions, n_layers, n_neurons, optimizer_type, learning_rate, epochs = hyperparameter_combination

            # Reduce dimensions of data
            reduced_train_features = reduce_dimensions_pca(self.train_features, n_dimensions)
            reduced_valid_features = reduce_dimensions_pca(self.valid_features, n_dimensions)

            # Balance training data
            balanced_train_features, balanced_train_target = balance_data(reduced_train_features, self.train_target)

            # Create model
            model = get_model(n_dimensions, n_layers, n_neurons)
            optimizer = get_optimizer(model, optimizer_type, learning_rate)

            # Train model
            accuracy = train_model(
                model, optimizer, epochs, 
                train_features = balanced_train_features, 
                train_target   = balanced_train_target,
                valid_features = reduced_valid_features,
                valid_target   = self.valid_target)

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
