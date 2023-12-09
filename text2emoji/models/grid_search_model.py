import pandas as pd
import numpy as np
from tqdm import tqdm
import torch

import itertools

from text2emoji.features.embedding_processing import balance_data, reduce_dimensions
from text2emoji.models.nn_classifier import get_model, get_optimizer, train_model


class GridSearchModel:
    """
    A model that performs a grid search over a set of hyperparameters using a neural network
    """

    # Use list for keys
    hyperparameters_keys = [
        "dimensionality_reduction",
        "n_dimensions",
        "n_layers",
        "n_neurons",
        "optimizer_type",
        "learning_rate",
        "epochs",
    ]

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
        self.results = pd.DataFrame(
            columns=self.hyperparameters_keys + ["accuracy", "model"]
        )

        self.hyperparameters = hyperparameters
        self.current_dimensions_reduction = None
        self.n_dimensions = None

        # Assert if keys are in hyperparameters
        assert set(self.hyperparameters.keys()) == set(self.hyperparameters_keys)

    def add_result(self, hyperparameter_combination, accuracy, model):
        """
        Adds the result of a hyperparameter combination

        Args:
            hyperparameter_combination (tuple): The hyperparameter combination
            accuracy (float): The accuracy of the model with the hyperparameters
            model (torch.nn.Sequential): The trained model
        """

        # Create a dictionary of hyperparameters and accuracy
        training_details = dict(
            zip(self.hyperparameters.keys(), hyperparameter_combination)
        )
        training_details["accuracy"] = accuracy
        training_details = pd.Series(training_details)
        training_details["model"] = model

        # Add results to dataframe
        self.results.loc[len(self.results)] = training_details

    def reduce_and_balance_data(self, dimensions_reduction, n_dimensions):
        """
        Reduce the dimensions of the data and balance the training data

        Args:
            dimensions_reduction (string): The dimensionality reduction technique to use
            n_dimensions (int): The number of dimensions to reduce to

        Returns:
            tuple: The balanced training features, balanced training target and reduced validation features
        """

        # Reduce dimensions of data
        reduced_train_features = reduce_dimensions(
            self.train_features, n_dimensions, dimensions_reduction
        )
        reduced_valid_features = reduce_dimensions(
            self.valid_features, n_dimensions, dimensions_reduction
        )

        # Balance training data
        balanced_train_features, balanced_train_target = balance_data(
            reduced_train_features, self.train_target
        )

        return balanced_train_features, balanced_train_target, reduced_valid_features

    def run(self, verbose=True):
        """
        Run the grid search by exhaustively iterating over all hyperparameter combinations
        """

        # Create all combinations of hyperparameters
        hyperparameter_combinations = list(
            itertools.product(*self.hyperparameters.values())
        )

        if verbose:
            hyperparameter_combinations = tqdm(
                hyperparameter_combinations, desc="Hyperparameter search"
            )

        # Iterate over all combinations
        for hyperparameter_combination in hyperparameter_combinations:

            # Unpack hyperparameters
            (
                dimensions_reduction,
                n_dimensions,
                n_layers,
                n_neurons,
                optimizer_type,
                learning_rate,
                epochs,
            ) = hyperparameter_combination

            # Only balance and reduce data when the technique and number of dimensions changed
            if (n_dimensions != self.n_dimensions) or (dimensions_reduction != self.current_dimensions_reduction):

                (
                    balanced_train_features,
                    balanced_train_target,
                    reduced_valid_features,
                ) = self.reduce_and_balance_data(dimensions_reduction, n_dimensions)

                self.current_dimensions_reduction = dimensions_reduction
                self.n_dimensions = n_dimensions

            # Create model
            model = get_model(n_dimensions, n_layers, n_neurons)
            optimizer = get_optimizer(model, optimizer_type, learning_rate)

            # Train model
            accuracy = train_model(
                model,
                optimizer,
                epochs,
                train_features=balanced_train_features,
                train_target=balanced_train_target,
                valid_features=reduced_valid_features,
                valid_target=self.valid_target,
            )

            # Save results
            self.add_result(hyperparameter_combination, accuracy, model)

        # Save results to csv
        self.results.sort_values("accuracy", ascending=False, inplace=True)

    def get_best_hyperparameters(self):
        """
        Get the best hyperparameter combination

        Returns:
            tuple: The best hyperparameter combination
        """

        return self.results.iloc[0]

    def save_results(self):
        """
        Save the results to a csv file
        """

        # Save best model
        best_model = self.get_best_hyperparameters()["model"]
        torch.save(best_model, "out/best_model.pt")

        # Remove models from results
        self.results.drop("model", axis=1, inplace=True)

        # Save results
        self.results.to_csv("out/grid_search_results.csv", index=False)
