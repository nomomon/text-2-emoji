import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt

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
        "balancing_technique",
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

        self.embedding_type = embedding_type

        # Load data
        self.train_features = np.load(f"data/gold/train_{embedding_type}_features.npy")
        self.valid_features = np.load(f"data/gold/valid_{embedding_type}_features.npy")
        self.train_target = np.load(f"data/gold/train_{embedding_type}_target.npy")
        self.valid_target = np.load(f"data/gold/valid_{embedding_type}_target.npy")

        # Create results dataframe
        self.results = pd.DataFrame(
            columns=self.hyperparameters_keys
            + [
                "valid_accuracy",
                "valid_loss",
                "train_accuracy",
                "train_loss",
                "training_losses",
                "valid_losses",
                "model",
            ]
        )

        # Print number of rows
        print(f"Number of rows in training data: {len(self.train_features)}")
        print(f"Number of rows in validation data: {len(self.valid_features)}")

        # Print the most frequent label
        unique_labels, counts = np.unique(self.train_target, return_counts=True)
        most_frequent_label = unique_labels[np.argmax(counts)]
        print(f"Most frequent label: {most_frequent_label}")

        # Print the accuracy of a model that always predicts the most frequent label
        baseline_accuracy = (self.valid_target == most_frequent_label).mean()
        print(f"Baseline validation accuracy: {baseline_accuracy}")

        self.hyperparameters = hyperparameters
        self.current_dimensions_reduction = None
        self.n_dimensions = None

        # Assert if keys are in hyperparameters
        assert set(self.hyperparameters.keys()) == set(self.hyperparameters_keys)

    def add_result(
        self,
        hyperparameter_combination,
        valid_accuracy,
        valid_loss,
        train_accuracy,
        train_loss,
        training_losses,
        valid_losses,
        model,
    ):
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

        training_details = pd.Series(training_details)

        training_details["valid_accuracy"] = valid_accuracy
        training_details["valid_loss"] = valid_loss
        training_details["train_accuracy"] = train_accuracy
        training_details["train_loss"] = train_loss
        training_details["model"] = model
        training_details["training_losses"] = training_losses
        training_details["valid_losses"] = valid_losses

        # Add results to dataframe
        self.results.loc[len(self.results)] = training_details

    def reduce_and_balance_data(self, dimensions_reduction, n_dimensions, balance_technique):
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
            reduced_train_features, self.train_target, balance_technique
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
                balance_technique,
                n_layers,
                n_neurons,
                optimizer_type,
                learning_rate,
                epochs,
            ) = hyperparameter_combination

            # Only balance and reduce data when the technique and number of dimensions changed
            if (n_dimensions != self.n_dimensions) or (
                dimensions_reduction != self.current_dimensions_reduction
            ):
                (
                    balanced_train_features,
                    balanced_train_target,
                    reduced_valid_features,
                ) = self.reduce_and_balance_data(dimensions_reduction, n_dimensions, balance_technique)

                self.current_dimensions_reduction = dimensions_reduction
                self.n_dimensions = n_dimensions

            if dimensions_reduction == "none":
                n_dimensions = self.train_features.shape[1]

            # Create model
            model = get_model(n_dimensions, n_layers, n_neurons)
            optimizer = get_optimizer(model, optimizer_type, learning_rate)

            # Train model
            (
                valid_accuracy,
                valid_loss,
                train_accuracy,
                train_loss,
                training_losses,
                validation_losses,
            ) = train_model(
                model,
                optimizer,
                epochs,
                train_features=balanced_train_features,
                train_target=balanced_train_target,
                valid_features=reduced_valid_features,
                valid_target=self.valid_target,
            )

            # Save results
            self.add_result(
                hyperparameter_combination,
                valid_accuracy,
                valid_loss,
                train_accuracy,
                train_loss,
                training_losses,
                validation_losses,
                model,
            )

        # Save results to csv
        self.results.sort_values("valid_loss", ascending=True, inplace=True)

    def get_best_hyperparameters(self):
        """
        Get the best hyperparameter combination

        Returns:
            tuple: The best hyperparameter combination
        """

        return self.results.iloc[0]

    def plot_loss_curve(self):
        """
        Plot the loss curve of the best model
        """

        # Get losses
        training_losses = self.get_best_hyperparameters()["training_losses"]
        valid_losses = self.get_best_hyperparameters()["valid_losses"]

        # Plot losses
        plt.plot(training_losses, label="Training loss")
        plt.plot(valid_losses, label="Validation loss")
        plt.legend()
        plt.show()

    def save_results(self):
        """
        Save the results to a csv file
        """

        # Save best model
        best_model = self.get_best_hyperparameters()["model"]
        torch.save(best_model, f"out/{self.embedding_type}/best_model.pt")

        # Save losses of best model
        training_losses = self.get_best_hyperparameters()["training_losses"]
        valid_losses = self.get_best_hyperparameters()["valid_losses"]
        np.save(f"out/{self.embedding_type}/training_losses.npy", training_losses)
        np.save(f"out/{self.embedding_type}/valid_losses.npy", valid_losses)

        # Remove models and losses from results
        self.results.drop("model", axis=1, inplace=True)
        self.results.drop("training_losses", axis=1, inplace=True)
        self.results.drop("valid_losses", axis=1, inplace=True)

        # Save results
        self.results.to_csv(f"out/{self.embedding_type}/grid_search_results.csv", index=False)
