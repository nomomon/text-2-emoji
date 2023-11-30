import pandas as pd
import itertools
from features.embedding_processing import balance_dataframe, reduce_dimensions
from models.nn_classifier import create_model, create_optimizer, train_model


class GridSearchModel():
    """
    A model that performs a grid search over a set of hyperparameters using a neural network
    """

    hyperparameters = {
        "n_dimensions": [10, 50],
        "n_layers": [1, 2],
        "n_neurons": [10, 50],
        "optimizer_type": ["adam", "sgd"],
        "learning_rate": [0.01, 0.001],
        "epochs": [5, 10, 20],
    }

    def __init__(self):
        """
        Initialize the model, load data and create results dataframe
        """

        # Load data, change to real embeddings later
        self.train_data = pd.read_csv("../data/gold/train_random_embeddings.csv")
        self.val_data = pd.read_csv("../data/gold/val_random_embeddings.csv")

        # Create results dataframe
        self.results = pd.DataFrame(columns=list(self.hyperparameters.keys()) + ["accuracy"])

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

        # Iterate over all combinations
        for hyperparameter_combination in hyperparameter_combinations:

            if verbose:
                print(f"Running hyperparameter combination: {hyperparameter_combination}")

            # Unpack hyperparameters
            n_dimensions, n_layers, n_neurons, optimizer_type, learning_rate, epochs = hyperparameter_combination

            # Balance and reduce dimensions of data
            reduced_train_data = reduce_dimensions(self.train_data, n_dimensions)
            balanced_train_data = balance_dataframe(reduced_train_data)
            reduced_val_data = reduce_dimensions(self.val_data, n_dimensions)

            # Create model
            model = create_model(n_dimensions, n_layers, n_neurons)
            optimizer = create_optimizer(model, optimizer_type, learning_rate)

            # Train model
            accuracy = train_model(model, optimizer, epochs, balanced_train_data, reduced_val_data)

            # Save results
            self.add_result(hyperparameter_combination, accuracy)

        # Save results to csv
        self.results.to_csv("../data/grid_search_results.csv", index=False)

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
