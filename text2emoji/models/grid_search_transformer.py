import pandas as pd

from text2emoji.models.grid_search_model import GridSearchModel, create_results_df, print_baseline_metrics, create_hyperparameter_combinations
from text2emoji.models.unfrozen_transformer import set_up_model, get_tokenizer, create_data_loader, train_model

MAX_SEQ_LEN = 100
BATCH_SIZE = 128


class TransformerGridSearch(GridSearchModel):
    """
    A variant of the GridSearchModel that fine-tunes a transformer without freezing the weights
    """

    hyperparameters_keys = [
        "learning_rate",
        "dropout"
    ]

    def __init__(self, hyperparameters, model_type):
        """
        Initialize the model, load data and create results dataframe
        """

        self.model_type = model_type

        if self.model_type == "unfrozen_bert":
            self.model_name = "bert-base-uncased"
        else:
            raise ValueError("Unknown model type")

        self.hyperparameters = hyperparameters

        path = "unfrozen_transformer"
        train_data = pd.read_csv(f'./data/silver/{path}_train.csv')
        valid_data = pd.read_csv(f'./data/silver/{path}_valid.csv')

        self.train_features, self.train_target = train_data['text'], train_data['label']
        self.valid_features, self.valid_target = valid_data['text'], valid_data['label']

        print(f"Number of rows in training data: {len(self.train_features)}")
        print(f"Number of rows in validation data: {len(self.valid_features)}")

        # Create results dataframe
        self.results = create_results_df(self.hyperparameters_keys)

        # Print baseline metrics
        print_baseline_metrics(self.train_target, self.valid_target)

        assert set(self.hyperparameters.keys()) == set(self.hyperparameters_keys)

    def run(self, verbose=True):
        """
        Run the grid search
        """

        # Create all combinations of hyperparameters
        hyperparameter_combinations = create_hyperparameter_combinations(self.hyperparameters, verbose)

        tokenizer = get_tokenizer(self.model_name)
        train_loader = create_data_loader(self.train_features, self.train_target, tokenizer, MAX_SEQ_LEN, BATCH_SIZE)
        valid_loader = create_data_loader(self.valid_features, self.valid_target, tokenizer, MAX_SEQ_LEN, BATCH_SIZE)

        # Iterate over all combinations
        for hyperparameter_combination in hyperparameter_combinations:

            # Unpack hyperparameters
            learning_rate, dropout = hyperparameter_combination

            model, optimizer = set_up_model(self.model_name, learning_rate, dropout)

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
                train_loader,
                valid_loader,
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
