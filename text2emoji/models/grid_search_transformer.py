import pandas as pd

from text2emoji.models.grid_search_model import GridSearchModel, create_results_df, print_baseline_metrics


class TransformerGridSearch(GridSearchModel):
    """
    A variant of the GridSearchModel that fine-tunes a transformer without freezing the weights
    """

    hyperparameters_keys = [
        "learning_rate",
        "dropout"
    ]

    def __init__(self, hyperparameters, embedding_type):
        """
        Initialize the model, load data and create results dataframe
        """

        self.embedding_type = embedding_type

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
