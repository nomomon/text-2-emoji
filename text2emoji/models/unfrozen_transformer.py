import torch
import transformers

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm


class CustomDataset(Dataset):
    def __init__(self, text, label, tokenizer, max_len):
        self.text = text
        self.label = label
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        text = str(self.text[idx])
        label = self.label[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = CustomDataset(
        text=df.text.to_numpy(),
        label=df.label.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )

    return DataLoader(
        ds,
        batch_size=batch_size,
    )


def get_tokenizer(model_name):
    """
    Get the tokenizer for the given model

    Args:
        model_name (string): The name of the model

    Returns:
        transformers.PreTrainedTokenizer: The tokenizer
    """

    return transformers.AutoTokenizer.from_pretrained(model_name)


def set_up_model(model_name, learning_rate, dropout):
    """
    Create a transformer model

    Args:
        model_name (string): The name of the model to use
        learning_rate (float): The learning rate of the optimizer
        dropout (float): The dropout rate of the model

    Returns:
        transformers.PreTrainedModel: The model
        torch.optim.AdamW: The optimizer
    """

    # Create model
    model = transformers.AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=20, hidden_dropout_prob=dropout)

    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    return model, optimizer
