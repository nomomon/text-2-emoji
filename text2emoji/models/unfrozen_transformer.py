import torch
import transformers
import gc

import numpy as np

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.nn.functional import softmax
from tqdm import tqdm

# Number of epochs to train for, we expect the model to converge very quickly before this
MAX_EPOCHS = 10


class CustomDataset(Dataset):
    """
    Custom dataset class for the transformer model
    Uses the tokenizer to encode the text
    """

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


def create_data_loader(features, labels, tokenizer, max_len, batch_size):
    """
    Create a data loader for the given features and labels

    Args:
        features (series): The features
        labels (series): The labels
        tokenizer (Transformers.tokenizer): The tokenizer of the model
        max_len (int): The maximum length of the input in tokens
        batch_size (int): The number of samples per batch

    Returns:
        torch.utils.data.DataLoader: The data loader
    """

    ds = CustomDataset(
        text=features,
        label=labels,
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


def process_batch(model, device, batch):
    """
    Process a batch by using the input_ids and attention_mask to get the logits

    Args:
        model (Pytorch model): The model to use
        device (string): The device to train on
        batch (Dataloader batch): A batch of data obtained from the dataloader

    Returns:
        _type_: _description_
    """
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['label'].to(device)
    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

    loss = outputs.loss
    logits = outputs.logits
    preds = torch.argmax(logits, dim=1)
    correct_predictions = torch.sum(preds == labels)

    return loss, correct_predictions, len(input_ids)


def calculate_metric_over_batches(correct_predictions, total_examples, losses):
    """
    Calculate the accuracy and average loss over all batches

    Args:
        correct_predictions (tensor): The number of correct predictions
        total_examples (int): The total number of examples
        losses (list): The list of losses

    Returns:
        float: The accuracy
        float: The average loss
    """
    accuracy = (correct_predictions.double() / total_examples).item()
    average_loss = sum(losses) / len(losses)
    return accuracy, average_loss


def get_all_class_probabilities(text, model, tokenizer):
    """
    Get the probabilities for all classes for the given text

    Args:
        text (string): The text to get the probabilities for
        model (Pytorch model): The model to use
        tokenizer (Transformers.tokenizer): The tokenizer of the model

    Returns:
        list: The probabilities for all classes
    """

    inputs = tokenizer.encode_plus(text, return_tensors='pt')
    outputs = model(**inputs)
    probabilities = softmax(outputs[0], dim=1).tolist()[0]  # Apply softmax to get probabilities
    return probabilities


def train_epoch(model, device, train_loader, optimizer):
    """
    Train the model for one epoch

    Args:
        model (Pytorch model): The model to train
        device (string): The device to train on
        train_loader (Dataloader): The training data loader
        optimizer (torch.optim): The optimizer to use

    Returns:
        float: The accuracy
        float: The average loss
    """

    model.train()
    losses = []
    correct_predictions = 0
    for batch in tqdm(train_loader, desc="Training", unit="batch"):
        loss, correct, num_examples = process_batch(model, device, batch)
        correct_predictions += correct
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Free up GPU memory
        del batch
        torch.cuda.empty_cache()
        gc.collect()

    return calculate_metric_over_batches(correct_predictions, len(train_loader.dataset), losses)


def eval_epoch(model, device, val_loader):
    """
    Evaluate the model for one epoch

    Args:
        model (Pytorch model): The model to evaluate
        device (string): The device to evaluate on
        val_loader (Dataloader): The validation data loader

    Returns:
        float: The accuracy
        float: The average loss
    """

    model.eval()
    losses = []
    correct_predictions = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation", unit="batch"):
            loss, correct, num_examples = process_batch(model, device, batch)
            correct_predictions += correct
            losses.append(loss.item())

    return calculate_metric_over_batches(correct_predictions, len(val_loader.dataset), losses)


def train_model(model, optimizer, train_loader, valid_loader):
    """
    Train the model

    Args:
        model (Pytorch model): The model to train
        optimizer (torch.optim): The optimizer to use
        train_loader (Dataloader): The training data loader
        valid_loader (Dataloader): The validation data loader

    Returns:
        float: The validation accuracy
        float: The validation loss
        float: The training accuracy
        float: The training loss
        list: The training losses
        list: The validation losses
    """

    training_losses, validation_losses = [], []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in tqdm(range(MAX_EPOCHS)):

        train_accuracy, train_loss = train_epoch(model, device, train_loader, optimizer)

        # Get the validation accuracy and loss
        valid_accuracy, valid_loss = eval_epoch(model, device, valid_loader)

        # Use early stopping, if val accuracy does not improve from earlier epoch, stop training
        if epoch > 1 and valid_loss > max(validation_losses[-1:]):
            break

        training_losses.append(train_loss)
        validation_losses.append(valid_loss)

    # Convert to numpy arrays
    training_losses = np.array(training_losses)
    validation_losses = np.array(validation_losses)

    return valid_accuracy, valid_loss, train_accuracy, train_loss, training_losses, validation_losses
