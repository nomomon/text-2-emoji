import torch
from tqdm import tqdm
import numpy as np


def get_model(n_inputs, n_layers, n_neurons, n_outputs=20):
    """
    Create a neural network model

    Args:
        n_inputs (int): Number of input features
        n_layers (int): Number of hidden layers
        n_neurons (int): Number of neurons in each hidden layer
        n_outputs (int): Number of output features, defaults to 20

    Returns:
        torch.nn.Sequential: The model
    """

    # Create a list of layers
    layers = []

    # Add input layer
    layers.append(torch.nn.Linear(n_inputs, n_neurons))
    layers.append(torch.nn.ReLU())

    # Add hidden layers
    for _ in range(n_layers):
        layers.append(torch.nn.Linear(n_neurons, n_neurons))
        layers.append(torch.nn.ReLU())

    # Add output layer
    layers.append(torch.nn.Linear(n_neurons, n_outputs))

    # Create model
    model = torch.nn.Sequential(*layers)

    return model


def get_optimizer(model, optimizer_type, learning_rate):
    """
    Create an optimizer for a model

    Args:
        model (torch.nn.Sequential): The model to create an optimizer for
        optimizer_type (string): The type of optimizer to create
        learning_rate (float): The learning rate of the optimizer

    Returns:
        torch.optim.Optimizer: The optimizer
    """

    # Create optimizer
    if optimizer_type == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_type == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")

    return optimizer


def get_performance(model, features, target):
    """
    Get the validation accuracy of a model, assuming the model is already on the GPU

    Args:
        model (torch.nn.Sequential): The model to get the validation accuracy of
        valid_features (np.array): The validation features
        valid_target (np.array): The validation target labels

    Returns:
        float: The validation accuracy and loss
    """

    # Evaluate model
    predictions = model(features)
    softmax_predictions = torch.nn.functional.softmax(predictions, dim=1)
    correct_predictions = softmax_predictions.argmax(dim=1) == target
    accuracy = correct_predictions.float().mean().item()

    # Get loss
    loss = torch.nn.functional.cross_entropy(predictions, target).item()

    return accuracy, loss


def train_model(
        model, optimizer, epochs,
        train_features, train_target,
        valid_features, valid_target,
        verbose=False, batch_size=128):
    """
    Train a model

    Args:
        model          (torch.nn.Sequential): The model to train
        optimizer      (torch.optim.Optimizer): The optimizer to use
        epochs         (int): The number of epochs to train for
        train_features (np.array): The training features
        train_target   (np.array): The training target labels
        valid_features (np.array): The validation features
        valid_target   (np.array): The validation target labels

    Returns:
        float: The accuracy of the model on the validation data
        float: The loss of the model on the validation data
        float: The accuracy of the model on training data
        float: The loss of the model on training data
        np.array: The training loss for each epoch
        np.array: The validation loss for each epoch
    """

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert to tensors on GPU
    train_features = torch.tensor(train_features).float().to(device)
    train_target = torch.tensor(train_target).to(device)
    valid_features = torch.tensor(valid_features).float().to(device)
    valid_target = torch.tensor(valid_target).to(device)

    model.to(device)

    epoch_iter = range(epochs)
    if verbose:
        epoch_iter = tqdm(epoch_iter, desc="Training")

    # Create numpy array to store training and validation loss
    train_loss_array = np.zeros(epochs)
    valid_loss_array = np.zeros(epochs)

    # Train model
    for epoch in epoch_iter:

        # Forward pass
        train_predictions = model(train_features)
        train_loss = torch.nn.functional.cross_entropy(train_predictions, train_target)

        # Store losses
        train_loss_array[epoch] = train_loss.item()
        valid_loss_array[epoch] = get_performance(model, valid_features, valid_target)[1]

        # Backward pass
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

    # Evaluate model
    valid_accuracy, valid_loss = get_performance(model, valid_features, valid_target)

    # Get training accuracy
    train_accuracy, train_loss = get_performance(model, train_features, train_target)

    return valid_accuracy, valid_loss, train_accuracy, train_loss, train_loss_array, valid_loss_array
