from typing import Any, Tuple

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
import torch
from torch import nn
from sklearn.model_selection import train_test_split

LEARNING_RATE = 0.01
COST_FUNCTION = nn.MSELoss()
EPOCHS = 10000
DATA_SAMPLES = 2000

'''
To do:
    * See how this code works.
    * Experiments:
        - Try different activation functions.
'''

ACTIVATION_FUNCTION = nn.Sigmoid()  # nn.Sigmoid(), nn.ReLU(), nn.Tanh(), nn.Identity()


def main():
    X, labels = make_moons(DATA_SAMPLES, noise=0.2)

    # Visualize the data
    plt.scatter(X[:, 0], X[:, 1], c=labels)
    plt.show()

    # Split data in training and test values
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, labels, test_size=0.25, random_state=0
    )
    # Convert training data to tensors which can be used by the neural network
    X_train = torch.FloatTensor(X_train)
    Y_train = torch.FloatTensor(Y_train)[:, None]

    # Build a feed-forward network
    neural_network = nn.Sequential(
        nn.Linear(in_features=2, out_features=10),
        ACTIVATION_FUNCTION,
        nn.Linear(in_features=10, out_features=10),
        ACTIVATION_FUNCTION,
        nn.Linear(in_features=10, out_features=1),
        ACTIVATION_FUNCTION,
    )
    # Initialize network weights
    neural_network.apply(init_weights)

    # Optimizer
    optimizer = torch.optim.Adam(neural_network.parameters(), lr=LEARNING_RATE)

    # Define mesh variables used in a later plot
    mesh_x_coordinates, mesh_y_coordinates = get_mesh_grid(data=X)
    mesh_shape = mesh_x_coordinates.shape
    all_xy_coordinates_in_mash = torch.FloatTensor(
        np.hstack(
            (
                mesh_x_coordinates.ravel().reshape(-1, 1),
                mesh_y_coordinates.ravel().reshape(-1, 1)
            )
        )
    )

    # Define lists where loss values and accuracy values are stored
    train_loss = []
    train_accuracy = []

    for i in range(EPOCHS):
        # Set gradients to 0 before proceeding with next iteration
        optimizer.zero_grad()
        # Compute the network output given the input
        y_hat = neural_network(X_train)
        # Calculate loss
        loss = COST_FUNCTION(y_hat, Y_train)
        # Compute gradients with backpropagation
        loss.backward()
        # Perform parameters update step
        optimizer.step()

        # Store accuracy and loss
        train_accuracy.append(compute_model_accuracy(y_hat, Y_train))
        train_loss.append(loss.item())

        # Visualize decision boundary
        if i in [100, 300, 1000, 9999]:
            # Compute output for each mesh value to color plot regions
            mesh_y_hat = neural_network(all_xy_coordinates_in_mash)
            mesh_y_hat_integer = convert_prediction_to_integer(mesh_y_hat)
            mesh_y_hat_integer = mesh_y_hat_integer.reshape(mesh_shape)

            # Plot color regions
            plt.figure(figsize=(5, 4))
            plt.contourf(
                mesh_x_coordinates,
                mesh_y_coordinates,
                mesh_y_hat_integer,
                cmap=plt.cm.Accent,
                alpha=0.5
            )
            plt.scatter(X_test[:, 0], X_test[:, 1], c=Y_test, cmap=plt.cm.Accent)
            plt.title(f'Epochs: {i}')
            plt.show()

    fig, ax = plt.subplots(2, 1, figsize=(12, 8))
    ax[0].plot(train_loss)
    ax[0].set_ylabel('Loss')
    ax[0].set_title('Training Loss')

    ax[1].plot(train_accuracy)
    ax[1].set_ylabel('Classification Accuracy')
    ax[1].set_title('Training Accuracy')

    plt.tight_layout()
    plt.show()


def get_mesh_grid(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get all x and y coordinates for plot.
    """
    # Determine grid range in x and y directions
    x_min = min(data[:, 0]) - 0.1
    x_max = max(data[:, 0]) + 0.1
    y_min = min(data[:, 1]) - 0.1
    y_max = max(data[:, 1]) + 0.1

    # Set grid spacing parameter by setting minimum resolution to 1000 points
    spacing = min(x_max - x_min, y_max - y_min) / 1000

    # Create grid
    x_coordinates, y_coordinates = np.meshgrid(
        np.arange(x_min, x_max, spacing), np.arange(y_min, y_max, spacing),
    )
    return x_coordinates, y_coordinates


def compute_model_accuracy(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute how many predictions were correctly classified.
    """
    y_hat_integer = convert_prediction_to_integer(y_hat)
    return torch.sum(y == y_hat_integer) / len(y)


def convert_prediction_to_integer(y_hat: torch.Tensor) -> torch.Tensor:
    return torch.where(y_hat < 0.5, 0, 1)


def init_weights(m):
    if m is nn.Linear:
        torch.nn.init.normal_(m.weight, mean=0.0, std=0.1)
        m.bias.data.fill_(0.1)


if __name__ == '__main__':
    main()
