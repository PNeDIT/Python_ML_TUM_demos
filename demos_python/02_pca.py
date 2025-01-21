from typing import Tuple, List

import matplotlib
import numpy as np
from mnist import MNIST
import matplotlib.pyplot as plt

'''
To do:
    * See how this code works and does exactly what we have seen in the lectures.
    * Experiments:
        - See MNIST samples (PLOT_SAMPLE_INDEX).
        - See how is "energy" distributed in eigenvalues (CUMULATIVE_EIGENVALUES_THRESHOLD).
        - See the eigenvectors represented.
        - See data with reduced dimensionality after applying PCA.
        - See how different classes are separated (SCATTER_PLOT_CLASSES).
        - See how we can interpret PCA as a "compression" mechanism by choosing d.
'''

CONSIDERED_SAMPLES = 1000

SCATTER_PLOT_CLASSES = [i for i in range(10)]

PLOT_SAMPLE_INDEX = 10
CUMULATIVE_EIGENVALUES_THRESHOLD = 0.90

d = 2

def main():
    x_all, labels = load_mnist_data(
        folder_path="/mnt/2EE473C3189A58E4/download/mnist",
        n_samples=CONSIDERED_SAMPLES,
    )
    M, N, _ = x_all.shape

    # 1. Estimate sample mean mu_x_hat and compute mean-centered samples
    mu_x_hat = np.mean(x_all, axis=0)
    x_all_centered = x_all - mu_x_hat
    show_sample(dataset=x_all, index=PLOT_SAMPLE_INDEX)

    # 2. Estimate sample covariance C_x_hat
    all_outer_products = []
    for i in range(CONSIDERED_SAMPLES):
        x_i_centered = x_all_centered[i]
        all_outer_products.append(x_i_centered @ x_i_centered.T)
    C_x_hat = 1/(M - 1) * np.sum(np.array(all_outer_products), axis=0)
    # Or, equivalently computed with the following command:
    # C_x_hat = np.cov(x_all_centered[:, :, 0].T)

    # 3. Compute sorted eigenvalues matrix Lambda and eigenvectors matrix U
    lambdas, U = np.linalg.eigh(C_x_hat)
    lambdas, U = sort_descending(lambdas, U)
    bar_plots_eigenvalues(lambdas)

    # 4. Take the d largest eigenvalues and eigenvectors
    lambdas = lambdas[:d]
    U = U[:, :d]
    # Let's visualize the eigenvectors!
    for i in range(d):
        show_sample(dataset=U.T[:, :, None], index=i, title=f"Eigenvector {i + 1}")

    # 5. Transform high-dimensional data to low-dimensional
    y_all = x_all_centered[:, :, 0] @ U
    plot_scattered_data(
        data_2d=y_all,
        labels=labels,
        labels_to_represent=SCATTER_PLOT_CLASSES
    )

    # The data y_all represents a compressed version of the data.
    # Let's decompress by projecting the compressed vector on the eigenvectors.
    reprojected_sample = sum(y_all[PLOT_SAMPLE_INDEX, i] * U[:, i] for i in range(d))
    show_sample(reprojected_sample[None, :, None], index=0)

    pass


def sort_descending(
    lambdas: np.ndarray,
    U: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sort eigenvalues and eigenvectors in descending order according to the eigenvalues.
    """
    sorting_indices = np.argsort(-lambdas)
    return lambdas[sorting_indices], U[:, sorting_indices]


def plot_scattered_data(
    data_2d: np.ndarray,
    labels: np.ndarray,
    labels_to_represent: List[int],
) -> None:
    """
    Scatter plot of data with labels_to_represent in the data_2d with labels.
    """
    plt.legend(labels)
    plt.axis('equal')
    plt.title("Data after applying PCA")
    plt.xlabel(r'$y_1$')
    plt.ylabel(r'$y_2$')

    cmap = matplotlib.colormaps.get_cmap('tab10')
    for label in range(np.min(labels), np.max(labels) + 1):
        if label not in labels_to_represent:
            continue
        mask = (labels == label)[:, 0]
        plt.scatter(
            data_2d[mask, 0],
            data_2d[mask, 1],
            label=f'Class {label}',
            color=cmap(label),
            s=20,
            alpha=0.7
        )
    plt.legend()
    plt.show()


def bar_plots_eigenvalues(lambdas: np.ndarray) -> None:
    """
    Plot bar plot of eigenvalues and the cumulative sum of eigenvectors.
    """
    # Show bar plot of eigenvalues
    N = len(lambdas)
    plt.bar(np.arange(N), lambdas)
    plt.xlabel("Eigenvalue index")
    plt.title("Eigenvalues")
    plt.show()

    # Show bar plot of cumulative eigenvalues, with vertical line where
    # CUMULATIVE_EIGENVALUES_THRESHOLD threshold is passed.
    cumulative_lambdas = np.cumsum(lambdas)
    threshold = CUMULATIVE_EIGENVALUES_THRESHOLD * np.sum(lambdas)
    threshold_index = 0
    for i in range(N):
        if cumulative_lambdas[i] >= threshold:
            threshold_index = i
            break
    plt.bar(np.arange(N), cumulative_lambdas)
    plt.xlabel("Eigenvalue index")
    plt.title("Cumulative eigenvalues")
    plt.axvline(threshold_index, color="red", linestyle="--")
    plt.show()


def show_sample(dataset: np.ndarray, index: int, title: str = None) -> None:
    """
    Visualize the dataset sample at the given index.
    """
    sample = dataset[index].reshape((28, 28))
    plt.imshow(sample, cmap='Greys')
    if title:
        plt.title(title)
    plt.show()


def load_mnist_data(
    *,
    folder_path: str,
    n_samples: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns n_samples MNIST image and labels from ubyte files stored in folder_path.
    """
    mnist_data = MNIST(folder_path)
    train_images, train_labels = mnist_data.load_training()
    test_images, test_labels = mnist_data.load_testing()
    train_images = np.array(train_images)[:, :, None]
    train_labels = np.array(train_labels)[:, None]
    test_images = np.array(test_images)[:, :, None]
    test_labels = np.array(test_labels)[:, None]

    images = np.concatenate((train_images, test_images), axis=0)
    labels = np.concatenate((train_labels, test_labels), axis=0)

    selected_samples_indices = np.random.choice(
        images.shape[0], n_samples, replace=False
    )

    return images[selected_samples_indices], labels[selected_samples_indices]


if __name__ == '__main__':
    main()
