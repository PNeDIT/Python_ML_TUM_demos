from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

'''
To do:
    * See how this code works.
    * See how least-squares estimation is impacted by:
        - Number of samples used.
        - Noise amount in our measurements.
        - Correct choice of number of parameters to estimate.
'''

NUMBER_OF_SAMPLES = 100

# Define how much we want our data to deviate (vertically) from the true values
STD_NOISE = 0.5

# The number of elements in the following array defines the grade of the polynomial:
# For example: [2 4]    => 2x + 4
#              [2 4 6] => 2x^2 + 4x + 6
POLYNOMIAL_TRUE_COEFFICIENTS = [2, -1]
# POLYNOMIAL_TRUE_COEFFICIENTS = [0.1, -2, 4]
# POLYNOMIAL_TRUE_COEFFICIENTS = [0.02, -0.3, 2, 4]


def main() -> None:
    x, y_original, y = generate_data()

    X = compute_measurement_matrix_X(x)

    theta = np.linalg.inv(X.T @ X) @ X.T @ y
    y_hat = X @ theta

    plot_data(x, y, y_original, y_hat)


def compute_measurement_matrix_X(x: np.ndarray) -> np.ndarray:
    """
    Construct the measurement matrix from data x.
    """
    polynomial_degree = len(POLYNOMIAL_TRUE_COEFFICIENTS) - 1
    x_powers = []
    for i in range(polynomial_degree, -1, -1):
        x_powers.append(np.power(x, i))
    return np.array(x_powers).T


def generate_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate data for linear regression.
    """
    x_coordinate = np.random.uniform(low=0, high=10, size=NUMBER_OF_SAMPLES)

    noise = STD_NOISE * np.random.randn(NUMBER_OF_SAMPLES)
    y_coordinate = np.polyval(POLYNOMIAL_TRUE_COEFFICIENTS, x_coordinate)
    y_coordinate_noisy = y_coordinate + noise

    # Sort data according to x_coordinate
    order = np.argsort(x_coordinate)
    x_coordinate = np.array(x_coordinate)[order]
    y_coordinate = np.array(y_coordinate)[order]
    y_coordinate_noisy = np.array(y_coordinate_noisy)[order]

    return x_coordinate, y_coordinate, y_coordinate_noisy


def plot_data(
    x_coordinate: np.ndarray,
    y_coordinate: np.ndarray,
    y_original_coordinate: np.ndarray,
    y_hat_coordinate: np.ndarray,
) -> None:
    plt.scatter(x_coordinate, y_coordinate)
    plt.minorticks_on()
    plt.axis('equal')
    plt.axhline(0, color='black')
    plt.axvline(0, color='black')
    plt.grid()
    plt.plot(x_coordinate, y_original_coordinate, label="True Polynomial", color='blue')
    plt.plot(x_coordinate, y_hat_coordinate, label="Predicted Polynomial", color='red')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
