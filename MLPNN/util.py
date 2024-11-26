""" Utilities for Multi-Layer Perceptron Neural Network
"""

import numpy as np

def sigmoid(x : np.ndarray) -> np.ndarray:
    """ Sigmoid activation function

    Args:
        x (np.ndarray): The input data

    Returns:
        np.ndarray: The output data
    """
    return 1.0 / (1.0 + np.exp(-x))



def sigmoid_derivative(x : np.ndarray) -> np.ndarray:
    """ Derivative of the sigmoid activation function

    Args:
        x (np.ndarray): The input data

    Returns:
        np.ndarray: The output data
    """
    return sigmoid(x) * (1 - sigmoid(x))

