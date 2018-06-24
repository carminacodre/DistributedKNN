import numpy as np
from numpy import dot
from numpy.linalg import norm

def euclidian(x_test, x_train):
    """
    Euclidian distance for two 1d np.arrays

    :param x_test: 1d np array
    :param x_train: 1d np array
    :return: euclidian distance
    """
    return np.sqrt(np.sum(np.power(x_test - x_train,2)))

def manhattan(x_test, x_train):
    """
    Manhattan distance for two 1d np.arrays

    :param x_test: 1d np array
    :param x_train: 1d np array
    :return: manhattan distance
    """
    return np.sum(np.abs(x_test - x_train))

def cosine(x_test, x_train):
    """
    Cosine distance for two 1d np.arrays

    :param x_test: 1d np array
    :param x_train: 1d np array
    :return: cosine distance
    """
    return dot(x_test, x_train) / (norm(x_test) * norm(x_train))