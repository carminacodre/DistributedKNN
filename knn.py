from data_utils import DataPoint


def get_neighbors(X_train, Y_train, x_test, distance, k):
    """
    Get k (or less than k if training set smaller) neighbors
    for a certain test sample

    :param X_train: training instances
    :param Y_train: classes for the training instances
    :param x_test: np array for test sample data
    :param distance: distance function to be used
    :param k: k for knn
    :return: k closest neighbors and their classes as DataPoint objects
    """
    distances = [DataPoint(distance(x_train, x_test),c) for x_train,c
                 in zip(X_train, Y_train)]

    distances_sorted = sorted(distances, key=lambda x: x.val)

    if len(distances_sorted) > k:
        return distances_sorted[:k]
    else:
        return distances_sorted


class KNNReducer:
    def __init__(self, distance, k):
        self.dist = distance
        self.k = k

    def reduce(self, X_train, Y_train, x_test):
        return get_neighbors(X_train, Y_train, x_test, self.dist, self.k)
