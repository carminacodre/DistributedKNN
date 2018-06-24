import numpy as np


def combine_voters(voters, k, num_classes):
    """
    Function for combining the votings from multiple sources

    :param voters: list of lists of (distance, class) tuples
    :param k: k for knn
    :param num_classes: number of classes
    :return: the index for the class with the most votes
    """
    all_dist = []
    for v in voters:
        all_dist += v
    knn = sorted(all_dist, key= lambda x: x.val)
    knn = knn[:k]

    votes = [0 for i in range(num_classes)]
    for n in knn:
        votes[n[1]] += 1

    imax = 0
    max = 0

    for i,v in enumerate(votes):
        if v > max:
            max = v
            imax = i

    return imax
