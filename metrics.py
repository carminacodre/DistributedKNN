def accuracy(true_values, predicted_values):
    """

    :param true_values:
    :param predicted_values:
    :return:
    """
    return sum(1 for x,y in zip(true_values,predicted_values) if x == y) / len(true_values)