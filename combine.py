from knn import predict_class
import pickle

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

    knn = sorted(all_dist, key=lambda x: x.val)
    knn = knn[:k]
    cls = predict_class(knn, num_classes)
    return cls

def combine_votes_different_files(file_list, k, num_classes):
    results = []
    for file in file_list:
        with open(file,'rb') as f:
            results.append(pickle.load(f))

    predictions = []
    for sample in range(len(results[0])):
        voters = []
        for r in range(len(results)):
            voters.append(results[r][sample])
        cls = combine_voters(voters, k, num_classes)
        predictions.append(cls)
    return predictions



def merge_results(file_list, k, num_classes):
    results = []
    for file in file_list:
        with open(file, 'rb') as f:
            results+= pickle.load(f)
    predictions = []
    for sample in range(len(results)):
        cls = predict_class(results[sample], num_classes)
        predictions.append(cls)
    return predictions
