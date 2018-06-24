from random import shuffle
import csv
import os
import numpy as np

data_encoding = {
    'setosa': 0,
    'versicolor': 1,
    'virginica': 2
}

class DataPoint:

    def __init__(self, val, cls):
        self.val = val
        self.cls = cls

    def __getitem__(self, i):
        if 0 == i:
            return self.val
        elif 1 == i:
            return self.cls
        else:
            raise IndexError("index out of bounds on DataPoint!")


def read_data(file_path):

    with open(file_path) as f:
        data = [line.split(',') for line in f]
        for d in data:
            d[-1] = d[-1].strip()

    print("Read the dataset")
    return data

def split_data(dataset, nr_splits, test_factor , fshuffle=True , dest_dir='data', name='dataset'):

    header = dataset[0]
    data = dataset[1:]

    if fshuffle:
        shuffle(data)


    # save the test samples separately
    test_data = data[:int(len(data) * test_factor)]
    data = data[int(len(data) * test_factor):]

    test_file = os.path.join(dest_dir, 'test.csv')
    with open(test_file, 'w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(header)
        writer.writerows(test_data)


    # save the rest of the files
    num_samples = len(data) // nr_splits

    for i in range(nr_splits):
        if i < nr_splits -1:
            to_write = data[(num_samples * i):(num_samples * (i+1))]
        else:
            to_write = data[(num_samples * i):]
        # save to csvs
        file_path = os.path.join(dest_dir, name + str(i) + '.csv')
        with open(file_path, 'w') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(header)
            writer.writerows(to_write)

    print("Data splitted into test and separate files")

def read_preprocess_data(file_path, shuffle=False):

    dataset = read_data(file_path)
    header = dataset[0]
    print("Header:" + str(header))
    data = dataset[1:]

    X = [np.array(d[:-1]) for d in data]
    Y = [d[-1].strip() for d in data]

    # encode classes as int
    Y = [data_encoding[y] for y in Y]

    return X, Y

if __name__ == '__main__':
    dataset = read_data('data/dataset0.csv')
    split_data(dataset, 4, 0.15)