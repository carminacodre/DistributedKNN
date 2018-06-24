import threading
from knn import KNNReducer
from data_utils import read_preprocess_data
from distances import euclidian

class ThreadReducer(threading.Thread):
    def __init__(self, file_name, k, dist):
        self.file_name = file_name
        self.k = k
        self.reducer = KNNReducer(distance=dist, k=k)

        self.X, self.Y = read_preprocess_data(self.file_name)

    def run(self, test_samples):
           for test_sample in test_samples:
                res = self.reducer.reduce(self.X, self.Y, test_sample)

if __name__ == "__main__":
    t = ThreadReducer("data/iris.csv", 4, euclidian)
    test_samples, _ = read_preprocess_data('data/test.csv')
    t.run(test_samples)

