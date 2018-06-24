import threading
from knn import KNNReducer
from data_utils import read_preprocess_data
from distances import euclidian
import csv
import os
import pickle

class ThreadReducer(threading.Thread):
    def __init__(self, nr, file_name, k, dist, test_samples, dest='data'):
        self.nr = nr
        self.file_name = file_name
        self.k = k
        self.reducer = KNNReducer(distance=dist, k=k)
        self.X, self.Y = read_preprocess_data(self.file_name)
        self.test_samples = test_samples
        self.dest = dest

    def run(self):
        results = []
        for test_sample in self.test_samples:
            res = self.reducer.reduce(self.X, self.Y, test_sample)
            results.append(res)
        # write the results for all test samples

        res_file = os.path.join(self.dest, "res"+ str(self.nr) + ".pkl")

        with open(res_file, 'wb') as f:
            pickle.dump(results, f)


class Mapper:
    def __init__(self, k, distance):
        self.k = k
        self.distance = distance
        self.threads =[]

    def start_threads(self):
        for t in self.threads:
            t.start()

    def map_and_start(self, file_list, test_samples, dest):
        self.map_files(file_list, test_samples, dest)
        self.start_threads()

class MapperDifferentSources(Mapper):
    def __init__(self, k, distance):
        Mapper.__init__(k, distance)

    def map_files(self, file_list, test_samples, dest):

        self.threads = []

        for i,f in enumerate(file_list):
            self.threads.append(ThreadReducer(i, f, self.k, self.distance, test_samples, dest))

class MapperDifferentTestSamples:
    def __init__(self, k, distance):
        Mapper.__init__(k, distance)

    def map_files(self, data_file, nr_threads ,test_samples, dest):

        self.threads = []
        nr_test_samples = len(test_samples) // nr_threads


        for i in range(nr_threads):
            if i < nr_threads -1:
                self.threads.append(ThreadReducer(i, data_file, self.k, self.distance,
                                                  test_samples[(i * nr_test_samples):((i + 1) * nr_test_samples)],
                                                  dest))
            else:
                self.threads.append(ThreadReducer(i, data_file, self.k, self.distance,
                                                  test_samples[(i * nr_test_samples):],
                                                  dest))
