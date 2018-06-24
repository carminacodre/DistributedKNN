import knn
import distances
from data_utils import DataPoint

def simple_data():

    # create two classes, one near 0, with class 0
    # and one near 1 with class 1

    data = [ DataPoint(0.0, 0), DataPoint(0.1, 0),
             DataPoint(-0.01, 0), DataPoint(0.04, 0),
             DataPoint(1.0, 1), DataPoint(1.1, 1),
             DataPoint(0.92, 1) ]

    return [d.val for d in data], [d.cls for d in data]



def test():

    knn_red = knn.KNNReducer(distances.euclidian, 3)

    X_train, Y_train = simple_data()

    x_test = 0

    res = knn_red.reduce(X_train, Y_train, x_test)

    all_zero = all(map(lambda val: val[1] == 0, res))

    x_test = 1

    res = knn_red.reduce(X_train, Y_train, x_test)

    all_one = all(map(lambda val: val[1] == 1, res))

    all_good = all_zero and all_one

    return all_good

if __name__ == "__main__":
    assert(test())



