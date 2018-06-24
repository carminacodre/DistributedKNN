import combine
import knn
import distances

def simple_data():

    data = [ DataPoint(0.0, 0), DataPoint(0.1, 0),
             DataPoint(-0.01, 0), DataPoint(0.04, 0),
             DataPoint(1.0, 1), DataPoint(1.1, 1),
             DataPoint(0.92, 1) ]

    return [d.val for d in data], [d.cls for d in data]


def offset_vals(xy_tuple, delta):
    xs = xy_tuple[0]

    xs = map(lambda x: x + delta, xs)

    return xs, xy_tuple[1]


def test():

    data_a = simple_data()

    data_b = offset_vals(data_a, 0.01)

    data_c = offset_vals(data_a, -0.01)

    data = [ data_a, data_b, data_c ]

    knn_red = knn.KNNReducer(distances.euclidian, 3)

    x_test = 0
    data_red = map(lambda d: knn_red.reduce(d[0], d[1], x_test), data)
    zero_ok = 0 == combine.combine_voters(data_red, knn_red.k, 2)

    x_test = 1
    data_red = map(lambda d: knn_red.reduce(d[0], d[1], x_test), data)
    one_ok = 1 == combine.combine_voters(data_red, knn_red.k, 2)

    return zero_ok and one_ok


if __name__ == "__main__":
    assert(test())