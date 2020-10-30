def compute_acc(predict_y, test_y):
    assert len(predict_y) == len(test_y)
    return sum([1 if p == t else 0 for p, t in zip(predict_y, test_y)]) / len(test_y)
