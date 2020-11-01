from sklearn import svm
from sklearn.model_selection import KFold


def classify(trains, tests=None, clf=svm.SVC(), kf: KFold = None):
    """

    :param trains: [(train_X, train_y), ...]
    :param tests: 同上
    :param clf: 分类器
    :param kf: KFolds
    :return: acc
    """
    if kf:
        indexset_list = [(kf.split(train_X), kf.split(train_y)) for (train_X, train_y) in trains]
        acc = 0.0
        for a in indexset_list:
            print(a)
    # def classify(train_X, train_y, test_X, test_y, clf, kf):


#     if kf:
#
#         kf.split()
#
#         acc = 0.0
#         for (i_train, _), (_, i_test) in zip(kf.split(train_X, train_y), kf.split(test_X, test_y)):
#             train_X_fold = [train_X[i] for i in i_train]
#             train_y_fold = [train_y[i] for i in i_train]
#             print(train_y_fold)
#             test_X_fold = [test_X[i] for i in i_test]
#             test_y_fold = [test_y[i] for i in i_test]
#             clf.fit(train_X_fold, train_y_fold)
#             acc += compute_acc(clf.predict(test_X_fold), test_y_fold)
#             print_acc(compute_acc(clf.predict(test_X_fold), test_y_fold))
#         return acc / kf.n_splits
#     else:
#         clf.fit(train_X, train_y)
#         return compute_acc(clf.predict(test_X), test_y)
#

def print_acc(acc):
    print('\033[0;36moriginal acc :  {:>2.3f} \033[0m'.format(acc))


def compute_acc(predict_y, test_y, label=None):
    assert len(predict_y) == len(test_y)
    return sum([1 if p == t and (p == label if label else True) else 0 for p, t in zip(predict_y, test_y)]) / len(
        test_y)


def compute_TP_TN_FP_FN(labels_test, labels_predict, label_positive, label_negative):
    TP, TN, FP, FN = 0, 0, 0, 0
    for x, y in zip(labels_test, labels_predict):
        TP += x == y == label_positive
        TN += x == y == label_negative
        FP += x == label_negative != y
        FN += x == label_positive != y

    return TP, TN, FP, FN


def compute_classification_indicators(TP, TN, FP, FN):
    res = (acc_p, acc_n, accuracy, precision, recall, F1, G_mean) = 0, 0, 0, 0, 0, 0, 0
    try:
        acc_p = TP / (TP + FN)
    except ZeroDivisionError:
        pass
    try:
        acc_n = TN / (TN + FP)
    except ZeroDivisionError:
        pass
    try:
        accuracy = (TP + TN) / (TP + FN + FP + TN)
    except ZeroDivisionError:
        pass
    try:
        precision = TP / (TP + FP)
    except ZeroDivisionError:
        pass
    try:
        recall = TP / (TP + FN)
    except ZeroDivisionError:
        pass
    try:
        F1 = (2 * TP) / (2 * TP + FN + FP)
    except ZeroDivisionError:
        pass
    try:
        G_mean = ((TP / (TP + FN)) * (TN / (TN + FP))) ** 0.5
    except ZeroDivisionError:
        pass

    return acc_p, acc_n, accuracy, precision, recall, F1, G_mean
