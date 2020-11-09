import numpy as np

from sklearn import svm
from sklearn.model_selection import KFold

from Code.Gan.wgan import WGAN
from Code.data.dataset import MyDataset
from Code.utils.classify_helper import compute_classify_indicators, compute_TP_TN_FP_FN, print_classify_indicators
from Code.utils.sheet_helper import SheetWriter

if __name__ == '__main__':
    kf = KFold(5, shuffle=False)
    sheet_writer = SheetWriter('test2.csv')
    sheet_writer.writerow(['acc+', 'acc-', 'accuracy', 'precision', 'recall', 'F1', 'G-mean'])

    for _ in range(1):
        datanames = [
            # 'yeast-0-5-6-7-9_vs_4.dat',
            # 'ecoli4.dat',
            # 'glass5.dat',
            # 'yeast5.dat',
            # 'yeast6.dat',
            'kdd99_new_multi.dat',
        ]
        for dataname in datanames:
            print('present dataset: ', dataname)
            dataset = MyDataset(dataname)
            i_symbol = [
                i for i, m in enumerate(dataset.attr_map)
                if m and len(m) > 2
            ]
            i_two_value = [
                i for i, m in enumerate(dataset.attr_map)
                if m and len(m) <= 2
            ]
            i_continue = [
                i for i, m in enumerate(dataset.attr_map)
                if not m
            ]
            for i_selection in [
                i_continue + i_two_value + i_symbol,
                i_continue,
                i_two_value,
                i_symbol
            ]:
                label_positive, label_negative = dataset.label_positive, dataset.label_negative

                data_positive = [dataset.data[i] for i, (_, l) in enumerate(dataset) if l == label_positive]
                labels_positive = len(data_positive) * [label_positive]
                data_negative = [dataset.data[i] for i, (_, l) in enumerate(dataset) if l == label_negative]
                labels_negative = len(data_negative) * [label_negative]
                # --------------------------------
                # gan
                # ---------------------------------
                gan = WGAN(dataname, label_negative)
                gan.train(200)
                data_fake_negative = gan.gen(len(data_positive))
                labels_fake_negative = len(data_fake_negative) * [label_negative]

                print('after gan')
                sheet_writer.writerow('use features at {}'.format(i_selection))
                args = None
                for (i_train, i_test), (j_train, j_test) in zip(
                        kf.split(data_positive, labels_positive),
                        kf.split(data_negative, labels_negative)
                ):
                    # ==================
                    # 筛选
                    # ==================

                    train_X = np.array(
                        [data_positive[i] for i in i_train] +
                        [data_negative[i] for i in j_train] +
                        data_fake_negative
                    )[:, i_selection]

                    train_y = np.array(
                        [labels_positive[i] for i in i_train] +
                        [labels_negative[i] for i in j_train] +
                        labels_fake_negative
                    )
                    test_X = np.array(
                        [data_positive[i] for i in i_test] +
                        [data_negative[i] for i in j_test]
                    )[:, i_selection]
                    test_y = np.array(
                        [labels_positive[i] for i in i_test] +
                        [labels_negative[i] for i in j_test]
                    )

                    clf = svm.SVC()
                    clf.fit(train_X, train_y)
                    predict = clf.predict(test_X)

                    temp = compute_classify_indicators(
                        *compute_TP_TN_FP_FN(test_y, predict, label_positive, label_negative))
                    if not args:
                        args = temp
                    else:
                        args = [a + t for a, t in zip(args, temp)]

                args = [round(a / 5, 5) for a in args]
                print_classify_indicators(args)
                # sheet_writer.writerow(args + [dataname])
