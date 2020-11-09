import numpy as np

import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import KFold

from Code.Gan.wgan import WGAN
from Code.data.dataset import MyDataset
from Code.utils.classify_helper import print_classify_indicators, compute_classify_indicators, compute_TP_TN_FP_FN
from Code.utils.sheet_helper import SheetWriter


def gan_rounds(round_d, round_g):
    for i in range(1, round_d + 1, 5):
        for j in range(1, round_g + 1, 5):
            yield i, j


if __name__ == '__main__':
    datanames = [
        # 'yeast-0-5-6-7-9_vs_4.dat',
        # 'ecoli4.dat',
        # 'glass5.dat',
        # 'yeast5.dat',
        'yeast6.dat',
    ]
    for dataname in datanames:
        print('present dataset: ', dataname)
        dataset = MyDataset(dataname)
        acc_dict = np.zeros([50, 50])
        label_positive, label_negative = dataset.label_positive, dataset.label_negative

        data_positive = [dataset.data[i] for i, (_, l) in enumerate(dataset) if l == label_positive]
        labels_positive = len(data_positive) * [label_positive]
        data_negative = [dataset.data[i] for i, (_, l) in enumerate(dataset) if l == label_negative]
        labels_negative = len(data_negative) * [label_negative]

        for round_d, round_g in gan_rounds(50, 50):
            filename = '{}_rd={}_rg={}'.format(dataname, round_d, round_g)
            print(filename)
            kf = KFold(5, shuffle=False)
            sheet_writer = SheetWriter(filename + '.csv')
            sheet_writer.writerow(['acc+', 'acc-', 'accuracy', 'precision', 'recall', 'F1', 'G-mean'])

            gan = WGAN(dataname, label_negative, d_round=round_d, g_round=round_g)
            gan.train(1000)

            data_fake_negative = gan.gen(len(data_positive))
            labels_fake_negative = len(data_fake_negative) * [label_negative]

            args = None
            for (i_train, i_test), (j_train, j_test) in zip(
                    kf.split(data_positive, labels_positive),
                    kf.split(data_negative, labels_negative)
            ):
                train_X = [data_positive[i] for i in i_train] + data_fake_negative
                train_y = [labels_positive[i] for i in i_train] + labels_fake_negative
                test_X = [data_positive[i] for i in i_test] + [data_negative[i] for i in j_test]
                test_y = [labels_positive[i] for i in i_test] + [labels_negative[i] for i in j_test]

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

            print('dr={} gr={} get acc={}'.format(round_d, round_g, args[1]))
            acc_dict[round_d - 1, round_g - 1] = args[1]

            print_classify_indicators(args)
            sheet_writer.writerow(args + [dataname])

            loss_log = np.mat(gan.loss_log)
            plt.plot(np.arange(1, len(loss_log) + 1), loss_log[:, 0], color='red', alpha=0.5)
            plt.plot(np.arange(1, len(loss_log) + 1), loss_log[:, 1], color='blue', alpha=0.5)
            plt.plot(np.arange(1, len(loss_log) + 1), np.zeros(len(loss_log)), color='black', alpha=0.8)

            plt.title('{}\nround_D={} round_G={}'.format(dataname, round_d, round_g))
            plt.xlabel('batches')
            plt.savefig(filename + 'gan_loss.png')
            plt.show()
            plt.close()
        print(acc_dict)