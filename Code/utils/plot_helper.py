import numpy as np

import matplotlib.pyplot as plt

from Code.Gan.wgan import WGAN
from Code.data.dataset import MyDataset

if __name__ == '__main__':
    datanames = [
        # 'yeast-0-5-6-7-9_vs_4.dat',
        # 'ecoli4.dat',
        'glass5.dat',
        # 'yeast5.dat',
        # 'yeast6.dat',
    ]
    for dataname in datanames:
        print('present dataset: ', dataname)
        dataset = MyDataset(dataname)
        label_positive, label_negative = dataset.label_positive, dataset.label_negative

        d_round = 43
        g_round = 3
        gan = WGAN(dataname, label_negative, d_round=d_round, g_round=g_round)
        gan.train(500)

        loss_log = np.mat(gan.loss_log)
        # print(np.arange(start=1, stop=10))
        plt.plot(np.arange(1, len(loss_log) + 1), loss_log[:, 0], color='red', alpha=0.5)
        plt.title('{}\nloss_D {} rounds per batch'.format(dataname, d_round))
        plt.xlabel('batch')
        plt.savefig('loss_D_{}.png'.format(dataname))
        plt.show()
        plt.close()
        plt.plot(np.arange(1, len(loss_log) + 1), loss_log[:, 1], color='blue', alpha=0.5)
        plt.title('{}\nloss_G {} rounds per batch'.format(dataname,g_round))
        plt.xlabel('batch')
        plt.savefig('loss_G_{}.png'.format(dataname))
        plt.show()
        plt.close()
