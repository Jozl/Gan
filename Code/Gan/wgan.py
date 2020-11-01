import torch
from sklearn import svm
from sklearn.model_selection import KFold
from torch import nn, Tensor
from torch.autograd import Variable
from torch.utils.data import DataLoader

import numpy as np

from Code.data.dataset import MyDataset
from Code.utils.classify_helper import classify, print_acc, compute_acc, compute_classification_indicators, \
    compute_TP_TN_FP_FN

z_dim = 640


class WGAN:
    def __init__(self, dataname, target_label, batch_size=64, lr=0.0064):
        self.lr = lr
        self.generator = None

        self.dataset = MyDataset(dataname, target_label)
        # data loader 数据载入
        self.dataloader = DataLoader(
            dataset=self.dataset, batch_size=batch_size, shuffle=True
        )

        self.shape_input = len(self.dataset.data[0])

    def train(self, epochs):
        generator = self.Generator(self.shape_input)
        discriminator = self.Discriminator(self.shape_input)

        # Optimizers
        optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=self.lr)
        optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=self.lr)

        # ----------
        #  Training
        # ----------

        for epoch in range(epochs):
            for i, (data, _) in enumerate(self.dataloader):
                # Configure input
                data_real = Variable(data)

                # ---------------------
                #  Train Discriminator
                # ---------------------
                for _ in range(10):
                    optimizer_D.zero_grad()

                    # Sample noise as generator input
                    z = Variable(torch.randn(1000, z_dim))

                    # Generate a batch of images
                    data_fake = generator(z).detach()
                    # Adversarial loss
                    loss_D = -torch.mean(discriminator(data_real)) + torch.mean(discriminator(data_fake))
                    loss_D.backward()
                    optimizer_D.step()

                    # Clip weights of discriminator
                    for p in discriminator.parameters():
                        p.data.clamp_(-0.01, 0.01)

                # Train the generator every n_critic iterations
                for _ in range(1):
                    # -----------------
                    #  Train Generator
                    # -----------------

                    optimizer_G.zero_grad()

                    # Generate a batch of images
                    data_fake = generator(z)
                    # Adversarial loss
                    loss_G = -torch.mean(discriminator(data_fake))
                    # loss_G *= 10

                    loss_G.backward()
                    optimizer_G.step()

                # print("\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                #       % (epoch, epochs, i % len(self.dataloader), len(self.dataloader), loss_D.item(),
                #          loss_G.item()))

        self.generator = generator

    def gen(self, output_num):
        z = Variable(torch.randn(output_num, z_dim))

        # Generate a batch of images
        data_fake = [
            [round
             (f * np.std([td[i] for td in self.dataset.data]) + np.mean([td[i] for td in self.dataset.data]), 2
              ) for i, f in enumerate(fd.tolist())
             ] for fd in self.generator(z).detach()
        ]

        return data_fake

    class Generator(nn.Module):
        def __init__(self, shape_input):
            super(WGAN.Generator, self).__init__()

            def block(in_feat, out_feat, normalize=False):
                layers = [nn.Linear(in_feat, out_feat)]
                if normalize:
                    layers.append(nn.BatchNorm1d(out_feat, 0.8))
                layers.append(nn.LeakyReLU(0.2, inplace=True))
                return layers

            self.model = nn.Sequential(
                *block(z_dim, 128, normalize=False),
                *block(128, 256),
                *block(256, shape_input),
                # *block(512, 1024),
                # nn.Linear(1024, shape_input),
                nn.Tanh()
            )

        def forward(self, z):
            return self.model(z)

    class Discriminator(nn.Module):
        def __init__(self, shape_input):
            super(WGAN.Discriminator, self).__init__()
            self.model = nn.Sequential(
                nn.Linear(shape_input, 512),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(512, 256),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(256, 1),
            )

        def forward(self, data):
            return self.model(data)


if __name__ == '__main__':
    # kf = None
    kf = KFold(5, shuffle=False)
    classifier = svm.SVC
    use_gan = False

    datanames = [
        'yeast-0-5-6-7-9_vs_4.dat',
        'ecoli4.dat',
        'glass5.dat',
        'yeast5.dat',
        'yeast6.dat',
    ]
    for dataname in datanames:
        print('present dataset: ', dataname)
        dataset = MyDataset(dataname)

        label_positive, label_negative = dataset.label_positive, dataset.label_negative

        data_positive = [dataset.data[i] for i, (_, l) in enumerate(dataset) if l == label_positive]
        labels_positive = len(data_positive) * [label_positive]
        data_negative = [dataset.data[i] for i, (_, l) in enumerate(dataset) if l == label_negative]
        labels_negative = len(data_negative) * [label_negative]
        # print('negative num: ', len(data_negative))

        gan = WGAN(dataname, label_negative)
        data_fake_negative = gan.gen(len(data_positive))
        labels_fake_negative = len(data_negative) * [label_negative]

        print('after gan')
        acc = 0.0
        for (i_p, j_p), (i_n, j_n) in zip(kf.split(data_positive, labels_positive),
                                          kf.split(data_negative, labels_negative)):
            train_X = [data_positive[i] for i in i_p] + [data_negative[i] for i in i_n]
            train_y = [labels_positive[i] for i in i_p] + [labels_negative[i] for i in i_n]
            test_X = [data_positive[i] for i in j_p] + [data_negative[i] for i in j_n]
            test_y = [labels_positive[i] for i in j_p] + [labels_negative[i] for i in j_n]

            clf = classifier()
            clf.fit(train_X, train_y)
            predict = clf.predict(test_X)

            acc += compute_classification_indicators(
                *compute_TP_TN_FP_FN(test_y, predict, label_positive, label_negative))[1]

        print_acc(acc / 5)
