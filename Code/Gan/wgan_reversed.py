import numpy as np

import torch
from sklearn import svm
from sklearn.model_selection import KFold
from torch import nn, Tensor
from torch.autograd import Variable
from torch.utils.data import DataLoader

from Code.data.dataset import MyDataset
from Code.data.dataset_reversed import RevDataset

z_dim = 640


class WGAN:
    def __init__(self, dataset, target_label, rev=False, batch_size=64, lr=0.00064):
        self.lr = lr
        self.rev = rev
        self.generator = None

        self.dataloader = DataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=True
        )

        self.shape_input = len(dataset[0][0])

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
        data_fake = [[round(a, 2) for a in d.tolist()] for d in self.generator(z).detach()]

        return data_fake

    class Generator(nn.Module):
        def __init__(self, shape_input):
            super(WGAN.Generator, self).__init__()

            def block(in_feat, out_feat, normalize=True):
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
    datanames = [
        # 'yeast-0-5-6-7-9_vs_4.dat',
        'ecoli4.dat',
        # 'glass5.dat',
        # 'yeast5.dat',
        # 'yeast6.dat',
    ]
    for dataname in datanames:
        dataset = MyDataset(dataname)
        label_positive, label_negative = dataset.label_positive, dataset.label_negative
        dataset = RevDataset(dataname, label_negative)
        # subset = dataset[3]
        for subset in dataset:
            # label_positive, label_negative = label_negative, label_positive

            gan = WGAN(subset, target_label=label_negative)
            gan.train(100)
            data_fake = gan.gen(26)

            t = [round(g[0] * np.std(subset.data).astype(float) + np.mean(subset.data), 2) for g in data_fake]
            print(subset.data)
            print(t)
