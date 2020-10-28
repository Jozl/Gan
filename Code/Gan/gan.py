import torch
from sklearn import svm
from torch import nn, autograd, Tensor
from torch.utils.data.dataloader import DataLoader
from torch.autograd import Variable
from torchvision import transforms
from torchvision import datasets
from torchvision.utils import save_image

import os

# 创建文件夹
from Code.data.dataset import MyDataset

if not os.path.exists('./fake'):
    os.mkdir('./fake')

# batch_size = 32
# num_epoch = 100
# z_dim = 100
#
# dataset = MyDataset('ecoli4.dat')
# print(len(dataset))
# # data loader 数据载入
# dataloader = DataLoader(
#     dataset=dataset, batch_size=batch_size, shuffle=True
# )
#
# datalen = len(dataset.data[0])
# print(datalen)
#
#
# # 定义判别器  #####Discriminator######使用多层网络来作为判别器
# # 将图片28x28展开成784，然后通过多层感知器，中间经过斜率设置为0.2的LeakyReLU激活函数，
# # 最后接sigmoid激活函数得到一个0到1之间的概率进行二分类。
# class Discriminator(nn.Module):
#     def __init__(self):
#         super(Discriminator, self).__init__()
#         self.dis = nn.Sequential(
#             nn.Linear(datalen, 256),  # 输入特征数为784，输出为256
#             nn.LeakyReLU(0.2),  # 进行非线性映射
#             nn.Linear(256, 256),  # 进行一个线性映射
#             nn.LeakyReLU(0.2),
#             nn.Linear(256, 1),
#             nn.Sigmoid()  # 也是一个激活函数，二分类问题中，
#             # sigmoid可以班实数映射到【0,1】，作为概率值，
#             # 多分类用softmax函数
#         )
#
#     def forward(self, x):
#         x = self.dis(x)
#         return x
#
#
# # ###### 定义生成器 Generator #####
# # 输入一个100维的0～1之间的高斯分布，然后通过第一层线性变换将其映射到256维,
# # 然后通过LeakyReLU激活函数，接着进行一个线性变换，再经过一个LeakyReLU激活函数，
# # 然后经过线性变换将其变成784维，最后经过Tanh激活函数是希望生成的假的图片数据分布
# # 能够在-1～1之间。
# class Generator(nn.Module):
#     def __init__(self):
#         super(Generator, self).__init__()
#         self.gen = nn.Sequential(
#             nn.Linear(z_dim, 256),  # 用线性变换将输入映射到256维
#             nn.LeakyReLU(),  # relu激活
#             nn.Linear(256, 256),  # 线性变换
#             nn.LeakyReLU(),  # relu激活
#             nn.Linear(256, datalen),  # 线性变换
#             nn.Tanh()  # Tanh激活使得生成数据分布在【-1,1】之间，因为输入的真实数据的经过transforms之后也是这个分布
#         )
#
#     def forward(self, x):
#         x = self.gen(x)
#         return x
#
#
# # 创建对象
# D = Discriminator()
# G = Generator()
#
# # 首先需要定义loss的度量方式  （二分类的交叉熵）
# # 其次定义 优化函数,优化函数的学习率为0.0003
# # nn.BCELoss() = nn.BCELoss()  # 是单目标二分类交叉熵函数
# d_optimizer = torch.optim.Adam(D.parameters(), lr=0.000113)
# g_optimizer = torch.optim.Adam(G.parameters(), lr=0.000113)
#
# # ##########################进入训练##判别器的判断过程#####################
# for epoch in range(num_epoch):  # 进行多个epoch的训练
#     for i, (data, _) in enumerate(dataloader):
#         num_data = data.size(0)
#         # =============================训练判别器==================
#         real_data = Variable(data)  # 将tensor变成Variable放入计算图中
#         labels_real = Variable(torch.ones(num_data, 1))  # 定义真实的图片label为1
#         labels_fake = Variable(torch.zeros(num_data, 1))  # 定义假的图片的label为0
#
#         # ########判别器训练train#####################
#         # 分为两部分：1、真的图像判别为真；2、假的图像判别为假
#         # 计算真实图片的损失
#         labels_output = D(real_data)  # 将真实图片放入判别器中
#         d_loss_real = nn.BCELoss()(labels_output, labels_real)  # 得到真实图片的loss
#         real_scores = labels_output  # 得到真实图片的判别值，输出的值越接近1越好
#         # 计算假的图片的损失
#         z = Variable(torch.randn(num_data, z_dim))  # 随机生成一些噪声
#         fake_data = G(z).detach()  # 随机噪声放入生成网络中，生成一张假的图片。 # 避免梯度传到G，因为G不用更新, detach分离
#         labels_output = D(fake_data)  # 判别器判断假的图片，
#         d_loss_fake = nn.BCELoss()(labels_output, labels_fake)  # 得到假的图片的loss
#         fake_scores = labels_output  # 得到假图片的判别值，对于判别器来说，假图片的损失越接近0越好
#         # 损失函数和优化
#         d_loss = (d_loss_real + d_loss_fake) / 2  # 损失包括判真损失和判假损失
#         d_optimizer.zero_grad()  # 在反向传播之前，先将梯度归0
#         d_loss.backward()  # 将误差反向传播
#         d_optimizer.step()  # 更新参数
#
#         # ==================训练生成器============================
#         # ###############################生成网络的训练###############################
#         # 原理：目的是希望生成的假的图片被判别器判断为真的图片，
#         # 在此过程中，将判别器固定，将假的图片传入判别器的结果与真实的label对应，
#         # 反向传播更新的参数是生成网络里面的参数，
#         # 这样可以通过更新生成网络里面的参数，来训练网络，使得生成的图片让判别器以为是真的
#         # 这样就达到了对抗的目的
#         # 计算假的图片的损失
#         for _ in range(80):
#             z = Variable(torch.randn(num_data, z_dim))  # 得到随机噪声
#             data_output = G(z)  # 随机噪声输入到生成器中，得到一副假的图片
#             output = D(data_output)  # 经过判别器得到的结果
#             g_loss = nn.BCELoss()(output, labels_real)  # 得到的假的图片与真实的图片的label的loss
#             # bp and optimize
#             g_optimizer.zero_grad()  # 梯度归0
#             g_loss.backward()  # 进行反向传播
#             g_optimizer.step()  # .step()一般用在反向传播后面,用于更新生成网络的参数
#
#         # 打印中间的损失
#         if (i + 1) % (len(dataset) // batch_size) == 0:
#             print('Epoch[{}/{}],d_loss:{:.6f},g_loss:{:.6f} '
#                   'D real: {:.6f},D fake: {:.6f}'.format(
#                 epoch, num_epoch, d_loss.data.item(), g_loss.data.item(),
#                 real_scores.data.mean(), fake_scores.data.mean()  # 打印的是真实图片的损失均值
#             ))
#
# z = Variable(torch.randn(1, z_dim))  # 得到随机噪声
# data_output = G(z)  # 随机噪声输入到生成器中，得到一副假的图片
# print(data_output.tolist())
#
# # 保存模型
# torch.save(G.state_dict(), './generator.pth')
# torch.save(D.state_dict(), './discriminator.pth')

if __name__ == '__main__':
    clf = svm.SVC()
    dataset = MyDataset('ecoli4.dat', transform=True)
    data_positive = [d for i, (d, l) in enumerate(dataset) if l == dataset.label_positive]
    print([d for d in data_positive[0].numpy()])
    print(dataset.data[0])
    print(dataset.data[0])
    # labels_positive = [l for i, (d, l) in enumerate(dataset) if l == dataset.label_positive]
    # data_negative = [d.tolist() for i, (d, l) in enumerate(dataset) if l == dataset.label_negative]
    # print(data_negative[0])
    # labels_negative = [l for i, (d, l) in enumerate(dataset) if l == dataset.label_negative]
    # data_fake_negative = [d.tolist() for d in G(torch.randn(len(data_positive), z_dim))]  # 随机噪声输入到生成器中，得到一副假的图片
    # labels_fake_negative = [dataset.label_negative] * len(data_positive)

    # clf.fit(data_positive + data_fake_negative, labels_positive + labels_fake_negative)

    # predict = clf.predict(data_negative).tolist()
    # print(predict)
    # print(predict.count(dataset.label_negative) / len(predict))
