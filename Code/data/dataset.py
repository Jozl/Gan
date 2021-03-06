from torch import Tensor

from torch.utils.data.dataset import Dataset
from torchvision.transforms import transforms

import numpy as np

from Code.data.datareader import MyDatareader


class MyDataset(Dataset):
    def __init__(self, dataname, target_label=None, rev=False, transform=None, ):
        super(MyDataset, self).__init__()
        self.transform = transform

        datalist = MyDatareader(dataname, target_label).read_datalist()

        self.data = [d[:-1] for d in datalist]
        if rev:
            self.data = MyDataset.reverse(self.data)
        self.labels = [d[-1] for d in datalist]
        self.label_dict = {l: self.labels.count(l) for l in set(self.labels)}

    @staticmethod
    def reverse(data):
        data_rev = [[d[i] for d in data] for i in range(len(data[0]))]
        return data_rev

    @property
    def label_positive(self):
        return sorted(self.label_dict)[0]

    @property
    def label_negative(self):
        return sorted(self.label_dict)[-1]

    def __getitem__(self, index):
        data, label = self.data[index], self.labels[index]

        if self.transform is not None:
            data = [(d - 0.5) / 0.5 for d in data]

        # return data, label
        return Tensor(data), label

    def __len__(self):
        return len(self.data)

