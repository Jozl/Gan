from torch import Tensor

from torch.utils.data.dataset import Dataset

from Code.data.datareader import MyDatareader


class RevDataset:
    def __init__(self, dataname, target_label=None, transform=None, ):
        self.target_label = target_label
        datalist = MyDatareader(dataname, target_label).read_datalist()

        self.data = [[d[i] for d in [d[:-1] for d in datalist]] for i in range(len(datalist[0]) - 1)]
        self.labels = [d[-1] for d in datalist]
        self.label_dict = {l: self.labels.count(l) for l in set(self.labels)}

    class SubDataset(Dataset):
        def __init__(self, datalist, target_label=None, transform=None, ):
            super(RevDataset.SubDataset, self).__init__()
            self.target_label = target_label
            self.data = datalist

        def __getitem__(self, index):
            data, label = (self.data[index],), self.target_label

            return Tensor(data), label

        def __len__(self):
            return len(self.data)

    @property
    def label_positive(self):
        return sorted(self.label_dict)[0]

    @property
    def label_negative(self):
        return sorted(self.label_dict)[-1]

    def __getitem__(self, index):
        return self.SubDataset(self.data[index], self.target_label)

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    dataset = RevDataset('ecoli4.dat', target_label='negative')
