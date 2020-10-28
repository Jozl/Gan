import os


class MyDatareader:
    def __init__(self, dataname: str, target_label: str = None):
        self.dataname = dataname
        self.target_label = target_label
        self.datapath = self.search_file(
            os.path.join(os.path.abspath('../../'), 'data'), dataname)

        if not self.datapath:
            raise FileNotFoundError

    def read_datalist(self):
        datalist = []

        with open(self.datapath) as datafile:
            for row in datafile:
                if row.startswith('@'):
                    pass
                else:
                    row = list(map(MyDatareader.str2obj, row.split(',')))
                    if self.target_label and row[-1] != self.target_label:
                        continue
                    datalist.append(row)

        return datalist

    @staticmethod
    def str2obj(t: str):
        t = t.strip()
        try:
            t = eval(t)
        except NameError:
            pass
        return t

    def search_file(self, rootpath, dataname):
        for file in os.listdir(rootpath):
            filepath = os.path.join(rootpath, file)
            if os.path.isdir(filepath):
                res = self.search_file(filepath, dataname)
                if res:
                    return res
            elif filepath.endswith(dataname):
                return filepath
        return None


if __name__ == '__main__':
    d = MyDatareader('yeast5.dat')
    print(d.read_datalist()[:][-1])
