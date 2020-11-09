import csv


class SheetWriter:
    def __init__(self, file_name='sheet_saved.csv'):
        self.file_name = file_name

    def writerow(self, row):
        with open(self.file_name, 'a', newline='') as f:
            w = csv.writer(f)
            w.writerow(row)


if __name__ == '__main__':
    c = SheetWriter()
    c.writerow(['acc+', 'acc-', 'accuracy', 'precision', 'recall', 'F1', 'G-mean'])
