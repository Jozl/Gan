import os

path_root = os.path.abspath('../../')
file_head = os.path.join(path_root, 'data', 'kdd99_head.dat')


def create_dataset(data_labels, data_nums):
    file_output = os.path.join(path_root, 'data', '多分类 data3', 'kdd99_new_multi.dat')
    outputs = open(file_output, 'w')
    with open(file_head) as f:
        for row in f:
            outputs.write(row)
        outputs.write('\n')
    for data_label, data_num in zip(data_labels, data_nums):
        inputs = open(os.path.join(path_root, 'data', 'KDD99', 'kdd99_{}.kdd99'.format(data_label)))
        for i, row in enumerate(inputs):
            if i == data_num:
                break
            outputs.write(row)

        inputs.close()
    outputs.close()

    return file_output.split(os.sep)[-1]


if __name__ == '__main__':
    data_name = create_dataset(['normal', 'ipsweep'], (1000, 360))
    print(data_name)
