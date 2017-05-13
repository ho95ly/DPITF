__author__ = 'Administrator'


# Created by ay27 at 16/3/25

import csv

import numpy


def read_data(file_path):
    tmp = []
    i = 0
    j = 0
    with open(file_path, 'r') as file:
        for line in file:
            num = line.split()
            tmp.append([int(num[0]), int(num[1]), float(num[3])])
            i = max(i, tmp[-1][0])
            j = max(j, tmp[-1][1])
    return i, j, numpy.array(tmp)


def auto_split_data(datas, I, J, cnt):
    every_proc_data_num = int(I / cnt)
    r_set = []
    current_set = []
    cur_index = 1
    for (row, ii) in zip(datas, range(len(datas))):
        if row[0] > every_proc_data_num * cur_index:
            r_set.append(current_set)
            current_set = [row]
            cur_index += 1
        else:
            current_set.append(list(row))
    if len(current_set) > 0 and len(r_set) == cnt:
        r_set[-1].extend(current_set)
    elif len(current_set) > 0 and len(r_set) < cnt:
        r_set.append(current_set)

    return r_set

'''
if __name__ == '__main__':
    # train_21568528.txt
    I, J, datas = read_data('data/movie_718user_all.txt')
    datas = sorted(datas, key=lambda row: row[0])
    sets = auto_split_data(datas, I, J, 12)
    for ii in range(len(sets)):
        f = open('data/tmpdata/data_%d.csv' % ii, 'w')
        writer = csv.writer(f)
        writer.writerows(sets[ii])

'''
def data_reader():
    I, J, datas = read_data('data/movie_718user_all.txt')
    datas = sorted(datas, key=lambda row: row[0])
    f = open('data/tmpdata/test.csv', 'w')
    writer = csv.writer(f)
    writer.writerows(datas)

data_reader()

# # Created by ay27 at 16/3/15
#
#
# def _count(data):
#     N = 0
#     M = 0
#     for row in data:
#         N = max(N, row[0])
#         M = max(M, row[1])
#     return N, M
#
#
# def read_data(file_path):
#     tmp = []
#     with open(file_path, 'r') as file:
#         for line in file:
#             tmp.append([float(word) for word in line.split()])
#     N, M = _count(tmp)
#     return N, M, tmp
#
# def write_data(tmp, file_path):
#     with open(file_path, 'w') as file:
#         for row in tmp:
#             file.write('%d %d %d\n' % (row[0], row[1], row[3]))
#
# if __name__ == '__main__':
#     N, M, tmp = read_data('movie_718user_all.txt')
#     write_data(tmp, 'movie_data_718.txt')
#
