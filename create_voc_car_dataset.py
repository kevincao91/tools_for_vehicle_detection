# coding: utf-8
"""
    将原始数据集进行划分成训练集、验证集和测试集或整体成为测试集
"""
import numpy as np
import os
import glob
import random


target_xml_root_dir = './Annotations/'
target_set_root_dir = './ImageSets/Main/'
train_file_path = 'train.txt'
valid_file_path = 'val.txt'
trainval_file_path = 'trainval.txt'
test_file_path = 'test.txt'

if not os.path.exists('./ImageSets/'):
    os.makedirs('./ImageSets/')
if not os.path.exists(target_set_root_dir):
    os.makedirs(target_set_root_dir)


train_percent = 0.5
valid_percent = 0.2
test_percent = 0.3


def make_indices_file():
    xml_file_path_list = glob.glob(target_xml_root_dir + '*.xml')
    xml_num = len(xml_file_path_list)
    print('load %d xml file' % xml_num)
    return xml_file_path_list


def make_file(file_path_list):

    train_file = open(os.path.join(target_set_root_dir, train_file_path), 'w', encoding='UTF-8')
    val_file = open(os.path.join(target_set_root_dir, valid_file_path), 'w', encoding='UTF-8')
    trainval_file = open(os.path.join(target_set_root_dir, trainval_file_path), 'w', encoding='UTF-8')
    test_file = open(os.path.join(target_set_root_dir, test_file_path), 'w', encoding='UTF-8')

    # train & valid
    random.seed(666)
    random.shuffle(file_path_list)
    file_num = len(file_path_list)

    train_point = int(file_num * train_percent)
    trainval_point = int(file_num * (train_percent + valid_percent))

    for i in range(file_num):
        line = file_path_list[i][-10:-4] + '\n'
        if i < train_point:
            train_file.write(line)
            trainval_file.write(line)
        elif i < trainval_point:
            val_file.write(line)
            trainval_file.write(line)
        else:
            test_file.write(line)

    train_file.close()
    val_file.close()
    trainval_file.close()
    test_file.close()

    print('all file:{}, train:{}, valid:{}, test:{}'.format(file_num, train_point, trainval_point-train_point,
                                                            file_num-trainval_point))


if __name__ == '__main__':
    file_path_list = make_indices_file()
    make_file(file_path_list)
