
# coding: utf-8
"""
    将原始数据集进行划分成训练集、验证集和测试集或整体成为测试集
"""
import numpy as np
import os
import glob
import random

target_jpg_root_dir = './JPEGImages/'
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


def make_indices_file():
    xml_file_path_list = glob.glob(target_xml_root_dir + '*.xml')
    xml_num = len(xml_file_path_list)
    print('load %d xml file' % xml_num)
    return xml_file_path_list


def change_name(file_path_list):

    for file_path in file_path_list:
        xml_file_path = file_path
        xml_name = file_path.split('/')[-1]
        name_str = xml_name.split('.')[0]
        idx_str = name_str.split('_')[-1]
        idx = int(idx_str)
        new_xml_file_path = os.path.join(target_xml_root_dir, '%06.f.xml' % idx)
        jpg_file_path = os.path.join(target_jpg_root_dir, name_str + '.jpg')
        new_jpg_file_path = os.path.join(target_jpg_root_dir, '%06.f.jpg' % idx)
        # # jpg_file
        source_file = jpg_file_path
        target_file = new_jpg_file_path
        os.rename(source_file, target_file)
        # xml_file
        source_file = xml_file_path
        target_file = new_xml_file_path
        os.rename(source_file, target_file)

        print('%06.f' % idx)


if __name__ == '__main__':
    file_path_list = make_indices_file()
    change_name(file_path_list)
