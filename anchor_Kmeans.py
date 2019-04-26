from xml.dom import minidom
import os
import glob
import shutil

import xml.etree.ElementTree as ET
import os
import pickle
import numpy as np
from sklearn.cluster import KMeans


xml_root_dir = '/media/kevin/文档/DataSet/VOCCARdevkit/VOCCar2010/Annotations/'
target_xml_root_dir = './Annotations/'


def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    rec_list = []
    for obj in tree.findall('object'):
        # print(obj.find('name').text)
        if obj.find('name').text == 'car':
            bbox = obj.find('bndbox')
            rec_list.append([int(bbox.find('xmin').text),
                             int(bbox.find('ymin').text),
                             int(bbox.find('xmax').text),
                             int(bbox.find('ymax').text)])

    return rec_list


def find_all_car_rec():
    xml_path_list = glob.glob(xml_root_dir + '*.xml')
    xml_num = len(xml_path_list)
    print(xml_num)
    all_rec_list = []
    all_whr_list = []
    rec_list = []
    for xml_path in xml_path_list:
        # print('find car rec in file : %s' % xml_path)
        rec_list = parse_rec(xml_path)

        for rec in rec_list:
            all_rec_list.append(rec)
            whr = rec2whr(rec)              # xmin,ymin,xmax,ymax ------> w, h ,ratio   ratio=h/w
            all_whr_list.append(whr)
        # print('ok')
    print('ok')

    # 最大值最小值
    weight = np.array(all_whr_list)
    w_weight = weight[:, 0]
    h_weight = weight[:, 1]
    r_weight = weight[:, 2]

    w_min = np.min(w_weight)
    h_min = np.min(h_weight)
    w_max = np.max(w_weight)
    h_max = np.max(h_weight)
    print('w_min : ', w_min, '  ANCHOR_SCALES value : ', round(w_min/16))    # 'ANCHOR_RATIOS', '[0.5,1,2]'
    print('h_min : ', h_min, '  ANCHOR_SCALES value : ', round(h_min/16))
    print('w_max : ', w_max, '  ANCHOR_SCALES value : ', round(w_max/16))
    print('h_max : ', h_max, '  ANCHOR_SCALES value : ', round(h_max/16))

    with open('k_means_result.txt', 'w', encoding='UTF-8') as fh:

        fh.write('w_min : ' + str(w_min) + '  ANCHOR_SCALES value : ' + str(w_min/16) + '\n')
        fh.write('h_min : ' + str(h_min) + '  ANCHOR_SCALES value : ' + str(h_min/16) + '\n')
        fh.write('w_max : ' + str(w_max) + '  ANCHOR_SCALES value : ' + str(w_max/16) + '\n')
        fh.write('h_max : ' + str(h_max) + '  ANCHOR_SCALES value : ' + str(h_max/16) + '\n')

        w_cluster_centers = whr_kmeans(w_weight)
        for point_xy in w_cluster_centers:
            point_y = point_xy[1]
            string = 'w_cluster_centers : ' + str(point_y) + '  ANCHOR_SCALES value : ' + str(point_y/16)
            print(string)
            fh.write(string + '\n')

        h_cluster_centers = whr_kmeans(h_weight)
        for point_xy in h_cluster_centers:
            point_y = point_xy[1]
            string = 'h_cluster_centers : ' + str(point_y) + '  ANCHOR_SCALES value : ' + str(point_y/16)
            print(string)
            fh.write(string + '\n')

        r_cluster_centers = whr_kmeans(r_weight)
        for point_xy in r_cluster_centers:
            point_y = point_xy[1]
            string = 'r_cluster_centers : ' + str(point_y) + '  ANCHOR_RATIOS value : ' + str(point_y)
            print(string)
            fh.write(string + '\n')


def rec2whr(rec):

    xmin, ymin, xmax, ymax = rec

    w = xmax - xmin
    h = ymax - ymin
    ratio = h / w
    whr = [w, h, ratio]

    return whr


def whr_kmeans(weight_input, k_value=3):

    #  开始功能
    string = 'K_Means RGB IMG Function Start.'
    print(string)

    # 1 加载语料
    # 显示开始信息
    string = '1-> 加载像素内容'
    print(string)
    d = len(weight_input)
    weight = np.zeros((d, 2))
    for i in range(d):
        weight[i, :] = [0, weight_input[i]]
    weight = np.array(weight, dtype=np.float64)

    # 2 对向量进行聚类
    # 显示开始信息
    string = '2-> 进行聚类'
    print(string)

    # 指定分成 class_num 个类
    k_means = KMeans(n_clusters=k_value)
    k_means.fit(weight)

    # 打印出各个族的中心点
    cluster_centers = []
    for center in k_means.cluster_centers_:
        center = [item for item in center]
        cluster_centers.append(center)
        print(center)
    # 分组结果
    # labels = []
    # for pix_index, class_label in enumerate(k_means.labels_, 1):
    #     print("index: {}, label: {}".format(pix_index, class_label))
    #     labels.append(class_label)

    # 样本距其最近的聚类中心的平方距离之和，用来评判分类的准确度，值越小越好
    # k-means的超参数n_clusters可以通过该值来评估
    print("inertia: {}".format(k_means.inertia_))

    return cluster_centers


if __name__ == '__main__':
    find_all_car_rec()
