import glob

import xml.etree.ElementTree as ET

import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


xml_root_dir = './Annotations/'


def show_ori_wh_point(x, y):
    # 绘制曲线
    plt.ion()
    plt.figure('wh ori', figsize=(8, 6))
    plt.scatter(x, y, alpha=0.5, edgecolors='white')  # plot绘制散点图
    plt.show()


def show_ori_r_point(r):
    # 绘制曲线
    plt.ion()
    plt.figure('r ori', figsize=(8, 6))
    y = np.zeros(np.shape(r))
    plt.scatter(r, y, alpha=0.5, edgecolors='white')  # plot绘制散点图
    plt.show()


def show_fit_wh_point(x, y, labels, cluster_centers_t):
    # 绘制曲线
    plt.ion()
    c_list = ['r', 'g', 'y', 'k']
    plt.figure('wh fit', figsize=(8, 6))
    for idx in range(3):
        keep_idx = [i for i, x in enumerate(labels) if x == idx]
        plt.scatter(x[keep_idx], y[keep_idx], s=20, c=c_list[idx], alpha=0.5, edgecolors='white')  # plot绘制散点图
        w, h = cluster_centers_t[idx]
        plt.scatter(w, h, s=50, c=c_list[idx], alpha=0.5, edgecolors='black')  # plot绘制散点图
    # 
    save_name_str = './result/wh_fit_result.jpg'
    plt.savefig(save_name_str, dpi=300)  # 保存图象
    plt.show()
    # plt.close()  # 关闭图表


def show_fit_r_point(r, labels, cluster_centers_t):
    # 绘制曲线
    plt.ioff()
    c_list = ['r', 'g', 'y', 'k']
    plt.figure('r fit', figsize=(8, 6))
    for idx in range(3):
        keep_idx = [i for i, x in enumerate(labels) if x == idx]
        y = np.zeros((len(keep_idx), 1))
        plt.scatter(r[keep_idx], y, s=20, c=c_list[idx], alpha=0.5, edgecolors='white')  # plot绘制散点图
        r_centers = cluster_centers_t[idx]
        plt.scatter(r_centers, 0, s=50, c=c_list[idx], alpha=0.5, edgecolors='black')  # plot绘制散点图
    # 
    save_name_str = './result/r_fit_result.jpg'
    plt.savefig(save_name_str, dpi=300)  # 保存图象
    plt.show()
    # plt.close()  # 关闭图表


def z_score(x):
    m = np.mean(x)
    q = np.std(x)
    x -= m
    x /= q
    # print(x)
    return x, m, q


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
            whr = rec2whr(rec)              # xmin,ymin,xmax,ymax ------> w, h , ratio=h/w
            all_whr_list.append(whr)
        # print('ok')
    print('ok')
    return all_whr_list


def value2value(w_value, h_value):

    w, h = w_value, h_value
    r = h / w

    w0 = 16 / np.sqrt(r)
    h0 = 16 * np.sqrt(r)

    s_w = w / w0
    s_h = h / h0
    
    ratio_value = r
    scale_value = np.mean([s_w, s_h])

    return scale_value, ratio_value


def rec2whr(rec):

    xmin, ymin, xmax, ymax = rec

    w = xmax - xmin
    h = ymax - ymin
    r = h / w
    whr = [w, h, r]

    return whr


def wh_kmeans(weight_input, k_value=3):

    #  开始功能
    string = 'K_Means Function Start.'
    print(string)

    # 指定分成 class_num 个类
    k_means = KMeans(n_clusters=k_value)
    k_means.fit(weight_input)

    # 打印出各个族的中心点
    cluster_centers = []
    for center in k_means.cluster_centers_:
        center = [item for item in center]
        cluster_centers.append(center)
        print(center)
    # 分组结果
    labels = []
    for pix_index, class_label in enumerate(k_means.labels_, 1):
        # print("index: {}, label: {}".format(pix_index, class_label))
        labels.append(class_label)

    # 样本距其最近的聚类中心的平方距离之和，用来评判分类的准确度，值越小越好
    # k-means的超参数n_clusters可以通过该值来评估
    print("inertia: {}".format(k_means.inertia_))

    return cluster_centers, labels


def r_kmeans(weight_input, k_value=3):

    #  开始功能
    string = 'K_Means Function Start.'
    print(string)

    # 指定分成 class_num 个类
    k_means = KMeans(n_clusters=k_value)
    k_means.fit(weight_input)

    # 打印出各个族的中心点
    cluster_centers = []
    for center in k_means.cluster_centers_:
        center = [item for item in center]
        cluster_centers.append(center)
        print(center)
    # 分组结果
    labels = []
    for pix_index, class_label in enumerate(k_means.labels_, 1):
        # print("index: {}, label: {}".format(pix_index, class_label))
        labels.append(class_label)

    # 样本距其最近的聚类中心的平方距离之和，用来评判分类的准确度，值越小越好
    # k-means的超参数n_clusters可以通过该值来评估
    print("inertia: {}".format(k_means.inertia_))

    return cluster_centers, labels


def do_wh_kmeans(all_whr_list):
    #
    weight = np.array(all_whr_list, dtype='float64')
    w_weight_ori = weight[:, 0]
    h_weight_ori = weight[:, 1]

    #
    # show_ori_wh_point(w_weight_ori, h_weight_ori)

    # 归一化
    w_weight, w_m, w_q = z_score(w_weight_ori.copy())
    h_weight, h_m, h_q = z_score(h_weight_ori.copy())

    # 拼接
    w_weight = w_weight.reshape(-1, 1)
    h_weight = h_weight.reshape(-1, 1)
    weight = np.concatenate((w_weight, h_weight), axis=1)

    # 计算结果
    scales = []
    cluster_centers_t = []
    cluster_centers, labels = wh_kmeans(weight)
    for point_wh in cluster_centers:
        point_w = point_wh[0]
        point_h = point_wh[1]
        w_value = point_w * w_q + w_m
        h_value = point_h * h_q + h_m
        r_value = h_value / w_value
        string = 'w_cluster_centers : %f | h_cluster_centers : %f ' % (point_w, point_h)
        print(string)
        string = 'w_value : %.3f | h_value : %.3f  | r_value : %.3f  ' %\
                 (w_value, h_value, r_value)
        print(string)

        scale_value, ratio_value = value2value(w_value, h_value)

        string = 'SCALES : %.3f\n' % scale_value
        print(string)
        scales.append(scale_value)

        cluster_centers_t.append([w_value, h_value])

    with open('wh_k_means_result.txt', 'w', encoding='UTF-8') as fh:
        scales.sort()
        SCALES_string = 'ANCHOR_SCALES: [{:.2f}, {:.2f}, {:.2f}]'.format(scales[0], scales[1], scales[2])
        fh.write(SCALES_string + '\n')

    show_fit_wh_point(w_weight_ori, h_weight_ori, labels, cluster_centers_t)


def do_r_kmeans(all_whr_list):
    #
    weight = np.array(all_whr_list, dtype='float64')
    r_weight_ori = weight[:, 2]

    #
    # show_ori_r_point(r_weight_ori)

    # 归一化
    r_weight, r_m, r_q = z_score(r_weight_ori.copy())

    # 拼接
    r_weight = r_weight.reshape(-1, 1)
    zero_weight = np.zeros(np.shape(r_weight))
    weight = np.concatenate((r_weight, zero_weight), axis=1)

    # 计算结果
    ratios = []
    cluster_centers_t = []
    cluster_centers, labels = r_kmeans(weight)
    for point_ry in cluster_centers:
        point_r = point_ry[0]
        r_value = point_r * r_q + r_m
        string = 'r_cluster_centers : %fv' % point_r
        print(string)
        string = 'r_value : %.3f  ' % r_value
        print(string)
        string = 'RATIOS : %.3f\n' % r_value
        print(string)
        ratios.append(r_value)
        cluster_centers_t.append(r_value)

    with open('r_k_means_result.txt', 'w', encoding='UTF-8') as fh:
        ratios.sort()
        RATIOS_string = 'ANCHOR_RATIOS: [{:.2f}, {:.2f}, {:.2f}]'.format(ratios[0], ratios[1], ratios[2])
        fh.write(RATIOS_string + '\n')

    show_fit_r_point(r_weight_ori, labels, cluster_centers_t)


if __name__ == '__main__':
    all_whr_list = find_all_car_rec()
    do_wh_kmeans(all_whr_list)
    do_r_kmeans(all_whr_list)
