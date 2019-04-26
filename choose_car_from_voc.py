from xml.dom import minidom
import os
import glob
import shutil


xml_root_dir = '/media/kevin/文档/DataSet/VOCdevkit/VOC2007/Annotations/'
jpg_root_dir = '/media/kevin/文档/DataSet/VOCdevkit/VOC2007/JPEGImages/'
target_xml_root_dir = './Annotations/'
target_jpg_root_dir = './JPEGImages/'
if not os.path.exists(target_xml_root_dir):
    os.makedirs(target_xml_root_dir)
if not os.path.exists(target_jpg_root_dir):
    os.makedirs(target_jpg_root_dir)


def find_file():
    xml_path_list = glob.glob(xml_root_dir + '*.xml')
    xml_num = len(xml_path_list)
    print(xml_num)
    for xml_path in xml_path_list:
        print('find car in file : %s' % xml_path)
        if is_car(xml_path):
            # print('find car in file.')
            copy_refine_file(xml_path)
        else:
            # print('find no car in file.')
            continue


def is_car(xml_path):
    # parse()获取DOM对象
    dom = minidom.parse(xml_path)
    # 获取根节点
    root = dom.documentElement
    # 通过dom对象或根元素，再根据标签名获取元素节点，是个列表
    name_node_list = root.getElementsByTagName('name')
    for name_node in name_node_list:
        name_str = name_node.firstChild.data
        if name_str in ['car']:
            return True
    return False


def copy_refine_file(xml_path):
    xml_name = xml_path.split('/')[-1]
    jpg_name = xml_name[:-3] + 'jpg'
    jpg_path = os.path.join(jpg_root_dir, jpg_name)
    # jpg_file
    source_file = jpg_path
    target_file = os.path.join(target_jpg_root_dir, jpg_name)
    shutil.copy(source_file, target_file)
    # print('copy file : %s  to %s.' % (source_file, target_file))
    # xml_file
    source_file = xml_path
    target_file = os.path.join(target_xml_root_dir, xml_name)
    shutil.copy(source_file, target_file)
    # print('copy file : %s  to %s.' % (source_file, target_file))

    # check xml file
    check_xml_file(target_file)


def check_xml_file(xml_path):
    # parse()获取DOM对象
    dom = minidom.parse(xml_path)
    # 获取根节点
    root = dom.documentElement
    # 通过dom对象或根元素，再根据标签名获取元素节点，是个列表
    object_node_list = root.getElementsByTagName('object')
    before_num = len(object_node_list)
    # print('before :', before_num)
    for object_node in object_node_list:
        name_node = object_node.getElementsByTagName('name')[0]
        name_str = name_node.firstChild.data
        if name_str not in ['car']:
            root.removeChild(object_node)
            # print('file %s remove node %s' % (xml_path, name_str))
    # 通过dom对象或根元素，再根据标签名获取元素节点，是个列表
    object_node_list = root.getElementsByTagName('object')
    after_num = len(object_node_list)
    # print('after :', after_num)

    if after_num != before_num:
        # print('change xml file : %s' % xml_path)
        try:
            with open(xml_path, 'w', encoding='UTF-8') as fh:
                # 4.writexml() 目标文件对象，缩进格式，子节点的缩进格式，换行格式，xml内容的编码。
                dom.writexml(fh, indent='', addindent='\t', newl='\n', encoding='UTF-8')
                # print('写入xml OK!')
            fix_xml_file(xml_path)
        except Exception as err:
            print('错误信息：{0}'.format(err))
    else:
        # print('no need to change xml file')
        pass


def fix_xml_file(xml_path):
    with open(xml_path, 'r', encoding='UTF-8') as fh:
        lines = fh.readlines()

    new_lines = []
    for line in lines[1:]:
        if line.strip() != '':
            new_lines.append(line)

    with open(xml_path, 'w', encoding='UTF-8') as fh:
        fh.writelines(new_lines)


if __name__ == '__main__':
    find_file()
    # change_xml_file()
    # xml_path = '/media/kevin/文档/PycharmProjects/find_car_in_voc/Annotations/000014.xml'
    # fix_xml_file(xml_path)

