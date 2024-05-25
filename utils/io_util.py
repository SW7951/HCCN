import os
import pickle
import re
import shutil

import yaml
from addict import Dict


def make_directory(directory, delete_old=False):
    """
    创建文件夹，若文件夹不存在则创建
    :param directory: 文件夹
    :return: 文件夹名称
    """
    # 若文件夹已存在、且要求删除时则删除
    if delete_old and os.path.exists(directory):
        shutil.rmtree(directory)
    # 若文件夹不存在则创建
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory


def read_object(file):
    """
    读取对象
    :param file: 存储对象的文件
    :return: 对象
    """
    with open(file, 'rb') as f:
        return pickle.load(f)


def write_object(obj, file):
    """
    写入对象
    :param obj: 要写入的对象
    :param file: 文件路径
    :return: 无
    """
    with open(file, 'wb') as f:
        pickle.dump(obj, f)


def read_text(file, encoding='utf-8'):
    """
    读取文件
    :param file: 要读取的文件
    :param encoding: 文件编码方式
    :return:
    """
    with open(file, 'r', encoding=encoding) as f:
        return f.read()


def write_text(text, file, encoding='utf-8'):
    """
    读取文件
    :param text: 要读取的文件
    :param file: 要读取的文件
    :param encoding: 文件编码方式
    :return:
    """
    with open(file, 'w', encoding=encoding) as f:
        f.write(text)


def read_config(config_file, encoding='utf-8'):
    """
    读取配置
    :param config_file: 配置文件
    :param encoding: 配置文件编码方式
    :return: 配置字典
    """
    with open(config_file, 'r', encoding=encoding) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)  # Loader为了更加安全
    return Dict(config)

def write_config(config, file, encoding='utf-8'):
    """
    写入配置
    :param config: 配置字典
    :param file: 配置文件
    :param encoding: 配置文件编码方式
    :return:
    """
    config = config.to_dict() if isinstance(config, Dict) else config
    with open(file, 'w', encoding=encoding) as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def update_config(config_file, node_path, node_value, encoding='utf-8'):
    """
    写入配置
    :param config_file: 配置文件
    :param node_path: 节点位置
    :param node_value: 新值
    :param encoding: 配置文件编码方式
    :return:
    """
    with open(config_file, 'r', encoding=encoding) as f:
        lines = f.readlines()
    # 读取注释信息
    comments = {}
    for i, line in enumerate(lines):
        if re.match(r'^\s*#', line):
            key = f'comment_{i}'
            comments[key] = line

    config = read_config(config_file, encoding)
    node_list = node_path.split('.')
    # 寻找节点
    current_node = config
    for node in node_list[:-1]:
        current_node = current_node[node]
    # 写入新值
    current_node[node_list[-1]] = node_value



if __name__ == '__main__':
    config = read_config('../conf/settings.yml')
    write_config(config.to_dict(), 'test.yml')
    print(config)