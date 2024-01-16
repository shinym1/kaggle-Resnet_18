import collections
import math
import os
import shutil
import pandas as pd
import torch
import torchvision
from torch import nn
from d2l import torch as d2l


def read_csv_labels(fname):
    with open(fname,'r') as f:
        lines = f.readlines()[1:]
    tokens = [l.rstrip().split(',')  for l in lines]
    return dict((name,label) for name,label in tokens)





def copyfile(filename, target_dir):
    """将文件复制到目标目录"""
    os.makedirs(target_dir, exist_ok=True)
    shutil.copy(filename, target_dir)

def reorg_train_valid(data_dir, labels, valid_ratio):
    """将验证集从原始的训练集中拆分出来"""
    # 训练数据集中样本最少的类别中的样本数
    n = collections.Counter(labels.values()).most_common()[-1][1]  #使用索引 -1 取出列表中的最后一个元素，即出现频率最低的元素及其频率。
    # 验证集中每个类别的样本数
    n_valid_per_label = max(1, math.floor(n * valid_ratio))
    label_count = {}
    for train_file in os.listdir(os.path.join(data_dir, 'train')):
        label = labels[train_file.split('.')[0]]    #先提取不含扩展名的文件名，然后寻找对应的标记
        fname = os.path.join(data_dir, 'train', train_file)
        copyfile(fname, os.path.join(data_dir, 'train_valid_test',
        'train_valid', label))
        if label not in label_count or label_count[label] < n_valid_per_label:
            copyfile(fname, os.path.join(data_dir, 'train_valid_test',
            'valid', label))
            label_count[label] = label_count.get(label, 0) + 1   #键存在则返回原本的值，不存在则返回0
        else:
            copyfile(fname, os.path.join(data_dir, 'train_valid_test',
                                         'train', label))
    return n_valid_per_label



def reorg_test(data_dir):
    """在预测期间整理测试集，以方便读取"""
    for test_file in os.listdir(os.path.join(data_dir, 'test')):
        copyfile(os.path.join(data_dir, 'test', test_file),
        os.path.join(data_dir, 'train_valid_test', 'test','unknown'))



def reorg_cifar10_data(data_dir, valid_ratio):
    labels = read_csv_labels(os.path.join(data_dir, 'trainLabels.csv'))
    reorg_train_valid(data_dir, labels, valid_ratio)
    reorg_test(data_dir)






data_dir = 'C:/Users/Administrator/Desktop/kaggle-CIFAR/data/kaggle_cifar10_tiny/'
batch_size = 32
valid_ratio = 0.1
reorg_cifar10_data(data_dir, valid_ratio)


