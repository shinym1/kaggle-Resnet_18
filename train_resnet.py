import collections
import math
import os
import shutil
import pandas as pd
import torch
import torchvision
from torch import nn
from d2l import torch as d2l
from colorama import init, AnsiToWin32
import sys
from matplotlib import pyplot as plt
import de_Animator as Animator
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'



transform_train = torchvision.transforms.Compose([
    # 在高度和宽度上将图像放大到40像素的正方形
    torchvision.transforms.Resize(40),
    # 随机裁剪出一个高度和宽度均为40像素的正方形图像，
    # 生成一个面积为原始图像面积0.64～1倍的小正方形，
    # 然后将其缩放为高度和宽度均为32像素的正方形
    torchvision.transforms.RandomResizedCrop(32, scale=(0.64, 1.0),
    ratio=(1.0, 1.0)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    # 标准化图像的每个通道
    torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                    [0.2023, 0.1994, 0.2010])])

transform_test = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                    [0.2023, 0.1994, 0.2010])])


data_dir = 'C:/Users/Administrator/Desktop/kaggle-CIFAR/data/kaggle_cifar10_tiny/'

train_ds, train_valid_ds = [torchvision.datasets.ImageFolder(
    #数据集中的图像将被组织为按类别分好的子文件夹，其中每个子文件夹代表一个类别，并包含属于该类别的图像文件。
    os.path.join(data_dir, 'train_valid_test', folder),
    transform=transform_train) for folder in ['train', 'train_valid']]

valid_ds, test_ds = [torchvision.datasets.ImageFolder(
    os.path.join(data_dir, 'train_valid_test', folder),
    transform=transform_test) for folder in ['valid', 'test']]

batch_size = 64

train_iter, train_valid_iter = [torch.utils.data.DataLoader(
    dataset, batch_size, shuffle=True, drop_last=True)
    for dataset in (train_ds, train_valid_ds)]

valid_iter = torch.utils.data.DataLoader(valid_ds, batch_size, shuffle=False,
drop_last=True)

test_iter = torch.utils.data.DataLoader(test_ds, batch_size, shuffle=False,
drop_last=False)


def get_net():
    num_classes = 10
    net = d2l.resnet18(num_classes, 3)
    return net
loss = nn.CrossEntropyLoss(reduction="none")


def train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,lr_decay):
    #trainer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9,weight_decay=wd)
    trainer = torch.optim.Adam(net.parameters(), lr=lr,weight_decay=wd)
    #设置L2正则化系数wd
    scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_period, lr_decay)
    #在 StepLR 调度器中，学习率将在每个 step_size 个 epoch 之后进行衰减，衰减系数为 gamma。衰减后的学习率将用于下一个 epoch。
    num_batches, timer = len(train_iter), d2l.Timer()
    #num_batches代表批次数量
    legend = ['train loss', 'train acc']
    if valid_iter is not None:
        legend.append('valid acc')

    animator = Animator.Animator(xlabel='epoch', ylabel='accuracy/loss',xlim=[1, num_epochs],legend=legend)

    # xlim控制x轴的显示范围
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    # 将一个单 GPU 的模型并行地复制到多个 GPU 上，以实现数据的并行处理和模型的加速。
    def init_weights(m):
        if type(m) in [nn.Linear, nn.Conv2d]:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    for epoch in range(num_epochs):
        net.train()
        metric = d2l.Accumulator(3) # 三个数据分别是训练损失总和、训练准确度总和、样本数
        for i, (features, labels) in enumerate(train_iter):  #i是索引，feature是图像，labels是标签
            timer.start()
            l, acc = d2l.train_batch_ch13(net, features, labels, loss, trainer, devices)
            metric.add(l, acc, labels.shape[0])
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                # d2l.plot( epoch + (i + 1)/num_batches, (metric[0] / metric[2], metric[1] / metric[2],None),legend=legend )
                # d2l.plt.show()
                animator.add(epoch + (i + 1) / num_batches,(metric[0] / metric[2], metric[1] / metric[2],None))
            #每个周期间隔5绘制一个点，表示的分别为每个样本的平均损失，平均准确度

        if valid_iter is not None:
            valid_acc = d2l.evaluate_accuracy_gpu(net, valid_iter)
            animator.add(epoch + 1, (None, None, valid_acc))
            scheduler.step()
        measures = (f'train loss {metric[0] / metric[2]:.3f}, 'f'train acc {metric[1] / metric[2]:.3f}')

    if valid_iter is not None:
        measures += f', valid acc {valid_acc:.3f}'
        print(measures + f'\n{metric[2] * num_epochs / timer.sum():.1f}'f' examples/sec on {str(devices)}')

    #d2l.plot(animator.xdata,animator.ydata)
    #plt.savefig(os.path.join(data_dir, 'accuracy.png'))
    #plt.show()
    #plt.pause(10)

devices, num_epochs, lr, wd = d2l.try_all_gpus(), 20, 2e-4, 5e-3
lr_period, lr_decay, net = 4, 0.9, get_net()


plt.ion()
train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,lr_decay)
plt.ioff()



