# -*- coding: utf-8 -*-
"""
MNIST 手写体识别实验（MindSpore 纯脚本版）
用法：在实验目录下执行
    python mnist_train.py
可选参数：
    --epochs 3          训练轮数（默认 3）
    --batch-size 32     批量大小（默认 32）
    --ckpt ckpt/lenet5.ckpt  模型保存路径（默认不保存）
"""
import argparse
import os
import time

import numpy as np

import mindspore.dataset as ds
import mindspore.dataset.transforms as C
import mindspore.dataset.vision as CV
from mindspore.common import dtype as mstype
from mindspore.common.initializer import Normal
import mindspore.nn as nn
from mindspore.train.metrics import Accuracy
from mindspore.train import Model


# ---------------- 数据处理 ----------------
def create_dataset(data_path, batch_size=32):
    """
    数据预处理与批量输出
    流程：加载 > shuffle > map(标准化/通道转换/标签类型) > batch > repeat
    """
    data = ds.MnistDataset(data_path)
    data = data.shuffle(buffer_size=10000)

    # MNIST 数据集的 mean = 33.3285, std = 78.5655
    mean, std = 33.3285, 78.5655

    rescale_op = CV.Rescale(1.0 / std, -mean / std)   # 标准化 image = (image-mean)/std
    hwc2chw_op = CV.HWC2CHW()                          # 通道前移 HWC -> CHW
    type_cast_op = C.TypeCast(mstype.int32)            # 标签转 int32

    data = data.map(operations=type_cast_op, input_columns='label')
    data = data.map(operations=rescale_op, input_columns='image')
    data = data.map(operations=hwc2chw_op, input_columns='image')

    data = data.batch(batch_size)
    data = data.repeat(1)
    return data


# ---------------- 网络定义（LeNet5） ----------------
class LeNet5(nn.Cell):
    def __init__(self, num_class=10, num_channel=1):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(num_channel, 6, 5, pad_mode='valid')
        self.conv2 = nn.Conv2d(6, 16, 5, pad_mode='valid')
        self.fc1 = nn.Dense(4 * 4 * 16, 120, weight_init=Normal(0.02))
        self.fc2 = nn.Dense(120, 84, weight_init=Normal(0.02))
        self.fc3 = nn.Dense(84, num_class, weight_init=Normal(0.02))
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

    def construct(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--ckpt', type=str, default='')
    args = parser.parse_args()

    train_path = os.path.join('data', 'train')
    test_path = os.path.join('data', 'test')

    train_data = create_dataset(train_path, args.batch_size)
    test_data = create_dataset(test_path, args.batch_size)
    print('训练集批数: %d, 测试集批数: %d' % (
        train_data.get_dataset_size(), test_data.get_dataset_size()))

    net = LeNet5()
    net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    net_opt = nn.Momentum(net.trainable_params(), 0.01, 0.9)
    model = Model(net, net_loss, net_opt, metrics={'accuracy': Accuracy()})

    print('>>> 开始训练，共 %d 个 epoch ...' % args.epochs)
    t0 = time.time()
    model.train(args.epochs, train_data, dataset_sink_mode=False)
    train_time = time.time() - t0

    print('>>> 在测试集上评估 ...')
    t0 = time.time()
    acc = model.eval(test_data, dataset_sink_mode=False)
    eval_time = time.time() - t0
    print('测试集结果:', acc)
    print('训练耗时: %.1f 秒, 评估耗时: %.1f 秒' % (train_time, eval_time))

    print('>>> 前 10 张测试图片预测展示:')
    ds_demo = create_dataset(test_path, batch_size=1)
    for i, dic in enumerate(ds_demo.create_dict_iterator()):
        output = model.predict(dic['image'])
        predict = int(np.argmax(output.asnumpy(), axis=1)[0])
        print('真实: %s, 预测: %s' % (dic['label'], predict))
        if i >= 9:
            break

    if args.ckpt:
        os.makedirs(os.path.dirname(args.ckpt), exist_ok=True)
        from mindspore import save_checkpoint
        save_checkpoint(net, args.ckpt)
        print('模型已保存到:', args.ckpt)


if __name__ == '__main__':
    main()
