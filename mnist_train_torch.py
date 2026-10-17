# -*- coding: utf-8 -*-
"""
MNIST 手写体识别实验（PyTorch 版，用于与 MindSpore 对比）
用法：python mnist_train_torch.py
与 MindSpore 版保持一致：
    - 网络结构 LeNet5（Conv6-5x5 > MaxPool > Conv16-5x5 > MaxPool > FC120 > FC84 > FC10）
    - 归一化 (x - 33.3285) / 78.5655
    - 优化器 SGD(momentum=0.9)，lr=0.01
    - 3 个 epoch，batch_size=32
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# ---------------- 读取 MNIST idx 原始文件 ----------------
def load_idx(path):
    with open(path, 'rb') as f:
        data = f.read()
    magic = int.from_bytes(data[0:4], 'big')
    assert magic in (2051, 2049), '不是合法的 MNIST idx 文件: %s' % path
    n = int.from_bytes(data[4:8], 'big')
    if magic == 2051:   # 图像
        rows, cols = int.from_bytes(data[8:12], 'big'), int.from_bytes(data[12:16], 'big')
        arr = np.frombuffer(data, dtype=np.uint8, offset=16).reshape(n, rows, cols)
    else:               # 标签
        arr = np.frombuffer(data, dtype=np.uint8, offset=8).reshape(n)
    return arr


# ---------------- 网络定义（LeNet5） ----------------
class LeNet5(nn.Module):
    def __init__(self, num_class=10, num_channel=1):
        super(LeNet5, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(num_channel, 6, 5),       # 28x28x1 -> 24x24x6
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                 # -> 12x12x6
            nn.Conv2d(6, 16, 5),                # -> 8x8x16
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                 # -> 4x4x16
            nn.Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(4 * 4 * 16, 120), nn.ReLU(),
            nn.Linear(120, 84), nn.ReLU(),
            nn.Linear(84, num_class),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def make_loader(x, y, batch_size, shuffle):
    x = (x.astype(np.float32) - 33.3285) / 78.5655   # 归一化
    x = torch.from_numpy(x).unsqueeze(1)             # N,H,W -> N,C,H,W
    y = torch.from_numpy(y.astype(np.int64))
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=shuffle)


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            pred = model(x.to(device)).argmax(dim=1).cpu()
            correct += (pred == y).sum().item()
            total += len(y)
    return correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()

    device = torch.device('cpu')
    print('设备: %s' % device)

    train_x = load_idx(os.path.join('data', 'train', 'train-images.idx3-ubyte'))
    train_y = load_idx(os.path.join('data', 'train', 'train-labels.idx1-ubyte'))
    test_x = load_idx(os.path.join('data', 'test', 't10k-images.idx3-ubyte'))
    test_y = load_idx(os.path.join('data', 'test', 't10k-labels.idx1-ubyte'))
    print('训练集: %d 张, 测试集: %d 张' % (len(train_x), len(test_x)))

    train_loader = make_loader(train_x, train_y, args.batch_size, shuffle=True)
    test_loader = make_loader(test_x, test_y, 256, shuffle=False)

    torch.manual_seed(1)
    model = LeNet5()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    print('>>> 开始训练，共 %d 个 epoch ...' % args.epochs)
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        running_loss, step = 0.0, 0
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(x.to(device)), y.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            step += 1
        print('epoch %d/%d, 平均损失: %.4f' % (epoch + 1, args.epochs, running_loss / step))
    train_time = time.time() - t0

    print('>>> 在测试集上评估 ...')
    t0 = time.time()
    acc = evaluate(model, test_loader, device)
    eval_time = time.time() - t0
    print('测试集结果: accuracy=%.4f' % acc)
    print('训练耗时: %.1f 秒, 评估耗时: %.1f 秒' % (train_time, eval_time))


if __name__ == '__main__':
    main()
