# -*- coding: utf-8 -*-
"""
MNIST 手写体识别实验（TensorFlow/Keras 版，用于与 MindSpore 对比）
用法：python mnist_train_tf.py
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
import tensorflow as tf


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


def load_dataset(data_root):
    train_x = load_idx(os.path.join(data_root, 'train', 'train-images.idx3-ubyte'))
    train_y = load_idx(os.path.join(data_root, 'train', 'train-labels.idx1-ubyte'))
    test_x = load_idx(os.path.join(data_root, 'test', 't10k-images.idx3-ubyte'))
    test_y = load_idx(os.path.join(data_root, 'test', 't10k-labels.idx1-ubyte'))
    return (train_x, train_y), (test_x, test_y)


def preprocess(x, y):
    x = (x.astype(np.float32) - 33.3285) / 78.5655
    x = x[..., np.newaxis]              # HWC，最后加通道维
    return x, y.astype(np.int32)


def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(6, 5, activation='relu'),        # 28x28x1 -> 24x24x6
        tf.keras.layers.MaxPooling2D(2),                        # -> 12x12x6
        tf.keras.layers.Conv2D(16, 5, activation='relu'),       # -> 8x8x16
        tf.keras.layers.MaxPooling2D(2),                        # -> 4x4x16
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(120, activation='relu'),
        tf.keras.layers.Dense(84, activation='relu'),
        tf.keras.layers.Dense(10),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()

    (train_x, train_y), (test_x, test_y) = load_dataset('data')
    train_x, train_y = preprocess(train_x, train_y)
    test_x, test_y = preprocess(test_x, test_y)
    print('训练集: %d 张, 测试集: %d 张' % (len(train_x), len(test_x)))

    model = build_model()

    print('>>> 开始训练，共 %d 个 epoch ...' % args.epochs)
    t0 = time.time()
    model.fit(train_x, train_y, batch_size=args.batch_size,
              epochs=args.epochs, verbose=2)
    train_time = time.time() - t0

    print('>>> 在测试集上评估 ...')
    t0 = time.time()
    loss, acc = model.evaluate(test_x, test_y, batch_size=256, verbose=0)
    eval_time = time.time() - t0
    print('测试集结果: accuracy=%.4f, loss=%.4f' % (acc, loss))
    print('训练耗时: %.1f 秒, 评估耗时: %.1f 秒' % (train_time, eval_time))


if __name__ == '__main__':
    main()
