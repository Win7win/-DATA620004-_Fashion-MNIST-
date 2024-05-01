import os
import gzip
import numpy as np


def load_data(path, kind):
    # 加载数据
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels


# 调整y
def integer_to_one_hot(labels, num_classes):
    return np.eye(num_classes)[labels]


def get_vali(x_train, y_train, split_ratio):
    # 随机化数据集
    random_indices = np.random.permutation(len(x_train))
    x_train_shuffled = x_train[random_indices]
    y_train_shuffled = y_train[random_indices]

    # 分割数据为训练集和验证集
    split_index = int(len(x_train_shuffled) * split_ratio)

    x_train_split, x_vali_split = x_train_shuffled[:split_index], x_train_shuffled[split_index:]
    y_train_split, y_vali_split = y_train_shuffled[:split_index], y_train_shuffled[split_index:]
    return x_train_split, y_train_split, x_vali_split, y_vali_split


def gen_batch(x_data, y_data, batch_size=32):
    # 生成batch
    n_samples = x_data.shape[0]
    indices = np.random.permutation(n_samples)
    x_data_batched = []
    y_data_batched = []
    for i in range(0, n_samples, batch_size):
        batch_indices = indices[i:i + batch_size]
        x_data_batched.append(x_data[batch_indices])
        y_data_batched.append(y_data[batch_indices])
    return [x_data_batched, y_data_batched]


def get_data_fashion_mnist(path, vali_split_ratio=0.9, batch_size=32):
    # 加载数据
    x_train, y_train = load_data(path, "train")
    x_train = x_train.astype('float32') / 255
    y_train = y_train.astype('int32')
    x_test, y_test = load_data(path, "t10k")
    x_test = x_test.astype('float32') / 255
    y_test = y_test.astype('int32')
    # 切出验证集
    x_train, y_train, x_vali, y_vali = get_vali(x_train, y_train, vali_split_ratio)
    # 生成batch
    batches = []
    batches.append(gen_batch(x_train, y_train, batch_size))
    batches.append(gen_batch(x_vali, y_vali, batch_size))
    batches.append(gen_batch(x_test, y_test, batch_size))
    return batches
