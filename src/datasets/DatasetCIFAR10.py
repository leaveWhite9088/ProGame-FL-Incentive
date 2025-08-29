import torch
from torch.utils.data import Dataset
import numpy as np
import pickle
import os

class CIFAR10Dataset(Dataset):
    def __init__(self, batch_files=None, images_path=None, labels_path=None, transform=None):
        """
        初始化CIFAR10数据集
        
        支持两种初始化方式：
        1. 从批处理文件加载（CIFAR原始格式）
        2. 从图像和标签路径加载（类似MNIST格式，如果实现的话）

        :param batch_files: 包含所有批处理文件路径的列表（例如 ['data_batch_1', ..., 'test_batch']）
        :param images_path: 图像文件路径（如果使用自定义格式）
        :param labels_path: 标签文件路径（如果使用自定义格式）
        :param transform: 可选的图像转换（如数据增强、归一化等）
        """
        self.transform = transform
        
        if batch_files is not None:
            # 从CIFAR原始批处理文件加载
            self.data, self.labels = self._load_batches(batch_files)
        elif images_path is not None and labels_path is not None:
            # 从图像和标签路径加载（如果需要实现这种格式）
            self.data, self.labels = self._load_images_and_labels(images_path, labels_path)
        else:
            raise ValueError("必须提供batch_files或images_path和labels_path")

    def _load_batches(self, batch_files):
        """
        从批处理文件加载CIFAR10数据

        :param batch_files: 批处理文件路径列表
        :return: (data, labels) 两个numpy数组
        """
        all_data = []
        all_labels = []
        for batch_file in batch_files:
            if not os.path.isfile(batch_file):
                raise FileNotFoundError(f"Batch file {batch_file} not found.")
            with open(batch_file, 'rb') as f:
                batch = pickle.load(f, encoding='bytes')
                # 验证文件格式
                if b'data' not in batch or b'labels' not in batch:
                    raise ValueError(f"Invalid CIFAR10 batch file format in {batch_file}")
                # 'data' 是一个 (10000, 3072) 的数组，每行是一个展平的图像
                data = batch[b'data']
                labels = batch[b'labels']
                all_data.append(data)
                all_labels.extend(labels)
        all_data = np.vstack(all_data).reshape(-1, 3, 32, 32).astype(np.float32) / 255.0  # 归一化到 [0, 1]
        all_labels = np.array(all_labels, dtype=np.int64)
        return all_data, all_labels

    def _load_images_and_labels(self, images_path, labels_path):
        """
        从图像和标签文件加载数据（可选实现，类似MNIST格式）
        
        :param images_path: 图像文件路径
        :param labels_path: 标签文件路径
        :return: (data, labels) 两个numpy数组
        """
        # 注意：CIFAR10标准格式是批处理文件，此方法为可选实现
        # 如果需要实现此功能，可以参考MNIST的实现方式
        raise NotImplementedError("从图像和标签文件加载CIFAR10数据的功能尚未实现")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """
        获取指定索引的图像及其标签

        :param idx: 索引
        :return: (image_tensor, label)
        """
        image = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            # 如果transform需要PIL Image，需要转换
            # image = transforms.ToPILImage()(torch.tensor(image))
            image = self.transform(image)
        else:
            image = torch.tensor(image)

        return image, label
