import struct
import numpy as np
import torch
from sklearn.preprocessing import PowerTransformer
from torch.utils.data import TensorDataset, DataLoader
import os
import pickle

from src.global_variable import parent_path

class UtilCIFAR10:
    @staticmethod
    def split_data_to_dataowners(dataowners, X_data, y_data):
        """
        将CIFAR10数据集切分成不等的N份，并将数据分配给每个DataOwner对象
        :param dataowners: 一个长度为N的DataOwner对象数组
        :param X_data: 数据集的特征（例如CIFAR10的图像数据，形状为 (num_samples, 3, 32, 32)）
        :param y_data: 数据集的标签
        :return: None (每个DataOwner对象将保存其对应的数据)
        """
        N = len(dataowners)
        total_samples = len(X_data)

        # 生成一个随机的索引排列
        permutation = np.random.permutation(total_samples)

        # 打乱数据
        X_shuffled = X_data[permutation]
        y_shuffled = y_data[permutation]

        # 生成一个不等的切分比例（可以根据需要修改）
        # 例如我们将每个DataOwner分配不同数量的样本，这里以随机切分为例
        split_sizes = np.random.multinomial(total_samples, [1 / N] * N)

        start_idx = 0
        for i, do in enumerate(dataowners):
            end_idx = start_idx + split_sizes[i]
            do.imgData = X_shuffled[start_idx:end_idx]  # 给每个DataOwner分配数据
            do.originalData = X_shuffled[start_idx:end_idx]
            do.labelData = y_shuffled[start_idx:end_idx]
            start_idx = end_idx

    @staticmethod
    def split_data_to_dataowners_with_large_gap(dataowners, X_data, y_data):
        """
        将CIFAR10数据集切分成不等的N份，差距较大，并将数据分配给每个DataOwner对象
        :param dataowners: 一个长度为N的DataOwner对象数组
        :param X_data: 数据集的特征（例如CIFAR10的图像数据，形状为 (num_samples, 3, 32, 32)）
        :param y_data: 数据集的标签
        :return: None (每个DataOwner对象将保存其对应的数据)
        """
        N = len(dataowners)
        total_samples = len(X_data)

        # 生成一个随机的索引排列
        permutation = np.random.permutation(total_samples)

        # 打乱数据
        X_shuffled = X_data[permutation]
        y_shuffled = y_data[permutation]

        # 创建一个不均匀的权重分布，差距大
        # 比如用一个几何分布生成差距大的权重，然后归一化
        raw_weights = np.random.geometric(p=0.2, size=N)  # 几何分布会生成较大的差距
        weights = raw_weights / raw_weights.sum()  # 归一化权重，确保总和为1

        # 根据权重生成切分比例
        split_sizes = (weights * total_samples).astype(int)

        # 修正分配：由于整数化可能导致分配的样本数不完全等于total_samples
        diff = total_samples - split_sizes.sum()
        split_sizes[0] += diff  # 将差值调整到第一个DataOwner

        start_idx = 0
        for i, do in enumerate(dataowners):
            end_idx = start_idx + split_sizes[i]
            do.imgData = X_shuffled[start_idx:end_idx]  # 给每个DataOwner分配数据
            do.originalData = X_shuffled[start_idx:end_idx]
            do.labelData = y_shuffled[start_idx:end_idx]
            start_idx = end_idx

        # 输出权重分布和切分情况
        print("Weights:", weights)
        print("Split sizes:", split_sizes)

    # 将CIFAR10数据集切分成N等份，并将数据分配给每个DataOwner对象
    @staticmethod
    def split_data_to_dataowners_evenly(dataowners, X_data, y_data):
        """
        将CIFAR10数据集切分成N等份，并将数据分配给每个DataOwner对象
        :param dataowners: 一个长度为N的DataOwner对象数组
        :param X_data: 数据集的特征（例如CIFAR10的图像数据）
        :param y_data: 数据集的标签
        :return: None (每个DataOwner对象将保存其对应的数据)
        """
        N = len(dataowners)
        total_samples = len(X_data)

        # 生成一个随机的索引排列
        permutation = np.random.permutation(total_samples)

        # 打乱数据
        X_shuffled = X_data[permutation]
        y_shuffled = y_data[permutation]

        # 计算每个DataOwner应分配的样本数量
        samples_per_owner = total_samples // N
        remainder = total_samples % N  # 处理不能整除的情况

        start_idx = 0
        for i, do in enumerate(dataowners):
            # 如果有余数，将余数分配到前面的DataOwner
            extra_samples = 1 if i < remainder else 0
            end_idx = start_idx + samples_per_owner + extra_samples

            do.imgData = X_shuffled[start_idx:end_idx]  # 给每个DataOwner分配数据
            do.originalData = X_shuffled[start_idx:end_idx]
            do.labelData = y_shuffled[start_idx:end_idx]

            start_idx = end_idx

        # 输出分配情况
        for i, do in enumerate(dataowners):
            print(f"DataOwner {i + 1} holds {len(do.imgData)} samples")

    @staticmethod
    def _load_batch(batch_file):
        """
        从批处理文件加载CIFAR10数据
        :param batch_file: 批处理文件路径
        :return: (data, labels) 两个numpy数组
        """
        with open(batch_file, 'rb') as f:
            batch = pickle.load(f, encoding='bytes')
            data = batch[b'data']  # shape: (10000, 3072)
            labels = batch[b'labels']  # list of 10000 labels
            data = data.reshape(-1, 3, 32, 32).astype(np.float32) / 255.0  # 归一化到 [0, 1]
            labels = np.array(labels, dtype=np.int64)
        return data, labels

    @staticmethod
    def load_cifar10_dataset(data_dir):
        """
        加载CIFAR10数据集
        :param data_dir: CIFAR10数据集所在的目录，包含批处理文件
        :return: (train_data, train_labels, test_data, test_labels)
        """
        train_data = []
        train_labels = []
        # 加载训练数据批处理文件
        for i in range(1, 6):
            batch_file = os.path.join(data_dir, f'data_batch_{i}')
            data, labels = UtilsCIFAR10._load_batch(batch_file)
            train_data.append(data)
            train_labels.extend(labels)
        train_data = np.vstack(train_data)
        train_labels = np.array(train_labels, dtype=np.int64)

        # 加载测试数据批处理文件
        test_batch_file = os.path.join(data_dir, 'test_batch')
        test_data, test_labels = UtilsCIFAR10._load_batch(test_batch_file)

        return train_data, train_labels, test_data, test_labels

    @staticmethod
    def create_data_loader(images, labels, batch_size=64, shuffle=True):
        """
        创建DataLoader对象
        :param images: 图像数据，形状为 (num_samples, 3, 32, 32)
        :param labels: 标签数据，形状为 (num_samples,)
        :param batch_size: 每个batch的大小
        :param shuffle: 是否打乱数据
        :return: DataLoader对象
        """
        tensor_x = torch.tensor(images, dtype=torch.float32)
        tensor_y = torch.tensor(labels, dtype=torch.long)
        dataset = TensorDataset(tensor_x, tensor_y)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    @staticmethod
    def sample_arrays(arr1, arr2, proportion):
        """
        从两个长度相同的数组中随机选取相同位置的元素，形成两个新的数组。

        参数:
        arr1 (np.array): 第一个数组。
        arr2 (np.array): 第二个数组。
        proportion (float): 选取的比例，范围在0到1之间。

        返回:
        np.array: 从arr1中选取的新数组。
        np.array: 从arr2中选取的新数组。
        """
        if len(arr1) != len(arr2):
            raise ValueError("两个数组的长度必须相同")
        if not (0 <= proportion <= 1):
            print("比例必须在0到1之间，已自动调整")
            proportion = np.clip(proportion, 0, 1)

        # 计算需要选取的元素数量
        num_samples = int(len(arr1) * proportion)

        # 随机生成索引
        indices = np.random.choice(len(arr1), num_samples, replace=False)

        # 使用随机索引选取数据
        sampled_arr1 = arr1[indices]
        sampled_arr2 = arr2[indices]

        return sampled_arr1, sampled_arr2

    @staticmethod
    def add_noise(dataowner, noise_type="gaussian", severity=0.1):
        """
        给 imgData 中的每个图像添加噪声，使图像质量变差
        :param noise_type: 噪声类型，"gaussian" 或 "salt_and_pepper"
        :param severity: 噪声的严重程度（0-1）
        """
        # 原数据进行加噪处理
        noisy_data = []
        for img in dataowner.imgData:
            if noise_type == "gaussian":
                noisy_data.append(UtilsCIFAR10._add_gaussian_noise(img, severity))
            elif noise_type == "salt_and_pepper":
                noisy_data.append(UtilsCIFAR10._add_salt_and_pepper_noise(img, severity))
            else:
                raise ValueError(f"Unsupported noise type: {noise_type}")
        dataowner.imgData = noisy_data

        # 原保留数据进行归一化
        temp_data = []
        for img in dataowner.originalData:
            temp_img = np.clip(img, 0, 1)  # 保证像素值在 [0, 1] 范围内
            temp_data.append(temp_img)
        dataowner.originalData = temp_data

    @staticmethod
    def _add_gaussian_noise(img, severity):
        """
        内部方法：给单张图像添加高斯噪声
        :param img: 单张图像数据 (numpy array)，形状为 (3, 32, 32)
        :param severity: 高斯噪声的标准差比例（0-1）
        :return: 添加噪声后的图像
        """
        noise = np.random.normal(0, severity, img.shape)  # 生成高斯噪声
        noisy_img = img + noise  # 添加噪声
        noisy_img = np.clip(noisy_img, 0, 1)  # 保证像素值在 [0, 1] 范围内
        return noisy_img

    @staticmethod
    def _add_salt_and_pepper_noise(img, severity):
        """
        内部方法：给每张图像添加椒盐噪声
        :param img: 图像数据，形状为 (3, 32, 32)
        :param severity: 噪声强度（0-1），表示椒盐噪声的比例
        :return: 添加噪声后的图像
        """
        noisy_img = img.copy()
        C, H, W = noisy_img.shape
        num_salt = int(np.ceil(severity * H * W * 0.5))
        num_pepper = int(np.ceil(severity * H * W * 0.5))

        for c in range(C):
            # 添加盐噪声
            coords = [np.random.randint(0, H, num_salt), np.random.randint(0, W, num_salt)]
            noisy_img[c, coords[0], coords[1]] = 1

            # 添加椒噪声
            coords = [np.random.randint(0, H, num_pepper), np.random.randint(0, W, num_pepper)]
            noisy_img[c, coords[0], coords[1]] = 0

        return noisy_img

    @staticmethod
    def evaluate_quality(dataowner, metric="mse"):
        """
        评价 imgData 的质量
        :param metric: 评价指标类型，支持 "mse" 或 "snr"
        :return: 数据质量得分列表（每张图像的质量得分）
        """
        if len(dataowner.originalData) != len(dataowner.imgData):
            raise ValueError("originalData and imgData must have the same length.")

        quality_scores = []
        for original, noisy in zip(dataowner.originalData, dataowner.imgData):
            if metric == "mse":
                quality_scores.append(UtilsCIFAR10._calculate_mse(original, noisy))
            elif metric == "snr":
                quality_scores.append(UtilsCIFAR10._calculate_snr(original, noisy))
            else:
                raise ValueError(f"Unsupported metric: {metric}")
        return quality_scores

    @staticmethod
    def _calculate_mse(original, noisy):
        """
        计算单张图像的均方误差 (MSE)
        :param original: 原始图像，形状为 (3, 32, 32)
        :param noisy: 噪声图像，形状为 (3, 32, 32)
        :return: 均方误差
        """
        mse = np.mean((original - noisy) ** 2)
        return mse

    @staticmethod
    def _calculate_snr(original, noisy):
        """
        计算单张图像的信噪比 (SNR)
        :param original: 原始图像，形状为 (3, 32, 32)
        :param noisy: 噪声图像，形状为 (3, 32, 32)
        :return: 信噪比 (dB)
        """
        signal_power = np.mean(original ** 2)
        noise_power = np.mean((original - noisy) ** 2)
        if noise_power == 0:
            return float('inf')  # 无噪声时返回无限大
        snr = 10 * np.log10(signal_power / noise_power)
        return snr

    @staticmethod
    def dataowner_pass_data_to_cpc(dataowner, cpc, proportion):
        """
        dataowner把数据传给cpc
        :param dataowner:
        :param cpc:
        :param proportion:比例
        :return:
        """
        cpc.imgData, cpc.labelData = UtilsCIFAR10.sample_arrays(np.array(dataowner.imgData),
                                                                  np.array(dataowner.labelData), proportion)

    @staticmethod
    def print_and_log(cifar_parent_path, message):
        # 获取当前文件的绝对路径
        current_file_path = os.path.abspath(__file__)

        # 获取当前文件所在的目录
        current_dir = os.path.dirname(current_file_path)

        # 查找项目根目录（假设项目根目录包含 README.md 文件）
        def find_project_root(current_dir):
            # 向上逐层查找，直到找到项目根目录
            while not os.path.exists(os.path.join(current_dir, 'README.md')):
                current_dir = os.path.dirname(current_dir)
                # 防止在 Unix/Linux 系统中向上查找过多
                if current_dir == '/' or (os.name == 'nt' and current_dir == os.path.splitdrive(current_dir)[0] + '\\'):
                    return None
            return current_dir

        # 查找项目根目录
        project_root = find_project_root(current_dir)

        if project_root is None:
            raise FileNotFoundError("未找到项目根目录，请确保项目根目录包含 README.md 文件")

        # 构建日志文件的完整路径
        log_file_path = os.path.join(project_root, 'data', 'log', cifar_parent_path, f'{cifar_parent_path}-CIFAR10.txt')

        # 确保日志目录存在
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

        # 打开一个文件用于追加写入
        with open(log_file_path, 'a') as f:
            # 将message转换为字符串
            message_str = str(message)

            # 打印到控制台
            print(message_str)

            # 写入到文件
            f.write(message_str + '\n')

    @staticmethod
    def normalize_list(lst):
        """
        因为等于0会出问题，这里将原始的归一化结果乘以0.5，然后加上0.5，从而将结果从0到1映射到0.5到1。
        :param lst:
        :return:
        """
        if not lst:  # 如果列表为空，返回空列表
            return []

        min_val = min(lst)
        max_val = max(lst)

        if max_val == min_val:  # 如果所有元素都相同，直接返回全1列表
            return [1] * len(lst)

        normalized_lst = [0.5 + 0.5 * ((x - min_val) / (max_val - min_val)) for x in lst]
        return normalized_lst

    @staticmethod
    def compare_elements(list1, list2):
        """
        取两个list中较大的元素
        """
        # 使用列表推导式和zip函数逐个比较元素
        comparison_results = [x if x > y else y for x, y in zip(list1, list2)]
        return comparison_results

    @staticmethod
    def power_transform_then_min_max_normalize(data):
        """
        使用Yeo-Johnson幂变换使数据接近高斯分布，然后进行Min-Max归一化。
        """
        data_np = np.array(data).reshape(-1, 1)

        # method='yeo-johnson' 支持正数、负数和零，对于您的数据更通用
        # standardize=False 意味着只进行幂变换，不进行Z-score标准化
        pt = PowerTransformer(method='yeo-johnson', standardize=False)
        transformed_data = pt.fit_transform(data_np).flatten().tolist()

        # 之后再进行Min-Max归一化
        min_value = min(transformed_data)
        max_value = max(transformed_data)
        if max_value == min_value:
            return [0.0] * len(transformed_data)
        normalized_data = [(x - min_value) / (max_value - min_value) for x in transformed_data]
        return normalized_data

    @staticmethod
    def calculate_average_fn(pn_list, fn_list, xn_list):
        """
        计算加权平均fn值
        
        先将pn_list和xn_list对应项相乘形成新列表，然后与fn_list相乘形成分子，
        最后以sum(xn_list)为分母计算加权平均值
        
        :param pn_list: 概率列表，0或1的列表
        :param fn_list: 对应的fn值列表
        :param xn_list: 对应的权重列表
        :return: 加权平均fn值
        """
        if len(pn_list) != len(fn_list) or len(pn_list) != len(xn_list):
            raise ValueError("pn_list、fn_list和xn_list的长度必须相同")
            
        # 计算分母：xn_list的总和
        denominator = sum(xn_list)
        
        # 防止除零错误
        if denominator == 0:
            UtilsCIFAR10.print_and_log(parent_path, "警告: xn_list的总和为零")
            return 0
            
        # 计算分子：pn_list和xn_list先相乘，然后与fn_list相乘，最后求和
        numerator = sum(pn * fn * xn for pn, fn, xn in zip(pn_list, fn_list, xn_list))
        
        # 返回加权平均值
        return numerator / denominator
        
    @staticmethod
    def generate_random_binary_pn_list(n):
        """
        生成一个随机的二元列表
        
        列表中的每个元素有50%的概率是1，50%的概率是0
        
        :param n: 列表长度
        :return: 随机生成的0-1列表
        """
        return np.random.randint(0, 2, n).tolist()
        
    @staticmethod
    def generate_probability_based_pn_list(fn_list):
        """
        根据fn_list生成概率列表，并据此构建pn_list
        
        :param fn_list: 值列表
        :return: 根据概率生成的0-1列表
        """
        # 确保fn_list中的值都是正数，可以作为概率基础
        fn_list_np = np.array(fn_list)
        
        # 如果有负值，将所有值平移使最小值为0
        if np.min(fn_list_np) < 0:
            fn_list_np = fn_list_np - np.min(fn_list_np)
            
        # 归一化到[0,1]区间作为概率
        if np.max(fn_list_np) > 0:
            probabilities = fn_list_np / np.max(fn_list_np)
        else:
            # 如果所有值都是0，则概率都设为0.5
            probabilities = np.ones_like(fn_list_np) * 0.5
            
        # 根据概率生成二元列表
        pn_list = []
        for prob in probabilities:
            if np.random.random() < prob:
                pn_list.append(1)
            else:
                pn_list.append(0)
                
        return pn_list
        
    @staticmethod
    def generate_inverse_probability_based_pn_list(fn_list):
        """
        根据fn_list生成反向概率列表，并据此构建pn_list
        fn值越小，pn为1的概率越大
        
        :param fn_list: 值列表
        :return: 根据反向概率生成的0-1列表
        """
        # 确保fn_list中的值都是正数，可以作为概率基础
        fn_list_np = np.array(fn_list)
        
        # 如果有负值，将所有值平移使最小值为0
        if np.min(fn_list_np) < 0:
            fn_list_np = fn_list_np - np.min(fn_list_np)
            
        # 归一化到[0,1]区间
        if np.max(fn_list_np) > 0:
            normalized_values = fn_list_np / np.max(fn_list_np)
            # 反转概率：fn值越小，概率越大
            probabilities = 1 - normalized_values
        else:
            # 如果所有值都是0，则概率都设为0.5
            probabilities = np.ones_like(fn_list_np) * 0.5
            
        # 根据反向概率生成二元列表
        pn_list = []
        for prob in probabilities:
            if np.random.random() < prob:
                pn_list.append(1)
            else:
                pn_list.append(0)
                
        return pn_list
        
    @staticmethod
    def generate_fix_binary_pn_list(n):
        """
        生成一个固定概率的二元列表
        
        列表中的每个元素固定有50%的概率是1，50%的概率是0
        
        :param n: 列表长度
        :return: 使用固定0.5概率生成的0-1列表
        """
        return [0.5] * n

    @staticmethod
    def load_and_create_stratified_subset(images_path, labels_path, fraction=0.1):
        """
        加载完整的CIFAR10数据集，并从中创建一个按类别分层的子集。

        这个函数确保每个类别（0-9）都按指定的比例被抽取，从而保持数据集的类别平衡。

        :param images_path: 图像文件的路径 (如指向包含所有训练数据的目录)
        :param labels_path: 标签文件的路径 
        :param fraction: 需要从每个类别中抽取的比例，默认为0.1 (10%)
        :return: (subset_images, subset_labels) 两个打乱顺序后的numpy数组
        """
        # 加载完整数据集
        # 假设存在一个方法可以加载CIFAR10数据
        train_data, train_labels, _, _ = UtilsCIFAR10.load_cifar10_dataset(images_path)
        full_images, full_labels = train_data, train_labels

        print(f"正在从完整数据集中创建分层子集，抽样比例: {fraction * 100:.1f}%...")

        subset_images_list = []
        subset_labels_list = []

        # 1. 对每个类别（0到9）进行循环
        for class_id in range(10):
            # 找到当前类别的所有样本的索引
            indices_for_class = np.where(full_labels == class_id)[0]

            # 计算要为这个类别抽取的样本数量
            num_samples_to_select = int(len(indices_for_class) * fraction)

            # 从当前类别的索引中，随机抽取指定数量的索引（不重复）
            selected_indices = np.random.choice(indices_for_class, size=num_samples_to_select, replace=False)

            # 根据选中的索引，获取对应的图像和标签
            subset_images_list.append(full_images[selected_indices])
            subset_labels_list.append(full_labels[selected_indices])

        # 2. 将所有类别的子集合并成一个大的numpy数组
        final_subset_images = np.concatenate(subset_images_list, axis=0)
        final_subset_labels = np.concatenate(subset_labels_list, axis=0)

        # 3. 对合并后的数据集进行整体随机打乱，这对于后续训练至关重要！
        shuffled_indices = np.random.permutation(len(final_subset_labels))

        print(f"子集创建完成，总样本数: {len(final_subset_labels)}")

        return final_subset_images[shuffled_indices], final_subset_labels[shuffled_indices]
