"""
Fixed模型 - 效用实验
Fixed Model - Utility Experiment

本实验实现了Fixed（固定Eta值）模型：
- 模型拥有者（领导者）使用固定的总支付η值
- 数据拥有者通过Stackelberg博弈竞争数据贡献量
- 动态数据质量调整机制
- 数据拥有者和计算中心的匹配机制

输出：U_Eta_list（模型拥有者效用）和U_qn_list（数据拥有者平均效用）
"""

import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import re
from datetime import datetime
import os

from sklearn.preprocessing import QuantileTransformer, PowerTransformer

from src.algorithms.Stackelberg import Stackelberg
from src.algorithms.GaleShapley import GaleShapley
from src.models.CNNCIFAR100 import CIFAR100CNN, evaluate_data_for_dynamic_adjustment, fine_tune_cifar100_cnn, \
    average_models_parameters, update_model_with_parameters
from src.roles.ComputingCenter import ComputingCenter
from src.roles.DataOwner import DataOwner
from src.roles.ModelOwner import ModelOwner
from src.utils.UtilCIFAR100 import UtilsCIFAR100
from src.global_variable import parent_path, Lambda, Rho, Alpha, Epsilon, adjustment_literation


# 获取项目根目录
def get_project_root():
    # 获取当前文件的绝对路径
    current_file_path = os.path.abspath(__file__)

    # 获取当前文件所在的目录
    current_dir = os.path.dirname(current_file_path)

    # 查找项目根目录
    project_root = UtilsCIFAR100.find_project_root(current_dir)

    project_root = project_root.replace("\\", "/")

    return project_root


# 定义参数值
def define_parameters(Lambda=1, Rho=1, Alpha=1, Epsilon=1, N=5, M=5, SigmaM=[1, 1, 1, 1, 1]):
    """
    定义参数值
    :param Lambda: 市场调整因子
    :param Rho: 单位数据训练费用
    :param Alpha: 模型质量调整参数
    :param Epsilon: 训练数据质量阈值
    :param N: DataOwner的数量
    :param M: ComputingCenter数量
    :param SigmaM: ComputingCenter的计算能力
    :return:
    """

    return Lambda, Rho, Alpha, Epsilon, N, M, SigmaM


# 为联邦学习任务做准备工作
def ready_for_task(rate, N, M, SigmaM):
    project_root = get_project_root()

    # 定义CIFAR100数据集的批处理文件路径
    train_batch_file = f"{project_root}/data/dataset/CIFAR100/train"
    test_batch_file = f"{project_root}/data/dataset/CIFAR100/test"

    # 加载训练数据和测试数据
    train_data, train_labels, test_data, test_labels = UtilsCIFAR100.load_cifar100_dataset(f"{project_root}/data/dataset/CIFAR100")

    # 创建DataOwner对象数组
    dataowners = [DataOwner(Lambda=Lambda, Rho=Rho) for _ in range(N)]  # 假设有5个DataOwner

    # 切分数据
    UtilsCIFAR100.split_data_to_dataowners_with_large_gap(dataowners, train_data, train_labels)

    # 初始化ModelOwner
    modelowner = ModelOwner(model=init_model(rate=rate))

    # 初始化ComputingCenter
    ComputingCenters = [ComputingCenter(Lambda, Epsilon, SigmaM[i]) for i in range(M)]

    return dataowners, modelowner, ComputingCenters, test_data, test_labels


# modelowner的初始model
def init_model(rate):
    """
    用于初始化一个模型给modeloowner
    :param rate: 初始数据占CIFAR100的比例
    :return:
    """
    UtilsCIFAR100.print_and_log(parent_path, f"初始数据占CIFAR100的比例：{rate * 100}%")
    UtilsCIFAR100.print_and_log(parent_path, "model initing...")

    project_root = get_project_root()

    # 加载训练数据
    train_data, train_labels, _, _ = UtilsCIFAR100.load_cifar100_dataset(f"{project_root}/data/dataset/CIFAR100")

    # 获取图像数量
    num_images = train_data.shape[0]
    # 计算需要选取的图像数量
    num_samples = int(num_images * rate)
    # 随机生成索引
    indices = np.random.choice(num_images, num_samples, replace=False)
    # 使用随机索引选取数据
    train_labels = train_labels[indices]
    train_data = train_data[indices]

    train_loader = UtilsCIFAR100.create_data_loader(train_data, train_labels, batch_size=64, shuffle=True)

    # 创建CNN模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CIFAR100CNN(num_classes=100).to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 如果不存在初始化模型，就训练模型，如果存在，就加载到model中
    model_save_path = f"{project_root}/data/model/initial/cifar100_cnn_initial_model"

    if os.path.exists(model_save_path):
        print(f"{model_save_path} 存在，加载初始化模型")
        model.load_model(model_save_path)
        model.save_model(f"{project_root}/data/model/cifar100_cnn_model")
    else:
        print(f"{model_save_path} 不存在，初始化模型")
        model.train_model(train_loader, criterion, optimizer, num_epochs=5, device=str(device),
                          model_save_path=model_save_path)
        model.save_model(f"{project_root}/data/model/cifar100_cnn_model")

    return model


# 给数据集添加噪声
def dataowner_add_noise(dataowners, rate):
    """
    给数据集添加噪声
    :param dataowners:
    :param rate: 加噪（高斯噪声）的程度，初始程度在0-1之间
    :return:
    """
    # 第一次训练时：添加噪声，以1-MSE为fn
    for i, do in enumerate(dataowners):
        random_num = random.random() * rate
        UtilsCIFAR100.add_noise(do, severity=random_num)
        UtilsCIFAR100.print_and_log(parent_path, f"DataOwner{i + 1}: noise random: {random_num}")


# ModelOwner发布任务， DataOwner计算数据质量（Dataowner自己计算）
def evaluate_data_quality(dataowners):
    """
    加噪声，模拟DataOwner的数据不好的情况
    :param dataowners:
    :return:
    """
    avg_f_list = []

    # 评价数据质量
    for i, do in enumerate(dataowners):

        mse_scores = UtilsCIFAR100.evaluate_quality(do, metric="mse")
        snr_scores = UtilsCIFAR100.evaluate_quality(do, metric="snr")

        # 计算图像的质量得分
        mse_sum = 0
        for j, (mse, snr) in enumerate(zip(mse_scores, snr_scores)):
            # UtilsCIFAR100.print_and_log(parent_path,f"DataOwner{i + 1}: Image {j + 1}: MSE = {mse:.4f}, SNR = {snr:.2f} dB")
            mse_sum += mse
        avg_mse = mse_sum / len(mse_scores)
        avg_f_list.append(1 - avg_mse)

    UtilsCIFAR100.print_and_log(parent_path, "DataOwners自行评估数据质量：")
    UtilsCIFAR100.print_and_log(parent_path, f"数据质量列表avg_f_list: {avg_f_list}")
    UtilsCIFAR100.print_and_log(parent_path, f"归一化后的数据质量列表avg_f_list: {UtilsCIFAR100.normalize_list(avg_f_list)}")

    return UtilsCIFAR100.normalize_list(avg_f_list)


# ModelOwner计算模型总体支付，DataOwner确定提供的最优数据量
def calculate_optimal_payment_and_data(avg_f_list, last_xn_list, N, Rho_val, Lambda_val):
    """
    ModelOwner计算模型总体支付，DataOwner确定提供的最优数据量
    :param avg_f_list:
    :return:
    """
    # 利用Stackelberg算法，求ModelOwner的支付，DataOwner提供的最优数据量
    # 这里传入FIX值
    fix_Eta = 1
    stackelberg_solver = Stackelberg(N, Rho_val * Lambda_val, avg_f_list)
    p_star, eta_star, q_star, leader_utility, follower_utilities = stackelberg_solver.solve_with_fixed_eta(fix_Eta)

    # 将q_star转化为x_opt
    # x_opt = [a / b for a, b in zip(q_star, avg_f_list)]
    x_opt = UtilsCIFAR100.power_transform_then_min_max_normalize(q_star)

    # 将pn_list(p_star)做归一化
    p_star = UtilsCIFAR100.power_transform_then_min_max_normalize(p_star)

    return UtilsCIFAR100.compare_elements(x_opt, last_xn_list), p_star, eta_star, leader_utility, follower_utilities / N


# DataOwner结合自身数据质量来算模型贡献，分配ModelOwner的支付
def compute_contribution_rates(xn_list, avg_f_list, pn_list, best_Eta, N):
    """
    DataOwner结合自身数据质量来算模型贡献，分配ModelOwner的支付
    :param xn_list:
    :param avg_f_list:
    :param pn_list:
    :param best_Eta:
    :return:
    """
    # 计算qn（qn = xn*fn*pn）
    contributions = [a * b for a, b in zip(xn_list, avg_f_list)]
    contributions = [a * b for a, b in zip(contributions, pn_list)]

    sum_qn = sum(contributions)

    UtilsCIFAR100.print_and_log(parent_path, f"ModelOwner的最优总支付：{best_Eta}")
    for i in range(len(xn_list)):
        UtilsCIFAR100.print_and_log(parent_path, f"DataOwner{i + 1}:")
        UtilsCIFAR100.print_and_log(
            parent_path,
            f"pn:{pn_list[i]}; xn:{xn_list[i]}; 分配到的支付：{contributions[i] / sum_qn * best_Eta:.4f}")


# 匹配DataOwner和ComputingCenter
def match_data_owners_to_cpc(xn_list, ComputingCenters, dataowners, SigmaM, N, Rho_val):
    """
    匹配DataOwner和ComputingCenter
    :param xn_list:
    :param ComputingCenters:
    :return:
    """
    proposals = GaleShapley.make_proposals(SigmaM, N)
    preferences = GaleShapley.make_preferences(xn_list, ComputingCenters, Rho_val, dataowners)
    matching = GaleShapley.gale_shapley(proposals, preferences)

    UtilsCIFAR100.print_and_log(parent_path, matching)
    return matching


# DataOwner向ComputingCenter提交数据
def submit_data_to_cpc(matching, dataowners, ComputingCenters, xn_list, pn_list):
    """
    DataOwner按照xn_list中约定的比例向ComputingCenter提交数据
    :param matching:
    :param dataowners:
    :param ComputingCenters:
    :param xn_list: 需要提交的数据的比例
    :param pn_list: 选择比例
    :return:
    """
    for item in matching.items():
        # 使用正则表达式匹配字符串末尾的数字
        dataowner_match = re.search(r'\d+$', item[0])
        dataowner_index = int(dataowner_match.group()) - 1
        ComputingCenter_match = re.search(r'\d+$', item[1])
        ComputingCenter_index = int(ComputingCenter_match.group()) - 1

        UtilsCIFAR100.print_and_log(parent_path, f"DataOwner{dataowner_index + 1} 把数据交给 ComputingCenter{ComputingCenter_index + 1}")

        data_rate_list = [a * b for a, b in zip(xn_list, pn_list)]

        UtilsCIFAR100.dataowner_pass_data_to_cpc(dataowners[dataowner_index],
                                             ComputingCenters[ComputingCenter_index],
                                             data_rate_list[dataowner_index])


# 使用ComputingCenter进行模型训练和全局模型的更新
def train_model_with_cpc(matching, cpcs, test_images, test_labels, literation, avg_f_list, adjustment_literation,
                         force_update, N):
    """
    使用CPC进行模型训练和全局模型的更新
    :param matching:
    :param cpcs:
    :param test_images:
    :param test_labels:
    :param literation:训练的伦茨
    :param avg_f_list:fn的列表
    :return: 第二轮要使用的fn的列表
    """

    # 指定轮次的时候要评估数据质量, 其余轮次直接训练即可
    # FIXME 这里的调整是失效的
    if literation == adjustment_literation:
        UtilsCIFAR100.print_and_log(parent_path, "重新调整fn，进而调整xn、Eta")
        avg_f_list = [0] * N
        for item in matching.items():
            dataowner_match = re.search(r'\d+$', item[0])
            dataowner_index = int(dataowner_match.group()) - 1
            cpc_match = re.search(r'\d+$', item[1])
            cpc_index = int(cpc_match.group()) - 1

            UtilsCIFAR100.print_and_log(
                parent_path,
                f"正在评估{item[0]}的数据质量, 本轮评估的样本数据量为：{len(cpcs[cpc_index].imgData) :.2f} :")
            if len(cpcs[cpc_index].imgData) == 0:
                UtilsCIFAR100.print_and_log(parent_path, "数据量为0，跳过此轮评估")
                continue

            train_loader = UtilsCIFAR100.create_data_loader(cpcs[cpc_index].imgData, cpcs[cpc_index].labelData,
                                                        batch_size=64, shuffle=True)
            test_loader = UtilsCIFAR100.create_data_loader(test_images, test_labels, batch_size=64, shuffle=False)

            # 准备评估
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            project_root = get_project_root()

            unitDataLossDiff = evaluate_data_for_dynamic_adjustment(train_loader, test_loader, num_epochs=5,
                                                                    device=str(device), lr=1e-5,
                                                                    model_path=f"{project_root}/data/model/cifar100_cnn_model")
            avg_f_list[dataowner_index] = unitDataLossDiff

        UtilsCIFAR100.print_and_log(parent_path, "经过服务器调节后的真实数据质量：")
        UtilsCIFAR100.print_and_log(parent_path, f"数据质量列表avg_f_list: {avg_f_list}")
        UtilsCIFAR100.print_and_log(parent_path, f"归一化后的数据质量列表avg_f_list:{UtilsCIFAR100.normalize_list(avg_f_list)}")

    # 准备训练
    project_root = get_project_root()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_loader = UtilsCIFAR100.create_data_loader(test_images, test_labels, batch_size=64, shuffle=False)

    new_accuracy = fine_tune_model(cpcs, matching, test_loader, lr=1e-5, device=str(device), num_epochs=5,
                                   force_update=force_update,
                                   model_path=f"{project_root}/data/model/cifar100_cnn_model")

    return UtilsCIFAR100.normalize_list(avg_f_list), new_accuracy


# 实现联邦学习的模型训练函数
def fine_tune_model(cpcs, matching, test_loader, lr=1e-5, device='cpu', num_epochs=5, force_update=False,
                    model_path=None):
    """
    实现联邦学习的模型训练函数

    :param cpcs: ComputingCenter列表
    :param matching: 匹配结果
    :param test_loader: 测试数据加载器
    :param num_epochs: 训练轮数
    :param device: 计算设备 ('cpu' 或 'cuda')
    :param lr: 学习率
    :param model_path: 全局模型路径
    """

    # 1. 创建并加载CNN模型
    model = CIFAR100CNN(num_classes=100).to(device)
    model.load_model(model_path)

    # 2. 获取全局模型参数
    global_params = model.get_parameters()

    # 3. 客户端各自调整 - 使用传入的训练数据进行本地训练，返回训练后的参数
    updated_params_list = []

    for item in matching.items():
        cpc_match = re.search(r'\d+$', item[1])
        cpc_index = int(cpc_match.group()) - 1

        UtilsCIFAR100.print_and_log(parent_path, f"{item[1]}调整模型中, 本轮训练的数据量为：{len(cpcs[cpc_index].imgData) :.2f} :")
        if len(cpcs[cpc_index].imgData) == 0:
            UtilsCIFAR100.print_and_log(parent_path, "数据量为0，跳过此轮调整")
            continue

        train_loader = UtilsCIFAR100.create_data_loader(cpcs[cpc_index].imgData, cpcs[cpc_index].labelData, batch_size=64,
                                                    shuffle=True)

        UtilsCIFAR100.print_and_log(parent_path, "开始本地模型训练...")
        updated_params = fine_tune_cifar100_cnn(
            parameters=global_params,
            train_loader=train_loader,
            num_epochs=num_epochs,
            device=device,
            lr=lr
        )
        updated_params_list.append(updated_params)

    # 4. 上传参数 (在实际的联邦学习系统中，这一步会将参数发送到服务器)
    # 在这个简化实现中，我们直接使用更新后的参数
    UtilsCIFAR100.print_and_log(parent_path, "本地训练完成，参数已准备好进行聚合")

    # 5. 合并参数 (实际联邦学习中，服务器会收集多个客户端的参数并合并)
    avg_params = average_models_parameters(updated_params_list)
    UtilsCIFAR100.print_and_log(parent_path, "参数聚合完成")

    # 6. 选择更新 - 评估合并后的参数，如果性能更好则更新全局模型
    UtilsCIFAR100.print_and_log(parent_path, "评估聚合后的模型参数...")
    new_accuracy = update_model_with_parameters(
        model=model,
        parameters=avg_params,
        test_loader=test_loader,
        device=device,
        force_update=force_update,
        model_save_path=model_path
    )

    UtilsCIFAR100.print_and_log(parent_path, "模型更新流程完成")
    return new_accuracy


if __name__ == "__main__":
    UtilsCIFAR100.print_and_log(parent_path, f"**** {parent_path} 运行时间： {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ****")

    # 记录第 adjustment_literation+1 轮的 U(Eta) 和 U(qn)/N
    U_Eta_list = []
    U_qn_list = []

    # 从这里开始进行不同数量客户端的循环 (前闭后开)
    for n in [9, 19, 29, 39, 49, 59, 69, 79, 89, 99]:
        UtilsCIFAR100.print_and_log(parent_path, f"========================= 客户端数量: {n + 1} =========================")

        UtilsCIFAR100.print_and_log(parent_path, "---------------------------------- 定义参数值 ----------------------------------")
        Lambda_val, Rho_val, Alpha_val, Epsilon_val, N, M, SigmaM = define_parameters(Lambda=Lambda, Rho=Rho, Alpha=Alpha,
                                                                                      Epsilon=Epsilon, M=n + 1, N=n + 1,
                                                                                      SigmaM=[1] * (n + 1))
        UtilsCIFAR100.print_and_log(parent_path, "DONE")

        UtilsCIFAR100.print_and_log(parent_path, "---------------------------------- 准备工作 ----------------------------------")
        dataowners, modelowner, ComputingCenters, test_images, test_labels = ready_for_task(rate=0.001, N=N, M=M, SigmaM=SigmaM)
        UtilsCIFAR100.print_and_log(parent_path, "DONE")

        literation = 0  # 迭代次数
        adjustment_lit = adjustment_literation  # 要进行fn，xn，eta调整的轮次，注意值要取：轮次-1
        avg_f_list = []
        last_xn_list = [0] * N
        matching = None  # 初始化matching变量
        while True:
            UtilsCIFAR100.print_and_log(parent_path, f"========================= literation: {literation + 1} =========================")

            # DataOwner自己报数据质量的机会只有一次
            if literation == 0:
                UtilsCIFAR100.print_and_log(parent_path, f"----- literation {literation + 1}: 为 DataOwner 的数据添加噪声 -----")
                dataowner_add_noise(dataowners, 0.1)
                UtilsCIFAR100.print_and_log(parent_path, "DONE")

                UtilsCIFAR100.print_and_log(parent_path, f"----- literation {literation + 1}: 计算 DataOwner 的数据质量 -----")
                avg_f_list = evaluate_data_quality(dataowners)
                UtilsCIFAR100.print_and_log(parent_path, "DONE")

            UtilsCIFAR100.print_and_log(
                parent_path,
                f"----- literation {literation + 1}: 计算 ModelOwner 总体支付和 DataOwners 最优数据量 -----")
            xn_list, pn_list, best_Eta, U_Eta, U_qn = calculate_optimal_payment_and_data(avg_f_list, last_xn_list, N, Rho_val, Lambda_val)
            last_xn_list = xn_list

            # 只有在调整轮次之后的轮次才记录
            if literation == adjustment_lit + 1:
                U_Eta_list.append(U_Eta)
                U_qn_list.append(U_qn)
            UtilsCIFAR100.print_and_log(parent_path, "DONE")

            # 提前中止
            if literation > adjustment_lit:
                UtilsCIFAR100.print_and_log(parent_path, f"U_qn_list: {U_qn_list}")
                UtilsCIFAR100.print_and_log(parent_path, f"U_Eta_list: {U_Eta_list}")
                break

            UtilsCIFAR100.print_and_log(parent_path, f"----- literation {literation + 1}: DataOwner 分配 ModelOwner 的支付 -----")
            compute_contribution_rates(xn_list, avg_f_list, pn_list, best_Eta, N)
            UtilsCIFAR100.print_and_log(parent_path, "DONE")

            # 一旦匹配成功，就无法改变
            if literation == 0:
                UtilsCIFAR100.print_and_log(parent_path, f"----- literation {literation + 1}: 匹配 DataOwner 和 ComputingCenter -----")
                matching = match_data_owners_to_cpc(xn_list, ComputingCenters, dataowners, SigmaM, N, Rho_val)
                UtilsCIFAR100.print_and_log(parent_path, "DONE")

            UtilsCIFAR100.print_and_log(parent_path, f"----- literation {literation + 1}: DataOwner 向 ComputingCenter 提交数据 -----")
            submit_data_to_cpc(matching, dataowners, ComputingCenters, xn_list, pn_list)
            UtilsCIFAR100.print_and_log(parent_path, "DONE")

            UtilsCIFAR100.print_and_log(parent_path, f"----- literation {literation + 1}: 模型训练 -----")
            avg_f_list, new_accuracy = train_model_with_cpc(matching, ComputingCenters, test_images, test_labels,
                                                            literation, avg_f_list, adjustment_lit,
                                                            force_update=True, N=N)
            UtilsCIFAR100.print_and_log(parent_path, "DONE")

            literation += 1

    UtilsCIFAR100.print_and_log(parent_path, "fixed 最终的列表：")
    UtilsCIFAR100.print_and_log(parent_path, f"U_qn_list: {U_qn_list}")
    UtilsCIFAR100.print_and_log(parent_path, f"U_Eta_list: {U_Eta_list}")