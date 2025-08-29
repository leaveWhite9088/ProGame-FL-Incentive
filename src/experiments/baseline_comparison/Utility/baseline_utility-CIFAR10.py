"""
基线两方Stackelberg-Cournot模型 - 效用实验
Baseline Two-Party Stackelberg-Cournot Model - Utility Experiment

本实验实现了基于智能电网论文思想的简化模型，用于与PGI-RDFL进行效用对比：
- 模型拥有者（领导者）只决定总支付η，不决定选择概率p
- 所有数据拥有者都参与（p固定为全1）
- 数据拥有者通过古诺博弈竞争数据贡献量
- 计算中心作为被动工具，不参与博弈决策

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

from src.algorithms.CournotGame import CournotGame
from src.algorithms.GaleShapley import GaleShapley
from src.models.CNNCIFAR10 import CIFAR10CNN, evaluate_data_for_dynamic_adjustment, fine_tune_cifar10_cnn, \
    average_models_parameters, update_model_with_parameters
from src.roles.ComputingCenter import ComputingCenter
from src.roles.DataOwner import DataOwner
from src.roles.ModelOwner import ModelOwner
from src.utils.UtilCIFAR10 import UtilCIFAR10
from src.global_variable import parent_path, Lambda, Rho, Alpha, Epsilon, adjustment_literation


# 获取项目根目录
def get_project_root():
    # 获取当前文件的绝对路径
    current_file_path = os.path.abspath(__file__)

    # 获取当前文件所在的目录
    current_dir = os.path.dirname(current_file_path)

    # 查找项目根目录
    project_root = UtilCIFAR10.find_project_root(current_dir)

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
    :param SigmaM: ComputingCenter的计算呢能力
    :return:
    """

    return Lambda, Rho, Alpha, Epsilon, N, M, SigmaM


# 为联邦学习任务做准备工作
def ready_for_task(rate, N, M, SigmaM):
    project_root = get_project_root()

    # 定义CIFAR10数据集的批处理文件路径
    train_batch_files = [
        f"{project_root}/data/dataset/CIFAR10/data_batch_1",
        f"{project_root}/data/dataset/CIFAR10/data_batch_2",
        f"{project_root}/data/dataset/CIFAR10/data_batch_3",
        f"{project_root}/data/dataset/CIFAR10/data_batch_4",
        f"{project_root}/data/dataset/CIFAR10/data_batch_5"
    ]
    test_batch_file = f"{project_root}/data/dataset/CIFAR10/test_batch"

    # 加载训练数据和测试数据
    train_data, train_labels, test_data, test_labels = UtilCIFAR10.load_cifar10_dataset(f"{project_root}/data/dataset/CIFAR10")

    # 创建DataOwner对象数组
    dataowners = [DataOwner(Lambda=Lambda, Rho=Rho) for _ in range(N)]  # 假设有5个DataOwner

    # 切分数据
    UtilCIFAR10.split_data_to_dataowners_with_large_gap(dataowners, train_data, train_labels)

    # 初始化ModelOwner
    modelowner = ModelOwner(model=init_model(rate=rate))

    # 初始化ComputingCenter
    ComputingCenters = [ComputingCenter(Lambda, Epsilon, SigmaM[i]) for i in range(M)]

    return dataowners, modelowner, ComputingCenters, test_data, test_labels


# modelowner的初始model
def init_model(rate):
    """
    用于初始化一个模型给modeloowner
    :param rate: 初始数据占CIFAR10的比例
    :return:
    """
    UtilCIFAR10.print_and_log(parent_path, f"初始数据占CIFAR10的比例：{rate * 100}%")
    UtilCIFAR10.print_and_log(parent_path, "model initing...")

    project_root = get_project_root()

    # 加载训练数据
    train_data, train_labels, _, _ = UtilCIFAR10.load_cifar10_dataset(f"{project_root}/data/dataset/CIFAR10")

    # 获取图像数量
    num_images = train_data.shape[0]
    # 计算需要选取的图像数量
    num_samples = int(num_images * rate)
    # 随机生成索引
    indices = np.random.choice(num_images, num_samples, replace=False)
    # 使用随机索引选取数据
    train_labels = train_labels[indices]
    train_data = train_data[indices]

    train_loader = UtilCIFAR10.create_data_loader(train_data, train_labels, batch_size=64, shuffle=True)

    # 创建CNN模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CIFAR10CNN(num_classes=10).to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 如果不存在初始化模型，就训练模型，如果存在，就加载到model中
    model_save_path = f"{project_root}/data/model/initial/cifar10_cnn_initial_model"

    if os.path.exists(model_save_path):
        print(f"{model_save_path} 存在，加载初始化模型")
        model.load_model(model_save_path)
        model.save_model(f"{project_root}/data/model/cifar10_cnn_model")
    else:
        print(f"{model_save_path} 不存在，初始化模型")
        model.train_model(train_loader, criterion, optimizer, num_epochs=5, device=str(device),
                          model_save_path=model_save_path)
        model.save_model(f"{project_root}/data/model/cifar10_cnn_model")

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
        UtilCIFAR10.add_noise(do, severity=random_num)
        UtilCIFAR10.print_and_log(parent_path, f"DataOwner{i + 1}: noise random: {random_num}")


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

        mse_scores = UtilCIFAR10.evaluate_quality(do, metric="mse")
        snr_scores = UtilCIFAR10.evaluate_quality(do, metric="snr")

        # 计算图像的质量得分
        mse_sum = 0
        for j, (mse, snr) in enumerate(zip(mse_scores, snr_scores)):
            # UtilCIFAR10.print_and_log(parent_path,f"DataOwner{i + 1}: Image {j + 1}: MSE = {mse:.4f}, SNR = {snr:.2f} dB")
            mse_sum += mse
        avg_mse = mse_sum / len(mse_scores)
        avg_f_list.append(1 - avg_mse)

    UtilCIFAR10.print_and_log(parent_path, "DataOwners自行评估数据质量：")
    UtilCIFAR10.print_and_log(parent_path, f"数据质量列表avg_f_list: {avg_f_list}")
    UtilCIFAR10.print_and_log(parent_path, f"归一化后的数据质量列表avg_f_list: {UtilCIFAR10.normalize_list(avg_f_list)}")

    return UtilCIFAR10.normalize_list(avg_f_list)


# 基线模型的核心函数：计算支付和数据量
def calculate_baseline_payment_and_data(avg_f_list, last_xn_list, N, Rho_val, Lambda_val):
    """
    使用基线Stackelberg-Cournot模型计算支付和数据量
    
    核心思想：映射智能电网论文的两方博弈到联邦学习三方场景
    - 固定p为全1（所有数据拥有者都参与，无选择机制）
    - 领导者设定一个固定的总支付（不进行优化）
    - 跟随者通过古诺博弈竞争数据贡献量
    
    :param avg_f_list: 数据质量列表
    :param last_xn_list: 上一轮的数据量列表
    :param N: 数据拥有者数量
    :param Rho_val: 单位数据训练费用
    :param Lambda_val: 市场调整因子
    :return: (xn_list, pn_list, best_Eta, U_Eta, U_qn)
    """
    # 基线模型核心特征：
    # 1. p固定为全1（所有人都参与，无选择机制）
    p_star = np.ones(N)
    p_star_normalized = p_star / np.sum(p_star)  # 归一化
    
    # 2. 领导者设定一个合理的固定总支付（基于成本和预期收益）
    # 根据智能电网论文的思想，总支付应该能够激励参与但不过度支付
    # 这里使用一个简单的启发式：基于平均质量和单位成本
    avg_quality = np.mean(avg_f_list)
    total_expected_contribution = N * avg_quality  # 预期总贡献
    unit_cost = Rho_val * Lambda_val
    
    # 总支付 = 预期总贡献 * 单位成本 * 调节因子
    # 调节因子设为0.8，确保有一定利润空间
    eta_star = total_expected_contribution * unit_cost * 0.8
    
    # 3. 给定固定的p和η，数据拥有者进行古诺博弈
    # 使用CournotGame直接计算均衡
    from src.algorithms.CournotGame import CournotGame
    cournot_solver = CournotGame(N, unit_cost, avg_f_list, Lambda_val)
    q_star = cournot_solver.compute_equilibrium(p_star, eta_star)
    
    # 4. 计算效用
    # 领导者效用：总质量贡献 - 总支付
    total_quality = np.sum(p_star * q_star)
    leader_utility = total_quality - eta_star
    
    # 跟随者效用
    if total_quality > 1e-9:
        price_per_quality = eta_star / total_quality
        follower_utilities = q_star * (price_per_quality - unit_cost)
    else:
        follower_utilities = np.zeros(N)
    
    # 将q_star转化为x_opt
    x_opt = [a / b if b > 0 else 0 for a, b in zip(q_star, avg_f_list)]
    
    # 比较新旧数据量
    xn_list = UtilCIFAR10.compare_elements(x_opt, last_xn_list)
    
    # 平均跟随者效用
    avg_follower_utility = np.mean(follower_utilities)
    
    UtilCIFAR10.print_and_log(parent_path, f"[基线模型] 固定总支付η: {eta_star:.4f}")
    UtilCIFAR10.print_and_log(parent_path, f"[基线模型] 领导者效用: {leader_utility:.4f}")
    UtilCIFAR10.print_and_log(parent_path, f"[基线模型] 跟随者平均效用: {avg_follower_utility:.4f}")
    
    return xn_list, p_star_normalized, eta_star, leader_utility, avg_follower_utility


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

    UtilCIFAR10.print_and_log(parent_path, f"ModelOwner的最优总支付：{best_Eta}")
    for i in range(len(xn_list)):
        UtilCIFAR10.print_and_log(parent_path, f"DataOwner{i + 1}:")
        UtilCIFAR10.print_and_log(parent_path,
            f"pn:{pn_list[i]}; xn:{xn_list[i]}; 分配到的支付：{contributions[i] / sum_qn * best_Eta:.4f}")


# 匹配DataOwner和ComputingCenter
def match_data_owners_to_cpc(xn_list, ComputingCenters, dataowners, SigmaM, N, Rho):
    """
    匹配DataOwner和ComputingCenter
    :param xn_list:
    :param ComputingCenters:
    :return:
    """
    proposals = GaleShapley.make_proposals(SigmaM, N)
    preferences = GaleShapley.make_preferences(xn_list, ComputingCenters, Rho, dataowners)
    matching = GaleShapley.gale_shapley(proposals, preferences)

    UtilCIFAR10.print_and_log(parent_path, matching)
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

        UtilCIFAR10.print_and_log(parent_path, f"DataOwner{dataowner_index + 1} 把数据交给 ComputingCenter{ComputingCenter_index + 1}")

        data_rate_list = [a * b for a, b in zip(xn_list, pn_list)]

        UtilCIFAR10.dataowner_pass_data_to_cpc(dataowners[dataowner_index],
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
        UtilCIFAR10.print_and_log(parent_path, "重新调整fn，进而调整xn、Eta")
        avg_f_list = [0] * N
        for item in matching.items():
            dataowner_match = re.search(r'\d+$', item[0])
            dataowner_index = int(dataowner_match.group()) - 1
            cpc_match = re.search(r'\d+$', item[1])
            cpc_index = int(cpc_match.group()) - 1

            UtilCIFAR10.print_and_log(parent_path,
                f"正在评估{item[0]}的数据质量, 本轮评估的样本数据量为：{len(cpcs[cpc_index].imgData) :.2f} :")
            if len(cpcs[cpc_index].imgData) == 0:
                UtilCIFAR10.print_and_log(parent_path, "数据量为0，跳过此轮评估")
                continue

            train_loader = UtilCIFAR10.create_data_loader(cpcs[cpc_index].imgData, cpcs[cpc_index].labelData,
                                                        batch_size=64, shuffle=True)
            test_loader = UtilCIFAR10.create_data_loader(test_images, test_labels, batch_size=64, shuffle=False)

            # 准备评估
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            project_root = get_project_root()

            unitDataLossDiff = evaluate_data_for_dynamic_adjustment(train_loader, test_loader, num_epochs=5,
                                                                    device=str(device), lr=1e-5,
                                                                    model_path=f"{project_root}/data/model/cifar10_cnn_model")
            avg_f_list[dataowner_index] = unitDataLossDiff

        UtilCIFAR10.print_and_log(parent_path, "经过服务器调节后的真实数据质量：")
        UtilCIFAR10.print_and_log(parent_path, f"数据质量列表avg_f_list: {avg_f_list}")
        UtilCIFAR10.print_and_log(parent_path, f"归一化后的数据质量列表avg_f_list:{UtilCIFAR10.normalize_list(avg_f_list)}")

    # 准备训练
    project_root = get_project_root()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_loader = UtilCIFAR10.create_data_loader(test_images, test_labels, batch_size=64, shuffle=False)

    new_accuracy = fine_tune_model(cpcs, matching, test_loader, lr=1e-5, device=str(device), num_epochs=5,
                                   force_update=force_update,
                                   model_path=f"{project_root}/data/model/cifar10_cnn_model")

    return UtilCIFAR10.normalize_list(avg_f_list), new_accuracy


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
    model = CIFAR10CNN(num_classes=10).to(device)
    model.load_model(model_path)

    # 2. 获取全局模型参数
    global_params = model.get_parameters()

    # 3. 客户端各自调整 - 使用传入的训练数据进行本地训练，返回训练后的参数
    updated_params_list = []

    for item in matching.items():
        cpc_match = re.search(r'\d+$', item[1])
        cpc_index = int(cpc_match.group()) - 1

        UtilCIFAR10.print_and_log(parent_path, f"{item[1]}调整模型中, 本轮训练的数据量为：{len(cpcs[cpc_index].imgData) :.2f} :")
        if len(cpcs[cpc_index].imgData) == 0:
            UtilCIFAR10.print_and_log(parent_path, "数据量为0，跳过此轮调整")
            continue

        train_loader = UtilCIFAR10.create_data_loader(cpcs[cpc_index].imgData, cpcs[cpc_index].labelData, batch_size=64,
                                                    shuffle=True)

        UtilCIFAR10.print_and_log(parent_path, "开始本地模型训练...")
        updated_params = fine_tune_cifar10_cnn(
            parameters=global_params,
            train_loader=train_loader,
            num_epochs=num_epochs,
            device=device,
            lr=lr
        )
        updated_params_list.append(updated_params)

    # 4. 上传参数 (在实际的联邦学习系统中，这一步会将参数发送到服务器)
    # 在这个简化实现中，我们直接使用更新后的参数
    UtilCIFAR10.print_and_log(parent_path, "本地训练完成，参数已准备好进行聚合")

    # 5. 合并参数 (实际联邦学习中，服务器会收集多个客户端的参数并合并)
    avg_params = average_models_parameters(updated_params_list)
    UtilCIFAR10.print_and_log(parent_path, "参数聚合完成")

    # 6. 选择更新 - 评估合并后的参数，如果性能更好则更新全局模型
    UtilCIFAR10.print_and_log(parent_path, "评估聚合后的模型参数...")
    new_accuracy = update_model_with_parameters(
        model=model,
        parameters=avg_params,
        test_loader=test_loader,
        device=device,
        force_update=force_update,
        model_save_path=model_path
    )

    UtilCIFAR10.print_and_log(parent_path, "模型更新流程完成")
    return new_accuracy


if __name__ == "__main__":
    UtilCIFAR10.print_and_log(parent_path, f"**** Baseline Two-Party Stackelberg-Cournot - Utility Experiment ****")
    UtilCIFAR10.print_and_log(parent_path, f"**** 基线两方斯塔克尔伯格-古诺模型 - 效用实验 ****")
    UtilCIFAR10.print_and_log(parent_path, f"**** 运行时间： {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ****")
    
    # 记录第 adjustment_literation+1 轮的效用值
    U_Eta_list = []  # 模型拥有者效用
    U_qn_list = []   # 数据拥有者平均效用
    
    # 客户端数量设置
    for n in [9, 19, 29, 39, 49, 59, 69, 79, 89, 99]:
        UtilCIFAR10.print_and_log(parent_path, f"========================= 客户端数量: {n + 1} =========================")
        
        # 定义参数
        UtilCIFAR10.print_and_log(parent_path, "---------------------------------- 定义参数值 ----------------------------------")
        Lambda_val, Rho_val, Alpha_val, Epsilon_val, N, M, SigmaM = define_parameters(
            Lambda=Lambda, Rho=Rho, Alpha=Alpha, Epsilon=Epsilon, M=n + 1, N=n + 1, SigmaM=[1] * (n + 1)
        )
        UtilCIFAR10.print_and_log(parent_path, "DONE")
        
        # 准备工作
        UtilCIFAR10.print_and_log(parent_path, "---------------------------------- 准备工作 ----------------------------------")
        dataowners, modelowner, ComputingCenters, test_images, test_labels = ready_for_task(rate=0.001, N=N, M=M, SigmaM=SigmaM)
        UtilCIFAR10.print_and_log(parent_path, "DONE")
        
        literation = 0
        adjustment_lit = adjustment_literation
        avg_f_list = []
        last_xn_list = [0] * N
        matching = None  # 初始化matching变量
        
        while True:
            UtilCIFAR10.print_and_log(parent_path, f"========================= literation: {literation + 1} =========================")
            
            # 第一轮：添加噪声和评估数据质量
            if literation == 0:
                UtilCIFAR10.print_and_log(parent_path, f"----- literation {literation + 1}: 为 DataOwner 的数据添加噪声 -----")
                dataowner_add_noise(dataowners, 0.1)
                UtilCIFAR10.print_and_log(parent_path, "DONE")
                
                UtilCIFAR10.print_and_log(parent_path, f"----- literation {literation + 1}: 计算 DataOwner 的数据质量 -----")
                avg_f_list = evaluate_data_quality(dataowners)
                UtilCIFAR10.print_and_log(parent_path, "DONE")
            
            # 使用基线模型计算支付和数据量
            UtilCIFAR10.print_and_log(parent_path, f"----- literation {literation + 1}: [基线模型] 计算 ModelOwner 总体支付和 DataOwners 最优数据量 -----")
            UtilCIFAR10.print_and_log(parent_path, "[基线特征] p固定为全1，只优化η")
            
            xn_list, pn_list, best_Eta, U_Eta, U_qn = calculate_baseline_payment_and_data(
                avg_f_list, last_xn_list, N, Rho_val, Lambda_val
            )
            last_xn_list = xn_list
            
            # 记录调整后的效用
            if literation == adjustment_lit + 1:
                U_Eta_list.append(U_Eta)
                U_qn_list.append(U_qn)
                UtilCIFAR10.print_and_log(parent_path, f"[记录效用] U_Eta: {U_Eta:.4f}, U_qn: {U_qn:.4f}")
            
            UtilCIFAR10.print_and_log(parent_path, "DONE")
            
            # 提前中止（只关注效用值）
            if literation > adjustment_lit:
                UtilCIFAR10.print_and_log(parent_path, f"完成效用实验，准备输出结果...")
                break
            
            # 分配支付
            UtilCIFAR10.print_and_log(parent_path, f"----- literation {literation + 1}: DataOwner 分配 ModelOwner 的支付 -----")
            compute_contribution_rates(xn_list, avg_f_list, pn_list, best_Eta, N)
            UtilCIFAR10.print_and_log(parent_path, "DONE")
            
            # 匹配（只在第一轮）
            if literation == 0:
                UtilCIFAR10.print_and_log(parent_path, f"----- literation {literation + 1}: 匹配 DataOwner 和 ComputingCenter -----")
                matching = match_data_owners_to_cpc(xn_list, ComputingCenters, dataowners, SigmaM, N, Rho_val)
                UtilCIFAR10.print_and_log(parent_path, "DONE")
            
            # 提交数据
            UtilCIFAR10.print_and_log(parent_path, f"----- literation {literation + 1}: DataOwner 向 ComputingCenter 提交数据 -----")
            submit_data_to_cpc(matching, dataowners, ComputingCenters, xn_list, pn_list)
            UtilCIFAR10.print_and_log(parent_path, "DONE")
            
            # 模型训练（为了更新数据质量）
            UtilCIFAR10.print_and_log(parent_path, f"----- literation {literation + 1}: 模型训练 -----")
            avg_f_list, _ = train_model_with_cpc(
                matching, ComputingCenters, test_images, test_labels,
                literation, avg_f_list, adjustment_lit, force_update=True, N=N
            )
            UtilCIFAR10.print_and_log(parent_path, "DONE")
            
            literation += 1
    
    # 输出最终结果
    UtilCIFAR10.print_and_log(parent_path, "\n===== 基线效用实验最终结果 =====")
    UtilCIFAR10.print_and_log(parent_path, f"模型拥有者效用 U_Eta_list: {U_Eta_list}")
    UtilCIFAR10.print_and_log(parent_path, f"数据拥有者平均效用 U_qn_list: {U_qn_list}")
    
    # 保存结果到文件
    result_path = os.path.join(get_project_root(), "data/log/baseline_comparison/")
    os.makedirs(result_path, exist_ok=True)
    
    with open(os.path.join(result_path, "baseline_utility_results.txt"), "w") as f:
        f.write(f"Baseline Two-Party Stackelberg-Cournot - Utility Results\n")
        f.write(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"U_Eta_list: {U_Eta_list}\n")
        f.write(f"U_qn_list: {U_qn_list}\n")