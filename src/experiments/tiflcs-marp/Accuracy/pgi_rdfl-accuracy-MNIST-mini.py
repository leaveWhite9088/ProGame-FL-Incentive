"""
【Mini版 - 快速测试】
PGI-RDFL (原始动态算法) - 精度实验
- 使用 UtilMNIST.load_and_create_stratified_subset 加载 10% 的数据以快速测试
- 保留了动态 Eta 和动态 fn 调整的原始逻辑
- 迭代次数被缩短以便测试
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
from src.models.CNNMNIST import MNISTCNN, evaluate_data_for_dynamic_adjustment, fine_tune_mnist_cnn, \
    average_models_parameters, update_model_with_parameters
from src.roles.ComputingCenter import ComputingCenter
from src.roles.DataOwner import DataOwner
from src.roles.ModelOwner import ModelOwner
from src.utils.UtilMNIST import UtilMNIST
# 我们将在 __main__ 中覆盖 adjustment_literation，所以这里的导入值不重要
from src.global_variable import parent_path, Lambda, Rho, Alpha, Epsilon, adjustment_literation


# 获取项目根目录
def get_project_root():
    # 获取当前文件的绝对路径
    current_file_path = os.path.abspath(__file__)

    # 获取当前文件所在的目录
    current_dir = os.path.dirname(current_file_path)

    # 查找项目根目录
    project_root = UtilMNIST.find_project_root(current_dir)

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


# 【修改后】: 为联邦学习任务做准备工作 (Mini版)
def ready_for_task(rate, N, M, SigmaM):
    project_root = get_project_root()

    train_images_path = f"{project_root}/data/dataset/MNIST/train-images.idx3-ubyte"
    train_labels_path = f"{project_root}/data/dataset/MNIST/train-labels.idx1-ubyte"
    test_images_path = f"{project_root}/data/dataset/MNIST/t10k-images.idx3-ubyte"
    test_labels_path = f"{project_root}/data/dataset/MNIST/t10k-labels.idx1-ubyte"

    # 【新】: 设置数据子集比例
    DATA_FRACTION = 0.1  # (10% 的数据用于快速测试)
    UtilMNIST.print_and_log(f"**** [MINI TEST] 使用 {DATA_FRACTION * 100:.0f}% 的分层子集数据 ****")

    # 【修改后】: 加载训练数据和测试数据的分层子集
    train_images, train_labels = UtilMNIST.load_and_create_stratified_subset(train_images_path, train_labels_path,
                                                                             fraction=DATA_FRACTION)
    test_images, test_labels = UtilMNIST.load_and_create_stratified_subset(test_images_path, test_labels_path,
                                                                           fraction=DATA_FRACTION)

    # 创建DataOwner对象数组
    dataowners = [DataOwner(Lambda=Lambda, Rho=Rho) for _ in range(N)]  # 假设有N个DataOwner

    # 切分数据
    UtilMNIST.split_data_to_dataowners_with_large_gap(dataowners, train_images, train_labels)

    # 初始化ModelOwner
    modelowner = ModelOwner(model=init_model(rate=rate))

    # 初始化ComputingCenter
    ComputingCenters = [ComputingCenter(Lambda, Epsilon, SigmaM[i]) for i in range(M)]

    return dataowners, modelowner, ComputingCenters, test_images, test_labels


# 【修改后】: modelowner的初始model (Mini版)
def init_model(rate):
    """
    用于初始化一个模型给modeloowner
    :param rate: 初始数据占MNIST的比例 (这里将作为分层抽样的比例)
    :return:
    """
    UtilMNIST.print_and_log(f"初始数据占MNIST的比例：{rate * 100}% (分层抽样)")
    UtilMNIST.print_and_log("model initing...")

    project_root = get_project_root()

    train_images_path = f"{project_root}/data/dataset/MNIST/train-images.idx3-ubyte"
    train_labels_path = f"{project_root}/data/dataset/MNIST/train-labels.idx1-ubyte"

    # 【修改后】: 直接调用新函数加载分层子集，rate参数决定了抽样比例
    train_images, train_labels = UtilMNIST.load_and_create_stratified_subset(train_images_path, train_labels_path,
                                                                             fraction=rate)

    train_loader = UtilMNIST.create_data_loader(train_images, train_labels, batch_size=64, shuffle=True)

    # 创建CNN模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MNISTCNN(num_classes=10).to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 【修改后】: 建议根据比例保存不同模型，避免冲突
    model_save_path = f"{project_root}/data/model/initial/mnist_cnn_initial_model_fraction_{rate}"

    if os.path.exists(model_save_path):
        print(f"{model_save_path} 存在，加载初始化模型")
        model.load_model(model_save_path)
        model.save_model(f"{project_root}/data/model/mnist_cnn_model")
    else:
        print(f"{model_save_path} 不存在，初始化模型")
        model.train_model(train_loader, criterion, optimizer, num_epochs=5, device=str(device),
                          model_save_path=model_save_path)
        model.save_model(f"{project_root}/data/model/mnist_cnn_model")

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
        UtilMNIST.add_noise(do, severity=random_num)
        UtilMNIST.print_and_log(f"DataOwner{i + 1}: noise random: {random_num}")


# 【修改后】: ModelOwner发布任务， DataOwner计算数据质量 (Mini版, 内部实现)
def evaluate_data_quality(dataowners, N):
    """
    加噪声，模拟DataOwner的数据不好的情况
    :param dataowners:
    :param N: 客户端总数 (尽管在此函数中未使用，但保持签名一致性)
    :return:
    """
    avg_f_list = []  # 【修改】: 在函数内部初始化

    # 评价数据质量
    for i, do in enumerate(dataowners):

        mse_scores = UtilMNIST.evaluate_quality(do, metric="mse")
        snr_scores = UtilMNIST.evaluate_quality(do, metric="snr")

        # 计算图像的质量得分
        mse_sum = 0

        # 【健壮性】: 检查是否有评分，防止除以0
        if not mse_scores:
            UtilMNIST.print_and_log(f"DataOwner{i + 1} 没有数据，跳过质量评估。")
            avg_f_list.append(0)  # 质量为0
            continue

        for j, (mse, snr) in enumerate(zip(mse_scores, snr_scores)):
            # UtilMNIST.print_and_log(parent_path,f"DataOwner{i + 1}: Image {j + 1}: MSE = {mse:.4f}, SNR = {snr:.2f} dB")
            mse_sum += mse
        avg_mse = mse_sum / len(mse_scores)
        avg_f_list.append(1 - avg_mse)

    UtilMNIST.print_and_log("DataOwners自行评估数据质量：")
    UtilMNIST.print_and_log(f"数据质量列表avg_f_list: {avg_f_list}")
    UtilMNIST.print_and_log(f"归一化后的数据质量列表avg_f_list: {UtilMNIST.normalize_list(avg_f_list)}")

    return UtilMNIST.normalize_list(avg_f_list)


# 【修改后】: ModelOwner计算模型总体支付，DataOwner确定提供的最优数据量 (添加参数)
def calculate_optimal_payment_and_data(avg_f_list, last_xn_list, N, Rho_val, Lambda_val):
    """
    ModelOwner计算模型总体支付，DataOwner确定提供的最优数据量
    :param avg_f_list:
    :param last_xn_list:
    :param N:
    :param Rho_val:
    :param Lambda_val:
    :return:
    """
    # 利用Stackelberg算法，求ModelOwner的支付，DataOwner提供的最优数据量
    stackelberg_solver = Stackelberg(N, Rho_val * Lambda_val, avg_f_list)

    p_star, eta_star, q_star, leader_utility, follower_utilities = stackelberg_solver.solve()

    # 将q_star转化为x_opt
    # x_opt = [a / b for a, b in zip(q_star, avg_f_list)]
    x_opt = UtilMNIST.power_transform_then_min_max_normalize(q_star)

    # 将pn_list(p_star)做归一化
    p_star = UtilMNIST.power_transform_then_min_max_normalize(p_star)

    return UtilMNIST.compare_elements(x_opt, last_xn_list), p_star, eta_star, leader_utility, follower_utilities / N


# 【修改后】: DataOwner结合自身数据质量来算模型贡献，分配ModelOwner的支付 (添加参数和检查)
def compute_contribution_rates(xn_list, avg_f_list, pn_list, best_Eta, N):
    """
    DataOwner结合自身数据质量来算模型贡献，分配ModelOwner的支付
    :param xn_list:
    :param avg_f_list:
    :param pn_list:
    :param best_Eta:
    :param N:
    :return:
    """
    # 计算qn（qn = xn*fn*pn）
    contributions = [a * b for a, b in zip(xn_list, avg_f_list)]
    contributions = [a * b for a, b in zip(contributions, pn_list)]

    sum_qn = sum(contributions)

    UtilMNIST.print_and_log(f"ModelOwner的最优总支付：{best_Eta}")

    if sum_qn == 0:
        UtilMNIST.print_and_log("总贡献为0，没有支付分配。")
        for i in range(N):
            UtilMNIST.print_and_log(f"DataOwner{i + 1}: pn:{pn_list[i]}; xn:{xn_list[i]}")
        return

    for i in range(len(xn_list)):
        UtilMNIST.print_and_log(f"DataOwner{i + 1}:")
        UtilMNIST.print_and_log(
            f"pn:{pn_list[i]}; xn:{xn_list[i]}; 分配到的支付：{contributions[i] / sum_qn * best_Eta:.4f}")


# 【修改后】: 匹配DataOwner和ComputingCenter (添加参数)
def match_data_owners_to_cpc(xn_list, ComputingCenters, dataowners, SigmaM, N, Rho_val):
    """
    匹配DataOwner和ComputingCenter
    :param xn_list:
    :param ComputingCenters:
    :param dataowners:
    :param SigmaM:
    :param N:
    :param Rho_val:
    :return:
    """
    proposals = GaleShapley.make_proposals(SigmaM, N)
    preferences = GaleShapley.make_preferences(xn_list, ComputingCenters, Rho_val, dataowners)
    matching = GaleShapley.gale_shapley(proposals, preferences)

    UtilMNIST.print_and_log(matching)
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

        UtilMNIST.print_and_log(f"DataOwner{dataowner_index + 1} 把数据交给 ComputingCenter{ComputingCenter_index + 1}")

        data_rate_list = [a * b for a, b in zip(xn_list, pn_list)]

        UtilMNIST.dataowner_pass_data_to_cpc(dataowners[dataowner_index],
                                             ComputingCenters[ComputingCenter_index],
                                             data_rate_list[dataowner_index])


# 【修改后】: 使用ComputingCenter进行模型训练和全局模型的更新 (添加N参数)
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
    :param adjustment_literation:
    :param force_update:
    :param N:
    :return: 第二轮要使用的fn的列表
    """

    # 指定轮次的时候要评估数据质量, 其余轮次直接训练即可
    # FIXME 这里的调整是失效的
    if literation == adjustment_literation:
        UtilMNIST.print_and_log("重新调整fn，进而调整xn、Eta")
        avg_f_list = [0] * N
        for item in matching.items():
            dataowner_match = re.search(r'\d+$', item[0])
            dataowner_index = int(dataowner_match.group()) - 1
            cpc_match = re.search(r'\d+$', item[1])
            cpc_index = int(cpc_match.group()) - 1

            UtilMNIST.print_and_log(
                f"正在评估{item[0]}的数据质量, 本轮评估的样本数据量为：{len(cpcs[cpc_index].imgData) :.2f} :")
            if len(cpcs[cpc_index].imgData) == 0:
                UtilMNIST.print_and_log("数据量为0，跳过此轮评估")
                continue

            train_loader = UtilMNIST.create_data_loader(cpcs[cpc_index].imgData, cpcs[cpc_index].labelData,
                                                        batch_size=64, shuffle=True)
            test_loader = UtilMNIST.create_data_loader(test_images, test_labels, batch_size=64, shuffle=False)

            # 准备评估
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            project_root = get_project_root()

            unitDataLossDiff = evaluate_data_for_dynamic_adjustment(train_loader, test_loader, num_epochs=5,
                                                                    device=str(device), lr=1e-5,
                                                                    model_path=f"{project_root}/data/model/mnist_cnn_model")
            avg_f_list[dataowner_index] = unitDataLossDiff

        UtilMNIST.print_and_log("经过服务器调节后的真实数据质量：")
        UtilMNIST.print_and_log(f"数据质量列表avg_f_list: {avg_f_list}")
        UtilMNIST.print_and_log(f"归一化后的数据质量列表avg_f_list:{UtilMNIST.normalize_list(avg_f_list)}")

    # 准备训练
    project_root = get_project_root()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_loader = UtilMNIST.create_data_loader(test_images, test_labels, batch_size=64, shuffle=False)

    new_accuracy = fine_tune_model(cpcs, matching, test_loader, lr=1e-5, device=str(device), num_epochs=5,
                                   force_update=force_update,
                                   model_path=f"{project_root}/data/model/mnist_cnn_model")

    return UtilMNIST.normalize_list(avg_f_list), new_accuracy


# 【修改后】: 实现联邦学习的模型训练函数 (添加健壮性检查)
def fine_tune_model(cpcs, matching, test_loader, lr=1e-5, device='cpu', num_epochs=5, force_update=False,
                    model_path=None):
    """
    实现联邦学习的模型训练函数
    """

    # 1. 创建并加载CNN模型
    model = MNISTCNN(num_classes=10).to(device)
    model.load_model(model_path)

    # 2. 获取全局模型参数
    global_params = model.get_parameters()

    # 3. 客户端各自调整 - 使用传入的训练数据进行本地训练，返回训练后的参数
    updated_params_list = []

    for item in matching.items():
        cpc_match = re.search(r'\d+$', item[1])
        cpc_index = int(cpc_match.group()) - 1

        UtilMNIST.print_and_log(f"{item[1]}调整模型中, 本轮训练的数据量为：{len(cpcs[cpc_index].imgData) :.2f} :")
        if len(cpcs[cpc_index].imgData) == 0:
            UtilMNIST.print_and_log("数据量为0，跳过此轮调整")
            continue

        train_loader = UtilMNIST.create_data_loader(cpcs[cpc_index].imgData, cpcs[cpc_index].labelData, batch_size=64,
                                                    shuffle=True)

        UtilMNIST.print_and_log("开始本地模型训练...")
        updated_params = fine_tune_mnist_cnn(
            parameters=global_params,
            train_loader=train_loader,
            num_epochs=num_epochs,
            device=device,
            lr=lr
        )
        updated_params_list.append(updated_params)

    # 4. 上传参数
    UtilMNIST.print_and_log("本地训练完成，参数已准备好进行聚合")

    # 5. 合并参数
    if not updated_params_list:
        UtilMNIST.print_and_log("没有客户端提交模型参数，跳过聚合和更新。")
        # 返回上一轮的精度
        # (需要重新创建 test_loader，因为它可能在其他地方被消耗)
        # test_loader = UtilMNIST.create_data_loader(test_images, test_labels, batch_size=64, shuffle=False)
        current_accuracy = model.evaluate_model(test_loader, device=device)
        return current_accuracy

    avg_params = average_models_parameters(updated_params_list)
    UtilMNIST.print_and_log("参数聚合完成")

    # 6. 选择更新 - 评估合并后的参数，如果性能更好则更新全局模型
    UtilMNIST.print_and_log("评估聚合后的模型参数...")
    new_accuracy = update_model_with_parameters(
        model=model,
        parameters=avg_params,
        test_loader=test_loader,
        device=device,
        force_update=force_update,
        model_save_path=model_path
    )

    UtilMNIST.print_and_log("模型更新流程完成")
    return new_accuracy


if __name__ == "__main__":
    UtilMNIST.print_and_log(
        f"**** {parent_path} (PGI-RDFL Original - MINI TEST) 运行时间： {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ****")

    # 记录精确度
    accuracy_list_total = []

    # 【Mini版】: 为了快速测试，只跑一个N=9的
    for n in [9]:
        UtilMNIST.print_and_log(f"========================= MINI: 客户端数量: {n + 1} =========================")

        UtilMNIST.print_and_log("---------------------------------- 定义参数值 ----------------------------------")
        Lambda_val, Rho_val, Alpha_val, Epsilon_val, N, M, SigmaM = define_parameters(
            Lambda=Lambda, Rho=Rho, Alpha=Alpha, Epsilon=Epsilon, M=n + 1, N=n + 1, SigmaM=[1] * (n + 1)
        )
        UtilMNIST.print_and_log("DONE")

        UtilMNIST.print_and_log("---------------------------------- 准备工作 ----------------------------------")
        dataowners, modelowner, ComputingCenters, test_images, test_labels = ready_for_task(
            rate=0.001, N=N, M=M, SigmaM=SigmaM
        )
        UtilMNIST.print_and_log("DONE")

        literation = 0  # 迭代次数

        avg_f_list = []
        last_xn_list = [0] * N
        accuracy_list = []
        matching = None  # 初始化

        while True:
            UtilMNIST.print_and_log(f"========================= literation: {literation + 1} =========================")

            # DataOwner自己报数据质量的机会只有一次
            if literation == 0:
                UtilMNIST.print_and_log(f"----- literation {literation + 1}: 为 DataOwner 的数据添加噪声 -----")
                dataowner_add_noise(dataowners, 0.1)
                UtilMNIST.print_and_log("DONE")

                UtilMNIST.print_and_log(f"----- literation {literation + 1}: 计算 DataOwner 的数据质量 -----")
                avg_f_list = evaluate_data_quality(dataowners, N)  # 传入 N
                UtilMNIST.print_and_log("DONE")

            UtilMNIST.print_and_log(
                f"----- literation {literation + 1}: 计算 ModelOwner 总体支付和 DataOwners 最优数据量 -----")
            xn_list, pn_list, best_Eta, U_Eta, U_qn = calculate_optimal_payment_and_data(
                avg_f_list, last_xn_list, N, Rho_val, Lambda_val
            )
            last_xn_list = xn_list

            # 【Mini版】: 提前中止
            if literation > adjustment_literation:
                UtilMNIST.print_and_log(f"accuracy_list: {accuracy_list}")
                accuracy_list_total.append(accuracy_list)
                break

            UtilMNIST.print_and_log(f"----- literation {literation + 1}: DataOwner 分配 ModelOwner 的支付 -----")
            compute_contribution_rates(xn_list, avg_f_list, pn_list, best_Eta, N)  # 传入 N
            UtilMNIST.print_and_log("DONE")

            # 一旦匹配成功，就无法改变
            if literation == 0:
                UtilMNIST.print_and_log(f"----- literation {literation + 1}: 匹配 DataOwner 和 ComputingCenter -----")
                matching = match_data_owners_to_cpc(
                    xn_list, ComputingCenters, dataowners, SigmaM, N, Rho_val
                )
                UtilMNIST.print_and_log("DONE")

            UtilMNIST.print_and_log(f"----- literation {literation + 1}: DataOwner 向 ComputingCenter 提交数据 -----")
            submit_data_to_cpc(matching, dataowners, ComputingCenters, xn_list, pn_list)
            UtilMNIST.print_and_log("DONE")

            UtilMNIST.print_and_log(f"----- literation {literation + 1}: 模型训练 -----")
            avg_f_list, new_accuracy = train_model_with_cpc(
                matching, ComputingCenters, test_images, test_labels,
                literation, avg_f_list, adjustment_literation,
                force_update=True, N=N
            )
            # 构建精准度列表
            accuracy_list.append(new_accuracy)
            UtilMNIST.print_and_log(f"[记录精度] 第{literation + 1}轮精度: {new_accuracy:.4f}")
            UtilMNIST.print_and_log(f"accuracy_list: {accuracy_list}")

            UtilMNIST.print_and_log("DONE")

            literation += 1

    UtilMNIST.print_and_log("\n===== (PGI-RDFL Original - MINI TEST) 实验最终结果 =====")
    UtilMNIST.print_and_log(f"accuracy_list_total: {accuracy_list_total}")

    if accuracy_list_total:
        final_run_accuracy = accuracy_list_total[0]
        if final_run_accuracy:
            UtilMNIST.print_and_log(f"最终精度: {final_run_accuracy[-1]:.4f}")
            UtilMNIST.print_and_log(f"平均精度: {np.mean(final_run_accuracy):.4f}")
            if len(final_run_accuracy) > 1:
                UtilMNIST.print_and_log(f"精度提升: {(final_run_accuracy[-1] - final_run_accuracy[0]):.4f}")
            else:
                UtilMNIST.print_and_log(f"精度提升: 0.0000 (仅一轮)")
        else:
            UtilMNIST.print_and_log("未记录任何精度。")