"""
基于 PGI-RDFL 修改的固定 Eta 和排名选择模型 - 精度实验
- 领导者（模型拥有者）决定一个固定的总支付 η
- 数据拥有者（跟随者）通过 Stackelberg 博弈计算潜在数据量 x_n
- 客户端选择机制：
    - 计算一个综合得分： score = 0.75 * x_n + 0.25 * f_n
    - 按得分排序，选择 Top-K (例如 K=10) 的客户端参与 (pn_list = 1)
- 去除了fn的动态调整
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


# 为联邦学习任务做准备工作
def ready_for_task(rate, N, M, SigmaM):  # 添加了 N, M, SigmaM 参数
    project_root = get_project_root()

    train_images_path = f"{project_root}/data/dataset/MNIST/train-images.idx3-ubyte"
    train_labels_path = f"{project_root}/data/dataset/MNIST/train-labels.idx1-ubyte"
    test_images_path = f"{project_root}/data/dataset/MNIST/t10k-images.idx3-ubyte"
    test_labels_path = f"{project_root}/data/dataset/MNIST/t10k-labels.idx1-ubyte"

    # 加载训练数据和测试数据
    train_images, train_labels = UtilMNIST.load_mnist_dataset(train_images_path, train_labels_path)
    test_images, test_labels = UtilMNIST.load_mnist_dataset(test_images_path, test_labels_path)

    # 创建DataOwner对象数组
    dataowners = [DataOwner(Lambda=Lambda, Rho=Rho) for _ in range(N)]  # 假设有N个DataOwner

    # 切分数据
    UtilMNIST.split_data_to_dataowners_with_large_gap(dataowners, train_images, train_labels)

    # 初始化ModelOwner
    modelowner = ModelOwner(model=init_model(rate=rate))

    # 初始化ComputingCenter
    ComputingCenters = [ComputingCenter(Lambda, Epsilon, SigmaM[i]) for i in range(M)]

    return dataowners, modelowner, ComputingCenters, test_images, test_labels


# modelowner的初始model
def init_model(rate):
    """
    用于初始化一个模型给modeloowner
    :param rate: 初始数据占MNIST的比例
    :return:
    """
    UtilMNIST.print_and_log(f"初始数据占MNIST的比例：{rate * 100}%")
    UtilMNIST.print_and_log("model initing...")

    project_root = get_project_root()

    train_images_path = f"{project_root}/data/dataset/MNIST/train-images.idx3-ubyte"
    train_labels_path = f"{project_root}/data/dataset/MNIST/train-labels.idx1-ubyte"

    # 加载训练数据和测试数据
    train_images, train_labels = UtilMNIST.load_mnist_dataset(train_images_path, train_labels_path)

    # 获取图像数量
    num_images = train_images.shape[0]
    # 计算需要选取的图像数量
    num_samples = int(num_images * rate)
    # 随机生成索引
    indices = np.random.choice(num_images, num_samples, replace=False)
    # 使用随机索引选取数据
    train_labels = train_labels[indices]
    train_images = train_images[indices]

    train_loader = UtilMNIST.create_data_loader(train_images, train_labels, batch_size=64, shuffle=True)

    # 创建CNN模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MNISTCNN(num_classes=10).to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 如果不存在初始化模型，就训练模型，如果存在，就加载到model中
    model_save_path = f"{project_root}/data/model/initial/mnist_cnn_initial_model"

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


# ModelOwner发布任务， DataOwner计算数据质量（Dataowner自己计算）
def evaluate_data_quality(dataowners, N):  # 添加N参数
    """
    加噪声，模拟DataOwner的数据不好的情况
    :param dataowners:
    :param N: 客户端总数
    :return:
    """
    avg_f_list = []  # 初始化

    # 评价数据质量
    for i, do in enumerate(dataowners):

        mse_scores = UtilMNIST.evaluate_quality(do, metric="mse")
        snr_scores = UtilMNIST.evaluate_quality(do, metric="snr")

        # 计算图像的质量得分
        mse_sum = 0
        for j, (mse, snr) in enumerate(zip(mse_scores, snr_scores)):
            # UtilMNIST.print_and_log(parent_path,f"DataOwner{i + 1}: Image {j + 1}: MSE = {mse:.4f}, SNR = {snr:.2f} dB")
            mse_sum += mse
        avg_mse = mse_sum / len(mse_scores)
        avg_f_list.append(1 - avg_mse)

    UtilMNIST.print_and_log("DataOwners自行评估数据质量：")
    UtilMNIST.print_and_log(f"数据质量列表avg_f_list: {avg_f_list}")
    UtilMNIST.print_and_log(f"归一化后的数据质量列表avg_f_list: {UtilMNIST.normalize_list(avg_f_list)}")

    return UtilMNIST.normalize_list(avg_f_list)


# 【新函数】: 替换 calculate_optimal_payment_and_data
def calculate_fixed_eta_and_ranked_selection(avg_f_list, last_xn_list, N, Rho_val, Lambda_val, fixed_eta,
                                             num_to_select):
    """
    使用固定Eta，并根据综合得分排名选择Top-K客户端
    综合得分 = 0.75 * 数据量 (x_n) + 0.25 * 数据质量 (f_n)

    :param avg_f_list: 数据质量列表
    :param last_xn_list: 上一轮的数据量列表
    :param N: 客户端总数
    :param Rho_val: 单位数据训练费用
    :param Lambda_val: 市场调整因子
    :param fixed_eta: 固定的总支付
    :param num_to_select: 要选择的客户端数量 (K)
    :return: (xn_list, pn_list, best_Eta, U_Eta, U_qn)
    """

    # 1. 仍然使用Stackelberg求解器来获取“潜在”的数据量贡献
    #    这是基于你原始代码的逻辑
    unit_cost = Rho_val * Lambda_val
    stackelberg_solver = Stackelberg(N, unit_cost, avg_f_list)

    # 我们只关心 q_star，它代表了博弈后的潜在贡献
    _, _, q_star, _, _ = stackelberg_solver.solve()

    # 2. 将q_star转化为x_opt (潜在数据量)，同样基于你原始代码的逻辑
    x_opt = UtilMNIST.power_transform_then_min_max_normalize(q_star)

    # 3. 【新】计算综合得分并选择Top-K
    #    综合得分 = 0.75 * x_n + 0.25 * f_n
    #    (x_opt 和 avg_f_list 都已经被归一化到 0-1)
    combined_score = [0.75 * x + 0.25 * f for x, f in zip(x_opt, avg_f_list)]

    # 确保选择的数量不超过总数
    k = min(num_to_select, N)

    # 获取Top-K的索引
    top_k_indices = np.argsort(combined_score)[-k:]

    # 4. 【新】创建 pn_list (参与列表)
    pn_list = np.zeros(N)
    pn_list[top_k_indices] = 1

    UtilMNIST.print_and_log(f"[Fixed Eta Model] 选择了 {k} 个客户端 (Top-{num_to_select})")
    UtilMNIST.print_and_log(f"[Fixed Eta Model] 客户端索引: {top_k_indices}")

    # 5. 确定最终数据量 (与你原始逻辑一致)
    xn_list = UtilMNIST.compare_elements(x_opt, last_xn_list)

    # 6. 【新】使用固定的 Eta 和计算效用
    best_Eta = fixed_eta

    # 计算实际的总质量贡献 q_n = p_n * x_n * f_n
    q_contributions = np.array([a * b * c for a, b, c in zip(xn_list, avg_f_list, pn_list)])
    total_quality = np.sum(q_contributions)

    # 领导者效用
    leader_utility = total_quality - best_Eta

    # 跟随者平均效用 (仅计算参与者)
    avg_follower_utility = 0
    if total_quality > 1e-9:
        price_per_quality = best_Eta / total_quality
        # 效用 = 收益 - 成本。 U_n = q_n * (pi - C_q)
        # C_q = unit_cost (单位质量成本)
        follower_utilities = q_contributions * (price_per_quality - unit_cost)

        # 计算参与者的平均效用
        participants_utility = follower_utilities[pn_list == 1]
        avg_follower_utility = np.mean(participants_utility) if k > 0 else 0

    UtilMNIST.print_and_log(f"[Fixed Eta Model] 固定总支付η: {best_Eta:.4f}")
    UtilMNIST.print_and_log(f"[Fixed Eta Model] 领导者效用: {leader_utility:.4f}")
    UtilMNIST.print_and_log(f"[Fixed Eta Model] 跟随者平均效用: {avg_follower_utility:.4f}")

    return xn_list, pn_list, best_Eta, leader_utility, avg_follower_utility


# DataOwner结合自身数据质量来算模型贡献，分配ModelOwner的支付
def compute_contribution_rates(xn_list, avg_f_list, pn_list, best_Eta, N):  # 添加N参数
    """
    DataOwner结合自身数据质量来算模型贡献，分配ModelOwner的支付
    :param xn_list:
    :param avg_f_list:
    :param pn_list:
    :param best_Eta:
    :param N: 客户端总数
    :return:
    """
    # 计算qn（qn = xn*fn*pn）
    contributions = [a * b for a, b in zip(xn_list, avg_f_list)]
    contributions = [a * b for a, b in zip(contributions, pn_list)]

    sum_qn = sum(contributions)

    UtilMNIST.print_and_log(f"ModelOwner的总支付：{best_Eta}")
    if sum_qn == 0:
        UtilMNIST.print_and_log("总贡献为0，没有支付分配。")
        return

    for i in range(N):  # 遍历所有N个客户端
        if pn_list[i] == 0:  # 如果客户端未参与
            UtilMNIST.print_and_log(f"DataOwner{i + 1}: 未参与 (pn=0)")
            continue

        UtilMNIST.print_and_log(f"DataOwner{i + 1}:")
        UtilMNIST.print_and_log(
            f"pn:{pn_list[i]}; xn:{xn_list[i]}; fn:{avg_f_list[i]:.4f}; 分配到的支付：{contributions[i] / sum_qn * best_Eta:.4f}")


# 匹配DataOwner和ComputingCenter
def match_data_owners_to_cpc(xn_list, ComputingCenters, dataowners, SigmaM, N, Rho):  # 添加 SigmaM, N, Rho 参数
    """
    匹配DataOwner和ComputingCenter
    :param xn_list:
    :param ComputingCenters:
    :param dataowners:
    :param SigmaM:
    :param N:
    :param Rho:
    :return:
    """
    proposals = GaleShapley.make_proposals(SigmaM, N)
    preferences = GaleShapley.make_preferences(xn_list, ComputingCenters, Rho, dataowners)
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

        # 【重要】: 只有被选中的人 (pn_list[i] == 1) 才提交数据
        if pn_list[dataowner_index] == 0:
            UtilMNIST.print_and_log(f"DataOwner{dataowner_index + 1} 未被选中，不提交数据。")
            continue

        UtilMNIST.print_and_log(f"DataOwner{dataowner_index + 1} 把数据交给 ComputingCenter{ComputingCenter_index + 1}")

        # 数据量 = 潜在数据量 * 参与率 (0或1)
        data_rate_list = [a * b for a, b in zip(xn_list, pn_list)]

        UtilMNIST.dataowner_pass_data_to_cpc(dataowners[dataowner_index],
                                             ComputingCenters[ComputingCenter_index],
                                             data_rate_list[dataowner_index])


# 【修改后】: 使用ComputingCenter进行模型训练和全局模型的更新 (去除了动态调整)
def train_model_with_cpc(matching, cpcs, test_images, test_labels, force_update):
    """
    使用CPC进行模型训练和全局模型的更新
    【已移除】: 动态调整fn的逻辑

    :param matching:
    :param cpcs:
    :param test_images:
    :param test_labels:
    :param force_update:
    :return: new_accuracy (只返回精度)
    """

    # 【已删除】:
    # if literation == adjustment_literation:
    #   ... (动态调整fn的代码块) ...
    # 【删除完毕】

    # 准备训练
    project_root = get_project_root()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_loader = UtilMNIST.create_data_loader(test_images, test_labels, batch_size=64, shuffle=False)

    new_accuracy = fine_tune_model(cpcs, matching, test_loader, lr=1e-5, device=str(device), num_epochs=5,
                                   force_update=force_update,
                                   model_path=f"{project_root}/data/model/mnist_cnn_model")

    return new_accuracy  # 只返回 new_accuracy


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
    # 【注意】: 如果没有客户端提交数据 (updated_params_list为空)，则不进行聚合
    if not updated_params_list:
        UtilMNIST.print_and_log("没有客户端提交模型参数，跳过聚合和更新。")
        # 返回上一轮的精度
        test_loader = UtilMNIST.create_data_loader(test_images, test_labels, batch_size=64, shuffle=False)
        current_accuracy = model.evaluate_model(test_loader, device=device)
        return current_accuracy

    avg_params = average_models_parameters(updated_params_list)
    UtilMNIST.print_and_log("参数聚合完成")

    # 6. 选择更新
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
        f"**** {parent_path} (Fixed Eta & Ranked Selection) 运行时间： {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ****")

    # 记录精确度
    accuracy_list_total = []

    # 【新】定义实验超参数
    FIXED_ETA = 50.0  # (要求1: 固定Eta, 这是一个示例值)
    NUM_TO_SELECT = 3  # (要求3: 选择10个客户端)

    # 从这里开始进行不同数量客户端的循环 (前闭后开)
    for n in [9]:  # 你的示例中 N=10 (n=9)
        UtilMNIST.print_and_log(f"========================= 客户端总数 N: {n + 1} =========================")
        UtilMNIST.print_and_log(f"========================= 固定支付 ETA: {FIXED_ETA} =========================")
        UtilMNIST.print_and_log(f"========================= 选择数量 K: {NUM_TO_SELECT} =========================")

        UtilMNIST.print_and_log("---------------------------------- 定义参数值 ----------------------------------")
        Lambda_val, Rho_val, Alpha_val, Epsilon_val, N, M, SigmaM = define_parameters(
            Lambda=Lambda, Rho=Rho, Alpha=Alpha, Epsilon=Epsilon, M=n + 1, N=n + 1, SigmaM=[1] * (n + 1)
        )
        UtilMNIST.print_and_log("DONE")

        UtilMNIST.print_and_log("---------------------------------- 准备工作 ----------------------------------")
        # 修正了 ready_for_task 的调用
        dataowners, modelowner, ComputingCenters, test_images, test_labels = ready_for_task(
            rate=0.001, N=N, M=M, SigmaM=SigmaM
        )
        UtilMNIST.print_and_log("DONE")

        literation = 0  # 迭代次数
        # (要求2: adjustment_literation 现在只用于控制总迭代次数)
        num_iterations = adjustment_literation
        avg_f_list = []
        last_xn_list = [0] * N
        accuracy_list = []
        matching = None  # 初始化 matching

        while literation < num_iterations:
            UtilMNIST.print_and_log(
                f"========================= literation: {literation + 1} / {num_iterations} =========================")

            # DataOwner自己报数据质量的机会只有一次 (固定，不再动态调整)
            if literation == 0:
                UtilMNIST.print_and_log(f"----- literation {literation + 1}: 为 DataOwner 的数据添加噪声 -----")
                dataowner_add_noise(dataowners, 0.1)
                UtilMNIST.print_and_log("DONE")

                UtilMNIST.print_and_log(f"----- literation {literation + 1}: 计算 DataOwner 的数据质量 -----")
                avg_f_list = evaluate_data_quality(dataowners, N)  # 传入 N
                UtilMNIST.print_and_log("DONE")

            UtilMNIST.print_and_log(
                f"----- literation {literation + 1}: [Fixed Eta & Rank] 计算支付和数据量 -----")

            # 【修改后】: 调用新函数
            xn_list, pn_list, best_Eta, U_Eta, U_qn = calculate_fixed_eta_and_ranked_selection(
                avg_f_list, last_xn_list, N, Rho_val, Lambda_val,
                fixed_eta=FIXED_ETA,
                num_to_select=NUM_TO_SELECT
            )
            last_xn_list = xn_list
            UtilMNIST.print_and_log("DONE")

            # 【已删除】: 移除 'if literation > adjustment_literation' 的中止逻辑

            UtilMNIST.print_and_log(f"----- literation {literation + 1}: DataOwner 分配 ModelOwner 的支付 -----")
            compute_contribution_rates(xn_list, avg_f_list, pn_list, best_Eta, N)  # 传入 N
            UtilMNIST.print_and_log("DONE")

            # 匹配只在第一轮进行
            if literation == 0:
                UtilMNIST.print_and_log(f"----- literation {literation + 1}: 匹配 DataOwner 和 ComputingCenter -----")
                matching = match_data_owners_to_cpc(xn_list, ComputingCenters, dataowners, SigmaM, N, Rho_val)  # 传入所需参数
                UtilMNIST.print_and_log("DONE")

            UtilMNIST.print_and_log(f"----- literation {literation + 1}: DataOwner 向 ComputingCenter 提交数据 -----")
            submit_data_to_cpc(matching, dataowners, ComputingCenters, xn_list, pn_list)
            UtilMNIST.print_and_log("DONE")

            UtilMNIST.print_and_log(f"----- literation {literation + 1}: 模型训练 -----")

            # 【修改后】: 调用修改后的 train_model_with_cpc
            # (不再返回 avg_f_list, 也不再传入 literation, adjustment_literation)
            new_accuracy = train_model_with_cpc(
                matching, ComputingCenters, test_images, test_labels,
                force_update=True
            )

            # 构建精准度列表
            accuracy_list.append(new_accuracy)
            UtilMNIST.print_and_log(f"[记录精度] 第{literation + 1}轮精度: {new_accuracy:.4f}")
            UtilMNIST.print_and_log(f"accuracy_list: {accuracy_list}")
            UtilMNIST.print_and_log("DONE")

            literation += 1

        # 循环结束后，保存该 N 值下的精度列表
        accuracy_list_total.append(accuracy_list)

    UtilMNIST.print_and_log("\n===== (Fixed Eta & Ranked Selection) 实验最终结果 =====")
    UtilMNIST.print_and_log(f"accuracy_list_total: {accuracy_list_total}")

    # 假设只跑了一组n (n=9)
    if accuracy_list_total:
        final_run_accuracy = accuracy_list_total[0]
        UtilMNIST.print_and_log(f"最终精度: {final_run_accuracy[-1]:.4f}")
        UtilMNIST.print_and_log(f"平均精度: {np.mean(final_run_accuracy):.4f}")
        if len(final_run_accuracy) > 1:
            UtilMNIST.print_and_log(f"精度提升: {(final_run_accuracy[-1] - final_run_accuracy[0]):.4f}")
        else:
            UtilMNIST.print_and_log(f"精度提升: 0.0000 (仅一轮)")