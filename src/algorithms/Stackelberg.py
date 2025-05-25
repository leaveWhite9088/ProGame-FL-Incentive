import numpy as np
import random
from src.utils.UtilMNIST import UtilMNIST
from src.algorithms.JointOptimization import JointOptimization

class Stackelberg:
    """
    实现联邦学习中的Stackelberg博弈算法
    
    Stackelberg博弈解决方案：领导者(模型拥有者)预期跟随者(数据拥有者)的最佳响应，
    并在此基础上选择最优策略。这是一个两阶段博弈的解。
    """
    
    def __init__(self, N, C, q_max_vector):
        """
        初始化Stackelberg博弈
        
        :param N: 数据拥有者数量
        :param C: 每个数据拥有者的单位成本（假设相同，或是一个向量C_vector）
        :param q_max_vector: 每个数据拥有者的最大质量向量 (q1_max, ..., qN_max)
        """
        self.N = N
        
        # 处理C可能是标量或向量的情况
        if isinstance(C, (int, float)):
            self.C = np.ones(N) * C
        else:
            self.C = np.array(C)
            
        self.q_max_vector = np.array(q_max_vector)
        
        # 创建联合优化求解器
        self.joint_optimizer = JointOptimization(N, C, q_max_vector)
    
    def solve(self):
        """
        求解完整的Stackelberg博弈
        
        通过调用JointOptimization完成模型拥有者的参数联合优化，然后计算数据拥有者的最佳响应。
        
        :return: 元组 (p_star, eta_star, q_star, leader_utility, follower_utilities, social_welfare)
                p_star: 均衡概率分配
                eta_star: 均衡总支付
                q_star: 均衡质量选择
                leader_utility: 领导者效用
                follower_utilities: 跟随者效用向量
                social_welfare: 社会福利(所有参与者效用总和)
        """
        UtilMNIST.print_and_log("开始求解Stackelberg博弈...")
        
        # 1. 调用JointOptimization求解领导者优化问题
        # 注意：这已经包含了将数据拥有者响应嵌入到领导者优化中的过程
        p_star, eta_star, q_star, leader_utility = self.joint_optimizer.optimize()
        
        # 2. 计算各数据拥有者(跟随者)在均衡时的效用
        S_star_actual = np.sum(p_star * q_star)

        # 检查 S* 是否接近于零以避免除零错误
        if S_star_actual > 1e-9:  # 使用一个小的正阈值
            price_at_equilibrium = eta_star / S_star_actual
            # 计算每个跟随者的效用
            follower_utilities = q_star * (price_at_equilibrium - self.C)
        else:
            # 如果 S* 为零或接近零 (例如 eta*=0 或 q*=0)
            # 那么 q_n * (price - C_n) 中的 q_n 通常也为零
            # 因此效用为零
            follower_utilities = np.zeros(self.N)

        # 3. 计算跟随者总效用
        total_follower_utility = np.sum(follower_utilities)
        
        # 输出完整的均衡结果
        UtilMNIST.print_and_log("Stackelberg博弈求解完成，以下为结果")
        UtilMNIST.print_and_log(f"均衡p: {p_star}")
        UtilMNIST.print_and_log(f"均衡eta: {eta_star:.4f}")
        UtilMNIST.print_and_log(f"均衡q: {q_star}")
        UtilMNIST.print_and_log(f"领导者效用: {leader_utility:.4f}")
        UtilMNIST.print_and_log(f"跟随者总效用: {total_follower_utility:.4f}")
        
        # 可以进一步分析各数据拥有者的个体效用
        # for n in range(self.N):
        #     UtilMNIST.print_and_log(f"数据拥有者{n+1}效用: {follower_utilities[n]:.4f}")
        
        return p_star, eta_star, q_star, leader_utility, total_follower_utility / self.N
        
    def solve_with_fixed_eta(self, eta_fixed, max_iterations=100, tolerance=1e-6):
        """
        在固定总支付eta的情况下求解追随者(数据拥有者)间的博弈均衡
        
        与solve函数保持一致的流程，唯一区别是使用固定的eta值而非优化得到的eta
        
        :param eta_fixed: 固定的总支付金额
        :param max_iterations: 最大迭代次数
        :param tolerance: 收敛容差
        :return: 元组 (p_star, eta_fixed, q_star, leader_utility, avg_follower_utility)
                p_star: 均衡概率分配
                eta_fixed: 固定的总支付金额
                q_star: 均衡质量选择
                leader_utility: 领导者效用
                avg_follower_utility: 平均跟随者效用
        """
        UtilMNIST.print_and_log(f"开始求解固定总支付eta={eta_fixed}下的Stackelberg博弈...")
        
        # 1. 调用JointOptimization求解，但不使用其返回的eta，而是使用固定的eta_fixed
        p_star, _, q_star, _ = self.joint_optimizer.optimize()
        
        # 2. 计算各数据拥有者(跟随者)在均衡时的效用
        S_star_actual = np.sum(p_star * q_star)

        # 计算领导者效用
        leader_utility = np.log(1 + S_star_actual) - eta_fixed

        # 检查 S* 是否接近于零以避免除零错误
        if S_star_actual > 1e-9:  # 使用一个小的正阈值
            price_at_equilibrium = eta_fixed / S_star_actual
            # 计算每个跟随者的效用
            follower_utilities = q_star * (price_at_equilibrium - self.C)
        else:
            # 如果 S* 为零或接近零 (例如 eta*=0 或 q*=0)
            # 那么 q_n * (price - C_n) 中的 q_n 通常也为零
            # 因此效用为零
            follower_utilities = np.zeros(self.N)

        # 3. 计算跟随者总效用
        total_follower_utility = np.sum(follower_utilities)
        avg_follower_utility = total_follower_utility / self.N
        
        # 输出完整的均衡结果
        UtilMNIST.print_and_log("固定总支付下的Stackelberg博弈求解完成，以下为结果")
        UtilMNIST.print_and_log(f"均衡p: {p_star}")
        UtilMNIST.print_and_log(f"固定eta: {eta_fixed:.4f}")
        UtilMNIST.print_and_log(f"均衡q: {q_star}")
        UtilMNIST.print_and_log(f"领导者效用: {leader_utility:.4f}")
        UtilMNIST.print_and_log(f"跟随者平均效用: {avg_follower_utility:.4f}")
        
        return p_star, eta_fixed, q_star, leader_utility, avg_follower_utility

    def solve_with_random_eta(self, eta_max, max_iterations=100, tolerance=1e-6):
        """
        在0到eta_max之间随机选择总支付eta的情况下求解追随者(数据拥有者)间的博弈均衡
        
        与solve_with_fixed_eta函数保持一致的流程，唯一区别是使用随机选择的eta值
        
        :param eta_max: eta的上限值，实际使用的eta将在0到此值之间随机选择
        :param max_iterations: 最大迭代次数
        :param tolerance: 收敛容差
        :return: 元组 (p_star, random_eta, q_star, leader_utility, avg_follower_utility)
                p_star: 均衡概率分配
                random_eta: 随机选择的总支付金额
                q_star: 均衡质量选择
                leader_utility: 领导者效用
                avg_follower_utility: 平均跟随者效用
        """
        # 在0到eta_max之间随机选择一个eta值,
        random_eta = random.uniform(0, eta_max)
        
        UtilMNIST.print_and_log(f"开始求解随机总支付eta={random_eta:.4f}(上限为{eta_max:.4f})下的Stackelberg博弈...")
        
        # 1. 调用JointOptimization求解，但不使用其返回的eta，而是使用随机选择的eta
        p_star, _, q_star, _ = self.joint_optimizer.optimize()
        
        # 2. 计算各数据拥有者(跟随者)在均衡时的效用
        S_star_actual = np.sum(p_star * q_star)

        # 计算领导者效用
        leader_utility = np.log(1 + S_star_actual) - random_eta

        # 检查 S* 是否接近于零以避免除零错误
        if S_star_actual > 1e-9:  # 使用一个小的正阈值
            price_at_equilibrium = random_eta / S_star_actual
            # 计算每个跟随者的效用
            follower_utilities = q_star * (price_at_equilibrium - self.C)
        else:
            # 如果 S* 为零或接近零 (例如 eta*=0 或 q*=0)
            # 那么 q_n * (price - C_n) 中的 q_n 通常也为零
            # 因此效用为零
            follower_utilities = np.zeros(self.N)

        # 3. 计算跟随者总效用
        total_follower_utility = np.sum(follower_utilities)
        avg_follower_utility = total_follower_utility / self.N
        
        # 输出完整的均衡结果
        UtilMNIST.print_and_log("随机总支付下的Stackelberg博弈求解完成，以下为结果")
        UtilMNIST.print_and_log(f"均衡p: {p_star}")
        UtilMNIST.print_and_log(f"随机eta: {random_eta:.4f}")
        UtilMNIST.print_and_log(f"均衡q: {q_star}")
        UtilMNIST.print_and_log(f"领导者效用: {leader_utility:.4f}")
        UtilMNIST.print_and_log(f"跟随者平均效用: {avg_follower_utility:.4f}")
        
        return p_star, random_eta, q_star, leader_utility, avg_follower_utility

    def solve_with_fixed_p(self, p_fixed):
        """
        在固定概率分配p的情况下求解Stackelberg博弈
        
        :param p_fixed: 固定的概率分配向量 (p_1, ..., p_N)
        :return: 元组 (p_fixed, eta_star, q_star, leader_utility, avg_follower_utility)
                p_fixed: 输入的固定概率分配
                eta_star: 优化得到的总支付
                q_star: 均衡质量选择
                leader_utility: 领导者效用
                avg_follower_utility: 平均跟随者效用
        """
        UtilMNIST.print_and_log("开始求解固定概率分配下的Stackelberg博弈...")
        
        # 1. 调用JointOptimization的固定p优化函数
        p_star, eta_star, q_star, leader_utility = self.joint_optimizer.optimize_with_fixed_p(p_fixed)
        
        # 2. 计算各数据拥有者(跟随者)在均衡时的效用
        S_star_actual = np.sum(p_star * q_star)

        # 检查 S* 是否接近于零以避免除零错误
        if S_star_actual > 1e-9:  # 使用一个小的正阈值
            price_at_equilibrium = eta_star / S_star_actual
            # 计算每个跟随者的效用
            follower_utilities = q_star * (price_at_equilibrium - self.C)
        else:
            # 如果 S* 为零或接近零 (例如 eta*=0 或 q*=0)
            # 那么 q_n * (price - C_n) 中的 q_n 通常也为零
            # 因此效用为零
            follower_utilities = np.zeros(self.N)

        # 3. 计算跟随者总效用和平均效用
        total_follower_utility = np.sum(follower_utilities)
        avg_follower_utility = total_follower_utility / self.N
        
        # 输出完整的均衡结果
        UtilMNIST.print_and_log("固定概率分配下的Stackelberg博弈求解完成，以下为结果")
        UtilMNIST.print_and_log(f"固定p: {p_star}")
        UtilMNIST.print_and_log(f"优化得到的eta: {eta_star:.4f}")
        UtilMNIST.print_and_log(f"均衡q: {q_star}")
        UtilMNIST.print_and_log(f"领导者效用: {leader_utility:.4f}")
        UtilMNIST.print_and_log(f"跟随者平均效用: {avg_follower_utility:.4f}")
        
        return p_star, eta_star, q_star, leader_utility, avg_follower_utility