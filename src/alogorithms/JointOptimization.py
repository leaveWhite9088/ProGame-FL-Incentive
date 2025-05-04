import numpy as np
from scipy.optimize import minimize
from src.utils.MNISTUtil import MNISTUtil
from src.alogorithms.CournotGame import CournotGame

class JointOptimization:
    """
    实现模型拥有者的联合参数优化算法
    
    模型拥有者联合优化两个决策变量：p（概率分配）和 η（总支付）
    以最大化自己的效用函数
    """
    
    def __init__(self, N, C, q_max_vector):
        """
        初始化联合优化问题
        
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
        
        # 创建古诺博弈求解器用于计算数据拥有者响应
        self.cournot_solver = CournotGame(N, C, q_max_vector)
        
    def optimize(self):
        """
        联合优化模型拥有者的参数 p 和 eta
        
        :return: 元组 (p_optimal, eta_optimal, q_optimal, leader_utility)
                p_optimal: 最优概率分配
                eta_optimal: 最优总支付
                q_optimal: 对应的数据拥有者最优质量选择
                leader_utility: 领导者最大效用
        """
        # 定义模型拥有者目标函数: U_M = sum(p_n * q_n) - eta
        def objective_function(variables):
            # 变量包括: [p_1, ..., p_(N-1), eta]
            # 最后一个p通过约束条件确定: p_N = 1 - sum(p_1, ..., p_(N-1))
            p_partial = variables[:-1]
            eta = variables[-1]
            
            # 构建完整的p向量
            p_vector = np.zeros(self.N)
            p_vector[:self.N-1] = p_partial
            p_vector[self.N-1] = 1 - np.sum(p_partial)
            
            # 如果p_N为负，则这不是有效的解
            if p_vector[self.N-1] < 0:
                return 1e10  # 返回一个很大的值作为惩罚
            
            # 计算对应于(p,eta)的数据拥有者最优响应q
            q_star_vector = self.cournot_solver.compute_equilibrium(p_vector, eta)
            
            # 计算模型拥有者效用
            utility = np.sum(p_vector * q_star_vector) - eta
            
            # 因为minimize函数是最小化目标，所以返回负的效用
            return -utility
        
        # 变量边界: 0 <= p_n <= 1 (只有N-1个自由变量), eta >= 0
        bounds = [(0, 1) for _ in range(self.N-1)]  # p_1, ..., p_(N-1)的边界
        bounds.append((0, None))  # eta的边界
        
        # 初始猜测
        initial_p = np.ones(self.N-1) / self.N  # 平均分配概率给前N-1个
        initial_eta = 1.0  # 初始支付
        initial_guess = np.concatenate([initial_p, [initial_eta]])
        
        # 使用SLSQP方法求解约束优化问题
        MNISTUtil.print_and_log("开始模型拥有者联合参数优化...")
        result = minimize(
            objective_function,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            options={'disp': True, 'maxiter': 500}
        )
        
        if result.success:
            MNISTUtil.print_and_log("模型拥有者联合参数优化成功!")
        else:
            MNISTUtil.print_and_log(f"警告: 优化未成功收敛。原因: {result.message}")
        
        # 提取结果
        optimal_variables = result.x
        p_partial_optimal = optimal_variables[:-1]
        eta_optimal = optimal_variables[-1]
        
        # 构建完整的最优p向量
        p_optimal = np.zeros(self.N)
        p_optimal[:self.N-1] = p_partial_optimal
        p_optimal[self.N-1] = 1 - np.sum(p_partial_optimal)
        
        # 计算对应的数据拥有者质量选择
        q_optimal = self.cournot_solver.compute_equilibrium(p_optimal, eta_optimal)
        
        # 模型拥有者最大效用
        leader_utility = -result.fun
        
        MNISTUtil.print_and_log(f"最优p: {p_optimal}")
        MNISTUtil.print_and_log(f"最优eta: {eta_optimal:.4f}")
        MNISTUtil.print_and_log(f"对应的q: {q_optimal}")
        MNISTUtil.print_and_log(f"模型拥有者最大效用: {leader_utility:.4f}")
        
        return p_optimal, eta_optimal, q_optimal, leader_utility
