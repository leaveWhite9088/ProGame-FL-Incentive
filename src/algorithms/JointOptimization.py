import numpy as np
from scipy.optimize import minimize
from src.utils import UtilT
from src.algorithms.CournotGame import CournotGame
from src.global_variable import Theta

class JointOptimization:
    """
    实现模型拥有者的联合参数优化算法 (Algorithm 3: Joint Optimization)
    
    模型拥有者联合优化两个决策变量：P*（概率分配）和 η*（总支付）
    以最大化自己的效用函数 U_s = max Σp_nq_n - η
    
    基于论文中的Lemma 1和Lemma 2，证明了效用函数关于p_n和η的凹性
    """
    
    def __init__(self, N, C, q_max_vector, lambda_param=0.9):
        """
        初始化联合优化问题
        
        :param N: 数据拥有者数量
        :param C: 每个数据拥有者的单位成本（λρ，假设相同，或是一个向量C_vector）
        :param q_max_vector: 每个数据拥有者的最大质量向量 (|X_1|, ..., |X_N|)
        :param lambda_param: 市场调节因子λ（默认0.9，基于论文参数设置）
        """
        self.N = N
        self.lambda_param = lambda_param
        
        # 处理C可能是标量或向量的情况
        if isinstance(C, (int, float)):
            self.C = np.ones(N) * C
        else:
            self.C = np.array(C)
            
        self.q_max_vector = np.array(q_max_vector)
        
        # 创建古诺博弈求解器用于计算数据拥有者响应
        self.cournot_solver = CournotGame(N, C, q_max_vector, lambda_param)
        
    def optimize(self, max_iterations=100):
        """
        联合优化模型拥有者的参数 p 和 eta
        实现Algorithm 3: Joint Optimization
        
        根据论文，通过交替优化p和η来达到最优解
        
        :param max_iterations: 交替优化的最大迭代次数
        :return: 元组 (p_optimal, eta_optimal, q_optimal, leader_utility)
                p_optimal: 最优概率分配
                eta_optimal: 最优总支付
                q_optimal: 对应的数据拥有者最优质量选择
                leader_utility: 领导者最大效用
        """
        # Algorithm 3 Step 1: 初始化 η^0, P^0
        eta = 1.0  # 初始总支付
        p_vector = np.ones(self.N) / self.N  # 初始均匀概率分配
        
        # 记录历史值用于收敛判断
        eta_history = [eta]
        p_history = [p_vector.copy()]
        
        UtilT.print_and_log("开始模型拥有者联合参数优化 (Algorithm 3)...")
        
        # Algorithm 3 Step 2: while h ≤ H do
        for h in range(max_iterations):
            # 保存上一轮的值
            eta_prev = eta
            p_prev = p_vector.copy()
            
            # Step 4: 给定 η^(h-1), 更新 P^h
            p_vector = self._optimize_p_given_eta(eta, p_prev)
            
            # Step 5: 给定 P^(h-1), 更新 η^h
            eta = self._optimize_eta_given_p(p_vector)
            
            # Step 6: 给定 η^h 和 P^h, 更新 U_s
            q_star = self.cournot_solver.compute_equilibrium(p_vector, eta)
            leader_utility = np.sum(p_vector * q_star) * Theta - eta
            
            # 检查收敛
            eta_change = abs(eta - eta_prev)
            p_change = np.linalg.norm(p_vector - p_prev)
            
            if eta_change < 1e-6 and p_change < 1e-6:
                UtilT.print_and_log(f"联合优化在第{h+1}轮收敛")
                break
                
            eta_history.append(eta)
            p_history.append(p_vector.copy())
            
        # 计算最终的数据拥有者质量选择
        q_optimal = self.cournot_solver.compute_equilibrium(p_vector, eta)
        
        # 模型拥有者最大效用
        leader_utility = np.sum(p_vector * q_optimal) * Theta - eta
        
        UtilT.print_and_log("模型拥有者联合参数优化完成!")
        # UtilT.print_and_log(f"最优p: {p_vector}")
        # UtilT.print_and_log(f"最优eta: {eta:.4f}")
        # UtilT.print_and_log(f"对应的q: {q_optimal}")
        # UtilT.print_and_log(f"模型拥有者最大效用: {leader_utility:.4f}")
        
        return p_vector, eta, q_optimal, leader_utility
    
    def _optimize_p_given_eta(self, eta, p_init):
        """
        给定η，优化概率分配p
        基于Lemma 1的推导：p_n* = A/q_n * (1/(2λρx_n) - 1)
        其中 A = Σp_nq_n
        
        :param eta: 固定的总支付
        :param p_init: 初始概率分配
        :return: 优化后的概率分配
        """
        # 首先计算当前的q值
        q_current = self.cournot_solver.compute_equilibrium(p_init, eta)
        
        # 计算A = Σp_nq_n
        A = np.sum(p_init * q_current)
        
        # 获取数据拥有者的数据量 x_n (从q_max_vector获取)
        x_n = self.q_max_vector
        
        # 使用公式 (3.15): p_n* = A/q_n * (1/(2λρx_n) - 1)
        # 注意：这里需要迭代求解，因为A依赖于p，而p又依赖于A
        p_new = p_init.copy()
        
        for iteration in range(10):  # 最多迭代10次
            # 计算当前的q值
            q_current = self.cournot_solver.compute_equilibrium(p_new, eta)
            
            # 重新计算A
            A = np.sum(p_new * q_current)
            
            if A <= 0:  # 防止除零
                break
                
            # 使用解析公式计算新的p
            # p_n* = A/q_n * (1/(2λρx_n) - 1)
            # 这里假设λρ就是self.C（单位成本）
            lambda_rho = self.C  # λρ参数
            
            p_analytical = np.zeros(self.N)
            for n in range(self.N):
                if q_current[n] > 0 and x_n[n] > 0:
                    term = 1.0 / (2 * lambda_rho[n] * x_n[n]) - 1
                    if term > 0:
                        p_analytical[n] = (A / q_current[n]) * term
                    else:
                        p_analytical[n] = 0
                else:
                    p_analytical[n] = 0
            
            # 归一化p使其满足约束 Σp_n = 1
            p_sum = np.sum(p_analytical)
            if p_sum > 0:
                p_analytical = p_analytical / p_sum
            else:
                p_analytical = np.ones(self.N) / self.N  # 回退到均匀分布
            
            # 确保所有p_n都在[0,1]范围内
            p_analytical = np.clip(p_analytical, 0, 1)
            
            # 再次归一化
            p_analytical = p_analytical / np.sum(p_analytical)
            
            # 检查收敛
            if np.linalg.norm(p_analytical - p_new) < 1e-6:
                break
                
            p_new = p_analytical.copy()
        
        return p_new
    
    def _optimize_eta_given_p(self, p_vector):
        """
        给定p，优化总支付η
        基于Lemma 2的推导：η* = A / (4λρx_n)
        其中 A = Σp_nq_n
        
        :param p_vector: 固定的概率分配
        :return: 优化后的总支付
        """
        # 使用初始eta计算q值
        eta_init = 1.0
        q_current = self.cournot_solver.compute_equilibrium(p_vector, eta_init)
        
        # 计算A = Σp_nq_n
        A = np.sum(p_vector * q_current)
        
        # 获取数据拥有者的数据量 x_n
        x_n = self.q_max_vector
        lambda_rho = self.C  # λρ参数
        
        # 使用公式 (3.18): η* = A / (4λρx_n)
        # 这里需要选择一个代表性的x_n值，或者使用加权平均
        # 使用加权平均: Σ(p_n * x_n) / Σp_n
        weighted_x = np.sum(p_vector * x_n) / np.sum(p_vector) if np.sum(p_vector) > 0 else np.mean(x_n)
        weighted_lambda_rho = np.sum(p_vector * lambda_rho) / np.sum(p_vector) if np.sum(p_vector) > 0 else np.mean(lambda_rho)
        
        if A > 0 and weighted_lambda_rho > 0 and weighted_x > 0:
            eta_analytical = A / (4 * weighted_lambda_rho * weighted_x)
            
            # 确保eta为正数且合理
            eta_analytical = max(eta_analytical, 1e-6)
        else:
            # 如果解析解不可行，使用数值优化作为后备
            eta_analytical = self._optimize_eta_numerically(p_vector)
        
        return eta_analytical
    
    def _optimize_eta_numerically(self, p_vector):
        """
        数值优化eta的后备方法
        """
        def objective_function(eta_array):
            eta = eta_array[0]
            
            if eta <= 1e-6:
                return 1e10
                 
            # 计算数据拥有者的响应
            q_star = self.cournot_solver.compute_equilibrium(p_vector, eta)
             
            # 计算模型拥有者效用（负值用于最小化）
            utility = np.sum(p_vector * q_star) - eta
            return -utility
        
        # 设置边界和初始值
        bounds = [(1e-6, None)]
        initial_eta = np.array([1.0])
        
        # 执行优化
        result = minimize(
            objective_function,
            initial_eta,
            method='SLSQP',
            bounds=bounds,
            options={'disp': False, 'maxiter': 200}
        )
        
        return result.x[0]
        
    def optimize_with_fixed_p(self, p_fixed):
        """
        在固定概率分配p的情况下，只优化模型拥有者的总支付参数eta
        
        :param p_fixed: 固定的概率分配向量 (p_1, ..., p_N)
        :return: 元组 (p_fixed, eta_optimal, q_optimal, leader_utility)
                p_fixed: 输入的固定概率分配
                eta_optimal: 最优总支付
                q_optimal: 对应的数据拥有者最优质量选择
                leader_utility: 领导者最大效用
        """
        
        # 定义只优化eta的目标函数
        def objective_function_eta(eta_array):
            eta = eta_array[0]  # 将一维数组转为标量
            
            # 严格检查eta是否为正数
            if eta <= 1e-6:  # 使用一个小的正数作为阈值
                return 1e10  # 返回一个很大的惩罚值
            
            # 计算对应于(p_fixed,eta)的数据拥有者最优响应q
            q_star_vector = self.cournot_solver.compute_equilibrium(p_fixed, eta)
            
            # 计算模型拥有者效用
            utility = np.sum(p_fixed * q_star_vector) - eta
            
            # 因为minimize函数是最小化目标，所以返回负的效用
            return -utility
        
        # eta的边界
        bounds = [(0, None)]  # eta的边界
        
        # 初始猜测
        initial_eta = 1.0  # 初始支付
        initial_guess = np.array([initial_eta])
        
        # 使用SLSQP方法求解约束优化问题
        UtilT.print_and_log("开始模型拥有者的eta优化(固定p)...")
        result = minimize(
            objective_function_eta,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            options={'disp': True, 'maxiter': 500}
        )
        
        if result.success:
            UtilT.print_and_log("模型拥有者的eta优化成功!")
        else:
            UtilT.print_and_log(f"警告: eta优化未成功收敛。原因: {result.message}")
        
        # 提取结果
        eta_optimal = result.x[0]
        
        # 计算对应的数据拥有者质量选择
        q_optimal = self.cournot_solver.compute_equilibrium(p_fixed, eta_optimal)
        
        # 模型拥有者最大效用
        leader_utility = -result.fun
        
        UtilT.print_and_log(f"固定p: {p_fixed}")
        UtilT.print_and_log(f"最优eta: {eta_optimal:.4f}")
        UtilT.print_and_log(f"对应的q: {q_optimal}")
        UtilT.print_and_log(f"模型拥有者最大效用: {leader_utility:.4f}")
        
        return p_fixed, eta_optimal, q_optimal, leader_utility