import numpy as np
import random
from src.utils import UtilT
from src.algorithms.JointOptimization import JointOptimization

class Stackelberg:
    """
    实现联邦学习中的Stackelberg博弈算法 (Algorithm 1: PGI-RDFL Algorithm)
    
    实现资源解耦联邦学习中的嵌套博弈模型(Nested Game Model)：
    - 模型拥有者(Model Owner)作为领导者(Leader)
    - 数据拥有者(Data Owners)作为次级领导者(Sub-leaders)
    - 计算中心(Computing Centers)作为跟随者(Followers)
    
    博弈求解采用逆向归纳法(Backward Induction)
    """
    
    def __init__(self, N, C, q_max_vector, lambda_param=0.9):
        """
        初始化Stackelberg博弈
        
        :param N: 数据拥有者数量
        :param C: 每个数据拥有者的单位成本（λρ）
        :param q_max_vector: 每个数据拥有者的最大质量向量 (|X_1|, ..., |X_N|)
        :param lambda_param: 市场调节因子λ（默认0.9）
        """
        self.N = N
        self.lambda_param = lambda_param
        
        # 处理C可能是标量或向量的情况
        if isinstance(C, (int, float)):
            self.C = np.ones(N) * C
        else:
            self.C = np.array(C)
            
        self.q_max_vector = np.array(q_max_vector)
        
        # 创建联合优化求解器
        self.joint_optimizer = JointOptimization(N, C, q_max_vector, lambda_param)
    
    def solve(self):
        """
        求解完整的Stackelberg博弈
        实现Algorithm 1的核心步骤
        
        通过调用Algorithm 3 (JointOptimization)完成模型拥有者的参数联合优化，
        其中已经嵌入了Algorithm 2 (Cournot Game)来计算数据拥有者的最佳响应
        
        :return: 元组 (p_star, eta_star, q_star, leader_utility, follower_utilities, social_welfare)
                p_star: 均衡概率分配P*
                eta_star: 均衡总支付η*
                q_star: 均衡质量选择Q*
                leader_utility: 领导者效用U_s
                follower_utilities: 跟随者效用向量U_n
                social_welfare: 社会福利(所有参与者效用总和)
        """
        UtilT.print_and_log("开始求解Stackelberg博弈...")
        
        # Algorithm 1 Step 2: 调用Algorithm 3(Joint Optimization)计算η*和P*
        # 这已经包含了将数据拥有者响应(通过Algorithm 2)嵌入到领导者优化中的过程
        p_star, eta_star, q_star, leader_utility = self.joint_optimizer.optimize()
        
        # Algorithm 1 Step 3: 调用Algorithm 2(Embedded Cournot Subgame)计算Q*
        # 注：这一步已经在JointOptimization中完成，q_star即为结果
        
        # 计算各数据拥有者(跟随者)在均衡时的效用
        # 根据公式(3.3): U_n = q_n * (η/Σp_nq_n - λρ)
        S_star_actual = np.sum(p_star * q_star)

        # 检查 S* 是否接近于零以避免除零错误
        if S_star_actual > 1e-9:  # 使用一个小的正阈值
            price_at_equilibrium = eta_star / S_star_actual
            # 计算每个跟随者的效用
            follower_utilities = q_star * (price_at_equilibrium - self.C)
        else:
            # 如果 S* 为零或接近零
            follower_utilities = np.zeros(self.N)

        # 计算跟随者总效用
        total_follower_utility = np.sum(follower_utilities)
        
        # 输出完整的均衡结果
        UtilT.print_and_log("Stackelberg博弈求解完成，以下为结果")
        UtilT.print_and_log(f"均衡p (P*): {p_star}")
        UtilT.print_and_log(f"均衡eta (η*): {eta_star:.4f}")
        UtilT.print_and_log(f"均衡q (Q*): {q_star}")
        UtilT.print_and_log(f"领导者效用 (U_s): {leader_utility:.4f}")
        UtilT.print_and_log(f"跟随者总效用: {total_follower_utility:.4f}")
        
        # 可以进一步分析各数据拥有者的个体效用
        # for n in range(self.N):
        #     UtilT.print_and_log(f"数据拥有者{n+1}效用: {follower_utilities[n]:.4f}")
        
        return p_star, eta_star, q_star, leader_utility, total_follower_utility / self.N
        
    def solve_with_fixed_eta(self, eta_fixed, max_iterations=100, tolerance=1e-6):
        """
        在固定总支付eta的情况下求解追随者(数据拥有者)间的博弈均衡
        用于对比实验中的固定支付策略
        
        :param eta_fixed: 固定的总支付金额
        :param max_iterations: 最大迭代次数
        :param tolerance: 收敛容差
        :return: 元组 (p_star, eta_fixed, q_star, leader_utility, avg_follower_utility)
        """
        UtilT.print_and_log(f"开始求解固定总支付eta={eta_fixed}下的Stackelberg博弈...")
        
        # 在固定eta的情况下，仍然需要优化p
        # 使用均匀分布作为初始p
        p_star = np.ones(self.N) / self.N
        
        # 使用JointOptimization的固定p优化功能，但这里我们固定的是eta
        # 所以需要直接计算数据拥有者的响应
        from src.algorithms.CournotGame import CournotGame
        cournot_solver = CournotGame(self.N, self.C, self.q_max_vector, self.lambda_param)
        q_star = cournot_solver.compute_equilibrium(p_star, eta_fixed)
        
        # 计算各方效用
        S_star_actual = np.sum(p_star * q_star)
        
        # 计算领导者效用 (根据公式(3.2))
        leader_utility = S_star_actual - eta_fixed
        
        # 计算跟随者效用
        if S_star_actual > 1e-9:
            price_at_equilibrium = eta_fixed / S_star_actual
            follower_utilities = q_star * (price_at_equilibrium - self.C)
        else:
            follower_utilities = np.zeros(self.N)
        
        # 计算跟随者平均效用
        avg_follower_utility = np.sum(follower_utilities) / self.N
        
        # 输出结果
        UtilT.print_and_log("固定总支付下的Stackelberg博弈求解完成，以下为结果")
        UtilT.print_and_log(f"均衡p: {p_star}")
        UtilT.print_and_log(f"固定eta: {eta_fixed:.4f}")
        UtilT.print_and_log(f"均衡q: {q_star}")
        UtilT.print_and_log(f"领导者效用: {leader_utility:.4f}")
        UtilT.print_and_log(f"跟随者平均效用: {avg_follower_utility:.4f}")
        
        return p_star, eta_fixed, q_star, leader_utility, avg_follower_utility

    def solve_with_random_eta(self, eta_max, max_iterations=100, tolerance=1e-6):
        """
        在0到eta_max之间随机选择总支付eta的情况下求解追随者(数据拥有者)间的博弈均衡
        用于对比实验中的随机支付策略
        
        :param eta_max: eta的上限值，实际使用的eta将在0到此值之间随机选择
        :param max_iterations: 最大迭代次数
        :param tolerance: 收敛容差
        :return: 元组 (p_star, random_eta, q_star, leader_utility, avg_follower_utility)
        """
        # 在0到eta_max之间随机选择一个eta值
        random_eta = random.uniform(0.1, eta_max)  # 避免选择过小的值
        
        UtilT.print_and_log(f"开始求解随机总支付eta={random_eta:.4f}(上限为{eta_max:.4f})下的Stackelberg博弈...")
        
        # 使用固定eta的求解方法
        return self.solve_with_fixed_eta(random_eta, max_iterations, tolerance)

    def solve_with_fixed_p(self, p_fixed):
        """
        在固定概率分配p的情况下求解Stackelberg博弈
        用于对比实验中的固定选择概率策略
        
        :param p_fixed: 固定的概率分配向量 (p_1, ..., p_N)
        :return: 元组 (p_fixed, eta_star, q_star, leader_utility, avg_follower_utility)
        """
        UtilT.print_and_log("开始求解固定概率分配下的Stackelberg博弈...")
        
        # 调用JointOptimization的固定p优化函数
        p_star, eta_star, q_star, leader_utility = self.joint_optimizer.optimize_with_fixed_p(p_fixed)
        
        # 计算各数据拥有者(跟随者)在均衡时的效用
        S_star_actual = np.sum(p_star * q_star)

        # 检查 S* 是否接近于零以避免除零错误
        if S_star_actual > 1e-9:
            price_at_equilibrium = eta_star / S_star_actual
            follower_utilities = q_star * (price_at_equilibrium - self.C)
        else:
            follower_utilities = np.zeros(self.N)

        # 计算跟随者总效用和平均效用
        total_follower_utility = np.sum(follower_utilities)
        avg_follower_utility = total_follower_utility / self.N
        
        # 输出完整的均衡结果
        UtilT.print_and_log("固定概率分配下的Stackelberg博弈求解完成，以下为结果")
        UtilT.print_and_log(f"固定p: {p_star}")
        UtilT.print_and_log(f"优化得到的eta: {eta_star:.4f}")
        UtilT.print_and_log(f"均衡q: {q_star}")
        UtilT.print_and_log(f"领导者效用: {leader_utility:.4f}")
        UtilT.print_and_log(f"跟随者平均效用: {avg_follower_utility:.4f}")
        
        return p_star, eta_star, q_star, leader_utility, avg_follower_utility
    
    def compute_computing_center_utility(self, m, d_m, sigma_m, epsilon, rho):
        """
        计算计算中心m的效用
        根据公式(3.4): U_m = max λ * (d_m / Σ_{m'} d_{m'}) * Σ_n ρx_n - εσ_m d_m
        
        :param m: 计算中心索引
        :param d_m: 计算中心m承担的数据量
        :param sigma_m: 计算中心m的计算能力
        :param epsilon: 单位训练成本
        :param rho: 单位训练支付
        :return: 计算中心m的效用
        """
        # 这个函数可以在未来扩展中使用
        # 当前实现主要关注模型拥有者和数据拥有者之间的博弈
        pass