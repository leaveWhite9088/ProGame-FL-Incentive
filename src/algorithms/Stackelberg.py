import numpy as np
from src.utils.MNISTUtil import MNISTUtil
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
        MNISTUtil.print_and_log("开始求解Stackelberg博弈...")
        
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
        MNISTUtil.print_and_log("Stackelberg博弈求解完成，以下为结果")
        MNISTUtil.print_and_log(f"均衡p: {p_star}")
        MNISTUtil.print_and_log(f"均衡eta: {eta_star:.4f}")
        MNISTUtil.print_and_log(f"均衡q: {q_star}")
        MNISTUtil.print_and_log(f"领导者效用: {leader_utility:.4f}")
        MNISTUtil.print_and_log(f"跟随者总效用: {total_follower_utility:.4f}")
        
        # 可以进一步分析各数据拥有者的个体效用
        # for n in range(self.N):
        #     MNISTUtil.print_and_log(f"数据拥有者{n+1}效用: {follower_utilities[n]:.4f}")
        
        return p_star, eta_star, q_star, leader_utility, follower_utilities