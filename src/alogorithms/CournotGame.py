import numpy as np
from src.utils.MNISTUtil import MNISTUtil

class CournotGame:
    """
    实现数据拥有者之间的古诺均衡博弈
    
    在给定模型拥有者选择的概率向量p和总支付η的情况下，
    计算数据拥有者的最优质量选择q
    """
    
    def __init__(self, N, C, q_max_vector):
        """
        初始化古诺均衡计算器
        
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
        
        # 检查输入有效性
        if N <= 1:
            MNISTUtil.print_and_log("警告: 数据拥有者数量必须大于1才能形成有效的古诺博弈")
        
        if np.any(self.C <= 0):
            MNISTUtil.print_and_log("警告: 成本C必须为正值")
            
        if np.any(self.q_max_vector < 0):
            MNISTUtil.print_and_log("警告: 最大质量必须非负")
    
    def compute_equilibrium(self, p_vector, eta):
        """
        计算给定p和eta下的古诺均衡
        
        :param p_vector: 领导者选择的概率/比例向量 (p1, ..., pN)
        :param eta: 领导者选择的总支付
        :return: 数据拥有者在该(p, eta)下的近似古诺均衡质量向量
        """
        # 将输入转换为numpy数组
        p_vector = np.array(p_vector)
        
        # 初始化结果向量
        q_star_vector = np.zeros(self.N)
        
        # 检查基本条件
        if self.N <= 1 or eta <= 0 or np.sum(self.C) <= 0:
            MNISTUtil.print_and_log("警告: 无效的参数，返回全零质量向量")
            return q_star_vector
        
        # 计算解析解中需要的项
        Sum_C = np.sum(self.C)
        term_factor = (self.N - 1) / Sum_C
        
        # 对每个数据拥有者计算质量
        for n in range(self.N):
            # 避免p_n为零时的除法错误
            if p_vector[n] <= 1e-9:
                q_analytic = 0
            else:
                # 计算解析解
                term_in_parenthesis = 1 - (self.C[n] * term_factor)
                numerator = eta * term_factor
                denominator = p_vector[n]
                
                q_analytic = (numerator / denominator) * term_in_parenthesis
            
            # 应用边界约束
            q_star_vector[n] = max(0, min(q_analytic, self.q_max_vector[n]))
        
        # 检查解的有效性
        S_actual = np.sum(p_vector * q_star_vector)
        S_analytic = eta * (self.N - 1) / Sum_C
        
        if abs(S_actual - S_analytic) > 0.1 * S_analytic and S_analytic > 0:
            MNISTUtil.print_and_log(f"注意: 边界约束显著影响了均衡解。实际S={S_actual:.4f}，理论S={S_analytic:.4f}")
            
        return q_star_vector
