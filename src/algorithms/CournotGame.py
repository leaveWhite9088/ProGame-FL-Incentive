import numpy as np
from src.utils import UtilT

class CournotGame:
    """
    实现数据拥有者之间的古诺均衡博弈 (Algorithm 2: Embedded Cournot Subgame)
    
    在给定模型拥有者选择的概率向量p和总支付η的情况下，
    通过迭代算法计算数据拥有者的Nash均衡质量选择q*
    
    基于论文中的Theorem 3和Theorem 4，该博弈存在唯一的Nash均衡解
    """
    
    def __init__(self, N, C, q_max_vector, lambda_param=None):
        """
        初始化古诺均衡计算器
        
        :param N: 数据拥有者数量
        :param C: 每个数据拥有者的单位成本（λρ，假设相同，或是一个向量C_vector）
        :param q_max_vector: 每个数据拥有者的最大质量向量 (|X_1|, ..., |X_N|)
        :param lambda_param: 市场调节因子λ（可选，用于更精确的计算）
        """
        self.N = N
        
        # 处理C可能是标量或向量的情况
        # C代表论文中的λρ（单位成本）
        if isinstance(C, (int, float)):
            self.C = np.ones(N) * C
        else:
            self.C = np.array(C)
            
        self.q_max_vector = np.array(q_max_vector)
        self.lambda_param = lambda_param  # 存储lambda参数，可用于未来扩展
        
        # 检查输入有效性
        if N <= 1:
            UtilT.print_and_log("警告: 数据拥有者数量必须大于1才能形成有效的古诺博弈")
        
        if np.any(self.C <= 0):
            UtilT.print_and_log("警告: 成本C必须为正值")
            
        if np.any(self.q_max_vector < 0):
            UtilT.print_and_log("警告: 最大质量必须非负")

    def compute_equilibrium(self, p_vector, eta, max_iterations=1000, tolerance=0.01):
        """
        计算给定p和eta下的古诺均衡，通过迭代达到收敛。
        实现Algorithm 2: Embedded Cournot Subgame
        
        根据论文公式(3.3)，数据拥有者n的效用函数为：
        U_n = max q_n * (η/Σp_nq_n - λρ)
        
        :param p_vector: 领导者选择的概率向量 (p_1, ..., p_N)
        :param eta: 领导者选择的总支付
        :param max_iterations: 最大迭代次数
        :param tolerance: 收敛的容忍度
        :return: 数据拥有者在该(p, eta)下的Nash均衡质量向量Q*
        """
        p_vector = np.array(p_vector)

        # 基本条件检查
        if self.N <= 0 or eta <= 0 or np.any(self.C <= 0) or not p_vector.any() or p_vector.shape[0] != self.N:
            UtilT.print_and_log("警告: 无效的参数输入 compute_equilibrium，返回全零质量向量")
            return np.zeros(self.N if self.N > 0 else 1)

        # 处理 N=1 (垄断) 的情况
        if self.N == 1:
            UtilT.print_and_log("compute_equilibrium: N=1，处理垄断情况。")
            q_star_vector_monopoly = np.zeros(1)
            # 在N=1时，按照公式会得到0，这符合理论推导
            q_star_vector_monopoly[0] = 0.0
            UtilT.print_and_log(f"N=1: 基于古诺公式的直接数学应用，计算得到 q_1 = {q_star_vector_monopoly[0]}")
            return q_star_vector_monopoly

        # --- Algorithm 2: Embedded Cournot Subgame ---
        # Step 1: 初始化 x^(0) = (x_n^(0))_{n=0}^N, 设置迭代轮数 r = 0
        q_current = np.zeros(self.N)  # 初始化当前质量向量Q*
        r = 0  # 迭代轮数
        
        # 使用解析解作为初始值（提高收敛速度）
        # 基于论文中的解析解推导
        Sum_C = np.sum(self.C)
        if Sum_C > 0:  # 避免除以零
            term_factor = (self.N - 1) / Sum_C
            for n_init in range(self.N):
                if p_vector[n_init] > 1e-9:  # 避免p_n为零时的除法错误
                    term_in_parenthesis = 1 - (self.C[n_init] * term_factor)
                    numerator = eta * term_factor
                    denominator = p_vector[n_init]
                    q_analytic_init = (numerator / denominator) * term_in_parenthesis
                    q_current[n_init] = max(0, min(q_analytic_init, self.q_max_vector[n_init]))
        else:
            UtilT.print_and_log("警告: Sum_C 非正，无法计算启发式初始值，从零开始。")

        # Algorithm 2 主循环
        for iteration in range(max_iterations):
            q_previous = q_current.copy()  # 保存上一轮的q
            
            # Step 2-4: 对每个数据拥有者n，更新 x_n^(r+1) = argmax U_n(x_n, x_{-n}^(r))
            for n in range(self.N):
                # 根据理论推导，数据拥有者n的最优响应具有解析解形式
                # 这里使用了论文中推导的解析解来实现argmax操作
                if p_vector[n] <= 1e-9:  # 避免p_n为零时的除法错误
                    q_analytic_n = 0
                else:
                    # 基于公式推导的解析解
                    Sum_C = np.sum(self.C)
                    term_factor = (self.N - 1) / Sum_C
                    term_in_parenthesis = 1 - (self.C[n] * term_factor)
                    numerator = eta * term_factor
                    denominator = p_vector[n]
                    q_analytic_n = (numerator / denominator) * term_in_parenthesis

                # 应用边界约束：0 ≤ q_n ≤ |X_n|
                q_current[n] = max(0, min(q_analytic_n, self.q_max_vector[n]))
            
            # Step 1: 检查终止条件
            diff_norm = np.linalg.norm(q_current - q_previous)
            if diff_norm < tolerance:
                # UtilT.print_and_log(f"Cournot博弈收敛 (迭代 {iteration + 1})")
                break
            
            # Step 5: t = t + 1 (通过for循环自动实现)
            r += 1
            
            if iteration == max_iterations - 1:
                UtilT.print_and_log(f"警告: Cournot博弈达到最大迭代次数 {max_iterations}，可能未完全收敛。")
                UtilT.print_and_log(f"最终差异范数: {diff_norm:.2e}")

        return q_current
    
    def compute_data_owner_utility(self, n, q_n, q_others, p_vector, eta):
        """
        计算数据拥有者n的效用
        根据公式(3.3): U_n = q_n * (η/Σp_iq_i - λρ)
        
        :param n: 数据拥有者索引
        :param q_n: 数据拥有者n的质量贡献
        :param q_others: 其他数据拥有者的质量贡献向量
        :param p_vector: 概率向量
        :param eta: 总支付
        :return: 数据拥有者n的效用
        """
        # 计算Σp_iq_i
        sum_pq = p_vector[n] * q_n + np.sum(p_vector[:n] * q_others[:n]) + np.sum(p_vector[n+1:] * q_others[n:])
        
        if sum_pq > 1e-9:  # 避免除零
            price = eta / sum_pq
            utility = q_n * (price - self.C[n])
        else:
            utility = 0.0
            
        return utility