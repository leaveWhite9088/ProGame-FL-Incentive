import numpy as np
from src.utils.UtilMNIST import MNISTUtil

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

    def compute_equilibrium(self, p_vector, eta, max_iterations=1000, tolerance=0.01):
        """
        计算给定p和eta下的古诺均衡，通过迭代达到收敛。

        :param p_vector: 领导者选择的概率/比例向量 (p1, ..., pN)
        :param eta: 领导者选择的总支付
        :param max_iterations: 最大迭代次数
        :param tolerance: 收敛的相对差异容忍度
        :return: 数据拥有者在该(p, eta)下的近似古诺均衡质量向量
        """
        p_vector = np.array(p_vector)

        # 基本条件检查
        if self.N <= 0 or eta <= 0 or np.any(self.C <= 0) or not p_vector.any() or p_vector.shape[0] != self.N:
            MNISTUtil.print_and_log("警告: 无效的参数输入 compute_equilibrium，返回全零质量向量")
            return np.zeros(self.N if self.N > 0 else 1)  # 确保在N=0时不会出错

        # 处理 N=1 (垄断) 的情况
        if self.N == 1:
            MNISTUtil.print_and_log("compute_equilibrium: N=1，处理垄断情况。")
            q_star_vector_monopoly = np.zeros(1)  # 初始化为1个元素的向量

            # 在古诺模型的多参与者推导中，关键项 (N-1) 会在 N=1 时变为 0。
            # term_factor = (self.N - 1) / np.sum(self.C)  => (1-1) / C[0] = 0
            # 这将导致 q_analytic = 0。
            # 这意味着如果直接套用为N>1推导的古诺均衡公式，垄断者的产量将是0。
            # 这通常不符合垄断者的实际行为（垄断者会根据市场需求和成本决定产量以最大化利润）。
            # 然而，如果任务是严格按照所给公式的数学结果来执行，那么结果就是0。

            # 假设我们严格遵循公式形式：
            q_analytic_monopoly = 0.0

            # 遵循原公式在N=1时的数学结果。
            q_star_vector_monopoly[0] = max(0, min(q_analytic_monopoly, self.q_max_vector[0]))

            MNISTUtil.print_and_log(f"N=1: 基于古诺公式的直接数学应用，计算得到 q_1 = {q_star_vector_monopoly[0]}")
            return q_star_vector_monopoly

        # --- 古诺博弈迭代过程 (N > 1) ---
        q_current = np.zeros(self.N)  # 初始化当前质量向量

        # 启发式初始值 (基于之前单次计算的逻辑，适用于 N > 1)
        Sum_C_initial = np.sum(self.C)
        if Sum_C_initial > 0:  # 避免除以零
            term_factor_initial = (self.N - 1) / Sum_C_initial  # N > 1, 所以 N-1 >= 1
            for n_init in range(self.N):
                if p_vector[n_init] > 1e-9:  # 避免p_n为零时的除法错误
                    term_in_parenthesis_init = 1 - (self.C[n_init] * term_factor_initial)
                    numerator_init = eta * term_factor_initial
                    denominator_init = p_vector[n_init]
                    q_analytic_init = (numerator_init / denominator_init) * term_in_parenthesis_init
                    q_current[n_init] = max(0, min(q_analytic_init, self.q_max_vector[n_init]))
        else:  # Sum_C_initial is 0 or negative, which is an invalid input caught earlier.
            # This block is defensive.
            MNISTUtil.print_and_log("警告: Sum_C_initial 非正，无法计算启发式初始值，从零开始。")

        for iteration in range(max_iterations):
            q_previous = q_current.copy()  # 保存上一轮的q

            # 在N>1的情况下，Sum_C应该是正的 (已在外部检查 self.C[n] > 0)
            Sum_C = np.sum(self.C)
            # term_factor的计算对于 N > 1 是安全的
            term_factor = (self.N - 1) / Sum_C

            for n in range(self.N):  # 对每个数据拥有者计算其最优响应（基于解析解形式）
                if p_vector[n] <= 1e-9:  # 避免p_n为零时的除法错误
                    q_analytic_n = 0
                else:
                    term_in_parenthesis = 1 - (self.C[n] * term_factor)
                    numerator = eta * term_factor
                    denominator = p_vector[n]
                    q_analytic_n = (numerator / denominator) * term_in_parenthesis

                q_current[n] = max(0, min(q_analytic_n, self.q_max_vector[n]))

            # 检查收敛条件
            diff_norm = np.linalg.norm(q_current - q_previous)
            q_previous_norm = np.linalg.norm(q_previous)

            # 如果 q_previous_norm 非常小，使用绝对差值进行比较，或者认为已收敛（如果 diff_norm 也小）
            if q_previous_norm < 1e-9:  # 调整一个更小的阈值来判断是否“接近零”
                if diff_norm < tolerance:  # 可以用一个绝对小的量，或者 tolerance 本身如果够小
                    MNISTUtil.print_and_log(
                        f"收敛达成 (迭代 {iteration + 1}): q_previous 接近零 (范数: {q_previous_norm:.2e}), 差异 ({diff_norm:.2e}) 小于或接近容忍度。")
                    break
            elif (diff_norm / q_previous_norm) < tolerance:
                MNISTUtil.print_and_log(f"收敛达成 (迭代 {iteration + 1})。相对差异: {diff_norm / q_previous_norm:.2e}")
                break

            if iteration == max_iterations - 1:
                MNISTUtil.print_and_log(f"警告: 达到最大迭代次数 {max_iterations}，可能未完全收敛。")
                current_relative_diff = diff_norm / q_previous_norm if q_previous_norm > 1e-9 else float('inf')
                MNISTUtil.print_and_log(f"最后相对差异: {current_relative_diff:.2e} (绝对差异: {diff_norm:.2e})")

        return q_current