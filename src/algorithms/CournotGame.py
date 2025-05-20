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

    def compute_equilibrium(self, p_vector, eta, max_iter=1000, epsilon=1e-6, random_init=False):
        """
        计算给定p和eta下的古诺均衡，遵循伪代码中的迭代收敛算法。

        :param p_vector: 领导者选择的概率/比例向量 (p1, ..., pN)
        :param eta: 领导者选择的总支付
        :param max_iter: 最大迭代次数
        :param epsilon: 收敛阈值 (L1范数)
        :param random_init: 是否随机初始化q向量 (True/False)
        :return: 数据拥有者在该(p, eta)下的近似古诺均衡质量向量
        """
        p_vector = np.array(p_vector)
        if p_vector.shape[0] != self.N:
            raise ValueError("概率向量p_vector的长度必须等于数据拥有者数量N")
        if np.any(p_vector < 0) or not np.isclose(np.sum(p_vector), 1.0, atol=1e-5):
            # 允许p_vector不严格加到1，例如代表的是权重而非严格概率，但一般应非负
            if np.any(p_vector < 0):
                MNISTUtil.print_and_log(f"警告: p_vector {p_vector} 包含负值。")
            if not np.isclose(np.sum(p_vector), 1.0, atol=1e-5) and self.N > 0:  # 仅当N>0时检查和
                MNISTUtil.print_and_log(f"警告: p_vector {p_vector} 的和 {np.sum(p_vector):.4f} 不接近1.0。")

        if self.N == 0:
            MNISTUtil.print_and_log("信息: 数据拥有者数量为0，返回空质量向量。")
            return np.array([])

        # 基本条件检查 (N=1的情况单独处理或视为非博弈)
        if eta <= 0:  # 成本可能为0，但总支付为0或负，一般无动力提供质量
            MNISTUtil.print_and_log(f"警告: 总支付eta ({eta}) 非正，可能导致全零质量向量。")
            # 如果eta非正，且成本为正，那么最优q通常为0。
            # 我们允许算法继续，因为效用函数和求解器应能处理此情况。

        # 1. Initialize q^(0) (质量向量的初始猜测)
        if random_init:
            # 随机初始化在 [0, q_max_vector_n] 之间
            q_current = np.random.rand(self.N) * self.q_max_vector
        else:
            q_current = np.zeros(self.N)  # 初始化为0，如伪代码 l=0, q_n^0 = 0

        q_previous = np.copy(q_current)  # 用于存储 q^(l-1)

        MNISTUtil.print_and_log(f"开始古诺博弈迭代。N={self.N}, eta={eta}, epsilon={epsilon}, max_iter={max_iter}")
        MNISTUtil.print_and_log(f"初始q: {q_current}")

        for l_iter in range(max_iter):  # l 是轮次，对应伪代码中的 l
            # MNISTUtil.print_and_log(f"--- Iteration {l_iter + 1} ---")
            q_iteration_start = np.copy(q_current)  # 保存本轮迭代开始时的q，用于收敛判断

            # 对应伪代码第6行: "for each data owner n ∈ N do"
            for n_owner_idx in range(self.N):
                # 准备 q_{-n}^{l-1} (其他数据拥有者在上一有效轮次的质量)
                # 注意：在顺序更新（Gauss-Seidel-like）中，当计算owner n时，
                # owner 0 to n-1 已经是当前轮次 l 的值了。
                # 伪代码 q_{-n}^{l-1} 暗示使用上一轮完成时的值 (Jacobi-like)。
                # 我们将严格按照伪代码，使用 q_previous (即 q^(l-1) ) 来获取 q_{-n}。

                # 如果使用Jacobi迭代 (所有q_n在本轮都基于上一轮的q_vector):
                q_minus_n_previous_round = np.delete(q_previous, n_owner_idx)

                # 7. Solve q_n^l_candidate <- argmax u_n(p, q_n, q_{-n}^{l-1}, eta)
                q_n_candidate_unconstrained = self.solve_optimal_q_for_owner(
                    n_owner_idx,
                    p_vector,
                    q_minus_n_previous_round,  # 使用上一完整轮次的q_{-n}
                    eta
                )

                # 应用边界约束 [0, q_max_vector[n]]
                q_n_candidate = max(0, min(q_n_candidate_unconstrained, self.q_max_vector[n_owner_idx]))

                # 8. if u_n(p, q_n^l_candidate, q_{-n}^{l-1}, eta) > u_n(p, q_n^{l-1}, q_{-n}^{l-1}, eta)
                # 9. then q_n^l <- q_n^l_candidate
                # 10. else q_n^l <- q_n^{l-1}

                # 计算采用新候选q时的效用
                utility_with_new_q_n = self.calculate_utility_n(
                    n_owner_idx,
                    p_vector,
                    q_n_candidate,
                    q_minus_n_previous_round,
                    eta
                )

                # 计算采用旧q时的效用 (q_n^{l-1} 即 q_previous[n_owner_idx])
                utility_with_old_q_n = self.calculate_utility_n(
                    n_owner_idx,
                    p_vector,
                    q_previous[n_owner_idx],  # 这是 q_n^{l-1}
                    q_minus_n_previous_round,
                    eta
                )

                if utility_with_new_q_n > utility_with_old_q_n:
                    q_current[n_owner_idx] = q_n_candidate
                else:
                    q_current[n_owner_idx] = q_previous[n_owner_idx]  # 保持上一轮的值 for n_owner_idx

            # 5. Check convergence: sum(|q_n^l - q_n^{l-1}|) <= epsilon
            # q_iteration_start 是本轮开始时的q (即上一轮的q_previous)
            # q_current 是本轮计算完成后的q
            # 所以我们比较 q_current 和 q_previous (因为 q_previous 在循环开始时被设为上一轮的 q_current)
            abs_diff_sum = np.sum(np.abs(q_current - q_previous))

            # MNISTUtil.print_and_log(f"Iter {l_iter+1}: q_previous={q_previous}, q_current={q_current}, diff_sum={abs_diff_sum:.2e}")

            if abs_diff_sum <= epsilon:
                MNISTUtil.print_and_log(f"在第 {l_iter + 1} 轮达到收敛。Sum(|q^l - q^(l-1)|) = {abs_diff_sum:.2e}")
                break

            # 更新 q_previous 以供下一轮迭代使用
            q_previous = np.copy(q_current)

            if l_iter == max_iter - 1:
                MNISTUtil.print_and_log(f"达到最大迭代次数 {max_iter}，可能未完全收敛。")
                MNISTUtil.print_and_log(f"最后一轮改变量之和 (Sum(|q^l - q^(l-1)|)): {abs_diff_sum:.2e}")

        # 最终检查解的有效性 (可选，但有助于理解结果)
        S_actual = np.sum(p_vector * q_current)
        # MNISTUtil.print_and_log(f"最终均衡质量向量 q_star = {q_current}")
        # MNISTUtil.print_and_log(f"最终实际加权总质量 S_actual = {S_actual:.4f}")

        return q_current
