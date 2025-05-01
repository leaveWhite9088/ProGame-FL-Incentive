class ComputingCenter:
    def __init__(self, Lambda, Epsilon, SigmaM):
        """
        :param Lambda: 市场调节因子
        :param Epsilon:  ComputingCenter 的单位计算费用
        :param Sigma: 当前 ComputingCenter 的算力因子
        """
        self.Lambda = Lambda
        self.Epsilon = Epsilon
        self.SigmaM = SigmaM
        self.imgData = []
        self.labelData = []

    # 效用函数
    def cal_utility(self, Rho, sum_dm, sum_xn, dm):
        """
        计算 ComputingCenter 的效用
        :param Rho: DataOwner支付的单位数据的训练价格
        :param sum_dm: 所有 ComputingCenter 承接到的数据总量
        :param sum_xn: 所有DataOwner提供的数据总量
        :param dm: 当前 ComputingCenter 承接的数据量
        :return: 当前 ComputingCenter 的效用
        """
        return self.Lambda * (dm / sum_dm) * Rho * sum_xn - self.Epsilon * self.SigmaM * dm