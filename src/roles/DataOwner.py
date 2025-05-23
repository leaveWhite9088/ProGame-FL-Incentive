class DataOwner:
    def __init__(self, Lambda, Rho):
        """
        :param Lambda: 市场调节因子
        :param Rho: DataOwner支付的单位数据的训练价格
        """
        self.Lambda = Lambda
        self.Rho = Rho
        self.imgData = []  # 保存多个图像数据
        self.labelData = []  # 保存多个标签
        self.originalData = []  # 保存多个原始图像数据（用于质量评价）

    # 效用函数
    def cal_utility(self, Eta, qn, sum_pn_qn):
        """
        计算DataOwner的效用函数
        :param Eta: ModelOwner的总支付
        :param qn: 当前DataOwner的模型贡献（qn = xn * fn）
        :param sum_pn_qn: 所有DataOwner的总的模型贡献（加权）
        :param xn: 当前DataOwner提供的数据量
        :return: 当前DataOwner的效用
        """
        return qn * (Eta / sum_pn_qn - self.Lambda * self.Rho)