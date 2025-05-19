import math

class ModelOwner:
    def __init__(self, model):
        """
        :param model: 模型
        """
        self.model = model

    # 效用公式
    def cal_utility(self, sum_pn_fn_xn, Eta):
        """
        计算ModelOwner的效用
        :param sum_pn_fn_xn: 所有DataOwner的总的模型贡献（加权）
        :param Eta: ModelOwner的总支付
        :return: ModelOwner的效用
        """
        # 默认模型评价函数

        return math.log(sum_pn_fn_xn) - Eta