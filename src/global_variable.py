"""全局变量"""

"""================================= 超参数 ================================="""

Lambda = 0.90
Rho = 1
Alpha = 5
Epsilon = 1

"""================================= 其他参数 ================================="""

# 这里通过命令行参数修改值
import argparse

# 创建 ArgumentParser 对象
parser = argparse.ArgumentParser()

# 添加命令行参数
parser.add_argument('--adjustment_literation', type=float, default=0.01, help="adjustment_literation")
parser.add_argument('--parent_path', type=str, default="log-main", help="parent_path")

# 解析命令行参数
args = parser.parse_args()

# 调整轮次
adjustment_literation = args.adjustment_literation
# 路径参数 可选：log-parameter_analysis,log-comparison,log-ablation,log-supplement
parent_path = args.parent_path