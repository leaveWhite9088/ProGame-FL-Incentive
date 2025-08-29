import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# 设置全局字体为 Times New Roman
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 16  # 同时设置字体大小

# 定义用户数量
users = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

utility_com = [-0.7864095866114917, -1.7303862207515408, -0.8006686215158911, -2.4103292599957484, -2.442369378153945, -1.6551291629955713, -3.0102631175697354, -1.8524520987818605, -3.3699795662469185, -1.829754678791197]

utility_pgirdfl = [0.047310270766802456, -0.08986110898419913, 0.024323110152766274, -0.042052599828645496, -0.027173932798036343, -0.27254956500243177, -0.17642256818968516, -0.1316295870079443, -0.20191539042056933, -0.25288294028806674]

# utility_random = [-0.1992243020800306, 0.12682024188139285, 0.48920543384122506, 0.08697339456316255, 0.07611981271646961, 0.6301191648713914, 0.48644798048881177, -0.2759086026749037, -0.23020039860317965, 0.40061759209813097]

# 绘制图表
plt.plot(users, utility_com, 'r--o', label='Comparison')  # 红色虚线，圆形标记
plt.plot(users, utility_pgirdfl, 'b--s', label='PGI-RDFL')  # 蓝色虚线，方形标记
# plt.plot(users, utility_random, 'g--^', label='Random')  # 绿色虚线，三角形标记

# 添加图例
plt.legend()

plt.tick_params(axis='x', labelsize=14)
plt.tick_params(axis='y', labelsize=14)

# 设置标题和坐标轴标签
plt.xlabel(r'$N$')
plt.ylabel(r'$U_s$')

# 强制显示完全
plt.tight_layout()

# 显示网格
plt.grid(True)

# 显示图表
plt.show()