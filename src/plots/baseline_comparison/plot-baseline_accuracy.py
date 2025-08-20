import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# 设置全局字体为 Times New Roman
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 16  # 同时设置字体大小

# 定义轮次
users = range(10)

utility_com = [0.3702, 0.396, 0.4156, 0.4301, 0.4471, 0.4547, 0.4671, 0.4738, 0.4826, 0.4956]

utility_pgirdfl = [0.4151, 0.4495, 0.5018, 0.5587, 0.6127, 0.6877, 0.7244, 0.739, 0.7484, 0.7557]

# utility_random = [0.004879251387796715, -0.015869673572045346, -0.021546778375379456, -0.007017844536257381, -0.0042825610874949035, -0.01372284916503349, -0.009729892916789858, 0.0010218434713672234, 0.0004126470571156018, -0.005948582491510149]

# 绘制图表
plt.plot(users, utility_com, 'r--o', label='Comparison')  # 红色虚线，圆形标记
plt.plot(users, utility_pgirdfl, 'b--s', label='PGI-RDFL')  # 蓝色虚线，方形标记
# plt.plot(users, utility_random, 'g--^', label='Random')  # 绿色虚线，三角形标记

# 添加图例
plt.legend()

plt.tick_params(axis='x', labelsize=14)
plt.tick_params(axis='y', labelsize=14)

# 设置标题和坐标轴标签
plt.xlabel(r'$Round$')
plt.ylabel(r'$Accuracy$')

# 强制显示完全
plt.tight_layout()

# 显示网格
plt.grid(True)

# 显示图表
plt.show()
