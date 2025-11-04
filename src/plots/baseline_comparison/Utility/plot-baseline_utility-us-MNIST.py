import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# 设置全局字体为 Times New Roman
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 16  # 同时设置字体大小

# 定义用户数量
users = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

utility_com = [-0.7864095866114917, -1.7303862207515408, -0.8006686215158911, -2.4103292599957484, -2.442369378153945, -1.6551291629955713, -3.0102631175697354, -1.8524520987818605, -3.3699795662469185, -1.829754678791197]

utility_pgirdfl = [-2.077723166928216, -1.9118704993151265, -2.9706976724606893, -2.874846440626209, -2.4846009261867463, -2.8496335240563146, -3.204779985888832, -2.6952512475197175, -3.0433612889071755, -2.417891176158819]

utility_fixed = [-0.4512191508601363, -0.3129169752086942, -0.4527648084305713, -0.3612674963705964, -0.439281700589079, -0.2904745648113334, -0.36066656414312126, -0.29766634427971006, -0.35245942404894937, -0.259997003225775]

utility_random = [-0.07597864419004396, -0.07135656998266732, -0.04628422413759958, 0.0040678370011497655, -0.05434832435598386, -0.06809105181698916, -0.036860982895497885, -0.038447415335757906, -0.06045126354693464, -0.018588422105013158]

# 绘制图表
plt.plot(users, utility_com, 'r--o', label='Comparison')  # 红色虚线，圆形标记
plt.plot(users, utility_pgirdfl, 'b--s', label='PGI-RDFL')  # 蓝色虚线，方形标记
plt.plot(users, utility_fixed, 'y--o', label='Fixed')
plt.plot(users, utility_random, 'g--^', label='Random')  # 绿色虚线，三角形标记

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