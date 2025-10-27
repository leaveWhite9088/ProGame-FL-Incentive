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

utility_fixed = [-0.4219679141322362, -0.35068594714542034, -0.4221886113760944, -0.4580141347782354, -0.3154631874746714, -0.292884691831656, -0.2920110084891073, -0.3627606771781968, -0.295922401084608, -0.3699238415335008]

utility_random = [-0.027526666686899293, -0.12873779720507283, -0.019055003179785673, -0.11471852497663415, 0.007435524791823786, 5.0582845224189654e-05, -0.010814807826947104, -0.038597270590625565, -0.007003308566739308, -0.15128096819138126]

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