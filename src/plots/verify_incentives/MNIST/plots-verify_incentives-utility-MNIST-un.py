import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# 设置全局字体为 Times New Roman
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 16  # 同时设置字体大小

# 定义用户数量
users = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

utility_fix = [0.04894562259051128, 0.018215153140557674, 0.014538701061342706, 0.010529328048674194, 0.006531947259905049, 0.007180323031195337, 0.005716699979473644, 0.005464516555852912, 0.004425182913846352, 0.003987756123235471]

utility_pgirdfl = [0.09116651059008643, 0.03148100271425539, 0.0676704708944355, 0.05089275572074662, 0.025024597553815812, 0.026564285339508286, 0.020882189002919504, 0.039837519785860157, 0.010990843112546174, 0.029959671948118126]

# utility_random = [0.004879251387796715, -0.015869673572045346, -0.021546778375379456, -0.007017844536257381, -0.0042825610874949035, -0.01372284916503349, -0.009729892916789858, 0.0010218434713672234, 0.0004126470571156018, -0.005948582491510149]

# 绘制图表
plt.plot(users, utility_fix, 'r--o', label='Fixed')  # 红色虚线，圆形标记
plt.plot(users, utility_pgirdfl, 'b--s', label='PGI-RDFL')  # 蓝色虚线，方形标记
# plt.plot(users, utility_random, 'g--^', label='Random')  # 绿色虚线，三角形标记

# 添加图例
plt.legend()

plt.tick_params(axis='x', labelsize=14)
plt.tick_params(axis='y', labelsize=14)

# 设置标题和坐标轴标签
plt.xlabel(r'$N$')
plt.ylabel(r'$U_n$')

# 强制显示完全
plt.tight_layout()

# 显示网格
plt.grid(True)

# 显示图表
plt.show()
