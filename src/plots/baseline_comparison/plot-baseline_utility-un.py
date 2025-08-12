import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# 设置全局字体为 Times New Roman
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 16  # 同时设置字体大小

# 定义用户数量
users = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

utility_com = [0.11727782369295085, 0.11667949560130161, 0.07255831921045175, 0.09940635898754797, 0.09118380996705401, 0.07282785604777216, 0.08706099188965286, 0.0684412985471118, 0.08293199123103015, 0.06637327170098657]

utility_pgirdfl = [0.023221536009889236, 0.06548709219964488, 0.01631740881482431, 0.027097818455455412, 0.018372144322269912, 0.9520343559186711, 0.04059446509117943, 0.01769942832152078, 0.05477536326142063, 0.08150775019566657]

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
plt.xlabel(r'$N$')
plt.ylabel(r'$U_n$')

# 强制显示完全
plt.tight_layout()

# 显示网格
plt.grid(True)

# 显示图表
plt.show()
