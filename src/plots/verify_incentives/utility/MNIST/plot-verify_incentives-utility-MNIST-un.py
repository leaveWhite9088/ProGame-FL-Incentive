import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# 设置全局字体为 Times New Roman
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 16  # 同时设置字体大小

# 定义用户数量
users = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

utility_fix = [0.6332452306002704, 0.019490462373627945, 0.01343576916046671, 0.01474264191313249, 0.00726548009884058, 0.01201551980344884, 0.006164578499337185, 0.004960571014568012, 0.0045023100837055675, 0.0124184960357638]

utility_pgirdfl = [6.442997503025005e-06, 0.016441423113981175, 0.0009839478484086337, 0.010975652323492055, 0.0006481431092099501, 0.006733522276387652, 0.0004714272931258528, 0.005134883576682212, 0.004299672576486055, 0.004543708556751627]

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
