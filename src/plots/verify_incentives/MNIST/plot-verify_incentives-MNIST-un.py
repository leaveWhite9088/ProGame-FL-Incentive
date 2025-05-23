import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# 设置全局字体为 Times New Roman
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 16  # 同时设置字体大小

# 定义用户数量
users = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

utility_fix = [0.020006568798037267, 0.006726424183708014, 0.00434226971576773, 0.003412426779752239, 0.0028677080471365326, 0.002203722492724228, 0.001958898443364741, 0.0016102484727703343, 0.0014829607370258546, 0.0013320738582377333]

utility_pgirdfl = [0.008181314580138663, 0.0022729971542812914, 0.0010003878892096663, 0.0029613567773557033, 0.002537415954562668, 0.00024465394428587075, 0.0015760796949463704, 0.00013799041595514758, 0.0012887014364351293, 8.753916457320805e-05]

utility_random = [0.004879251387796715, -0.015869673572045346, -0.021546778375379456, -0.007017844536257381, -0.0042825610874949035, -0.01372284916503349, -0.009729892916789858, 0.0010218434713672234, 0.0004126470571156018, -0.005948582491510149]

# 绘制图表
plt.plot(users, utility_pgirdfl, 'b--s', label='PGI-RDFL')  # 蓝色虚线，方形标记
plt.plot(users, utility_random, 'g--^', label='Random')  # 绿色虚线，三角形标记
plt.plot(users, utility_fix, 'r--o', label='Fixed')  # 红色虚线，圆形标记

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
