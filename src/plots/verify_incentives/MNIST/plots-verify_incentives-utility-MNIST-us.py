import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# 设置全局字体为 Times New Roman
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 16  # 同时设置字体大小

# 定义用户数量
users = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

utility_fix = [-0.9999917083742962, -0.49164883564978745, -0.49572341352873617, -0.5560653199361498, -0.46761956509661207, -0.5942817969929863, -0.5118183728344223, -0.4906613890736905, -0.4957796398075329, -0.7033162868336486]

utility_pgirdfl = [-3.9641231269456014e-06, -0.2659685478061652, 0.006713550435054699, -0.38774965132097783, 0.011868867352470225, -0.34100737200135844, 0.019030080628385626, -0.3475907935278073, -0.3205128180940232, -0.3960571661915522]

# utility_random = [-0.1992243020800306, 0.12682024188139285, 0.48920543384122506, 0.08697339456316255, 0.07611981271646961, 0.6301191648713914, 0.48644798048881177, -0.2759086026749037, -0.23020039860317965, 0.40061759209813097]

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
plt.ylabel(r'$U_s$')

# 强制显示完全
plt.tight_layout()

# 显示网格
plt.grid(True)

# 显示图表
plt.show()