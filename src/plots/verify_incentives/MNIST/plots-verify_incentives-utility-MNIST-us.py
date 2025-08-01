import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# 设置全局字体为 Times New Roman
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 16  # 同时设置字体大小

# 定义用户数量
users = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

utility_fix = [-0.43272913989456985, -0.29367006979017063, -0.37351225760031237, -0.35685902438551975, -0.2517748477725026, -0.36757709096857794, -0.3335211095146168, -0.3746236938535923, -0.3314071802735242, -0.3319729025817192]

utility_pgirdfl = [0.022722054398154473, 0.9989192193490499, 0.17959343676970846, 0.019965882648864675, 0.7996462224867379, 1.1083199388651725, 0.6855373946604832, 1.2936864712659695, 1.176564688778547, 1.1336146097343223]

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