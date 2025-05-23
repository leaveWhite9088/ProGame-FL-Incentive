import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# 设置全局字体为 Times New Roman
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 16  # 同时设置字体大小

# 定义用户数量
users = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

utility_fix = [-0.3637306280667436, -0.32614215740362695, -0.32400827914677177, -0.32723423818180464, -0.33116883390682206, -0.32499693391912743, -0.3276045792435279, -0.3230381038313248, -0.3255280916015989, -0.3255152984998666]

utility_pgirdfl = [-1.1102230246251565e-16, 0.050383855005614375, 0.06664214532077806, -0.021141537223899, -0.030347578589489665, 0.08137218586550599, -0.014069300594251488, 0.08495662004345284, -0.019289421962018105, 0.08735881999491613]

utility_random = [-0.1992243020800306, 0.12682024188139285, 0.48920543384122506, 0.08697339456316255, 0.07611981271646961, 0.6301191648713914, 0.48644798048881177, -0.2759086026749037, -0.23020039860317965, 0.40061759209813097]

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
plt.ylabel(r'$U_s$')

# 强制显示完全
plt.tight_layout()

# 显示网格
plt.grid(True)

# 显示图表
plt.show()