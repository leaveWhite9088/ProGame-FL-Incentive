import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# 设置全局字体为 Times New Roman
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 16  # 同时设置字体大小

# 定义轮次
users = range(100)

utility_baseline =
utility_pgirdfl =
utility_fixed =
utility_random =
utility_FL_Client_Sampling =

# 绘制数据线
plt.plot(users, utility_baseline, 'r--o', markersize=5, linewidth=2, label='Baseline')
plt.plot(users, utility_pgirdfl, 'b--s', markersize=5, linewidth=2, label='PGI-RDFL')
plt.plot(users, utility_fixed, 'y--D', markersize=5, linewidth=2, label='Fixed')
plt.plot(users, utility_random, 'g--^', markersize=5, linewidth=2, label='Random')
plt.plot(users, utility_FL_Client_Sampling, 'm--*', markersize=5, linewidth=2, label='Client-Sampling')

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
