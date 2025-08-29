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
utility_fair_flearn =
utility_FL_Client_Sampling =
utility_Oort =

# 绘制图表
plt.plot(users, utility_baseline, 'r--o', label='Baseline')  # 红色虚线，圆形标记
plt.plot(users, utility_pgirdfl, 'b--s', label='PGI-RDFL')  # 蓝色虚线，方形标记
plt.plot(users, utility_fair_flearn, 'y--o', label='Fair-FLearn')
plt.plot(users, utility_FL_Client_Sampling, 'g--^', label='Client-Sampling')
plt.plot(users, utility_Oort, 'g--^', label='Oort')

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
