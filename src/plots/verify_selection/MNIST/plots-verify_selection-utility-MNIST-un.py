import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# 设置全局字体为 Times New Roman
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 16  # 同时设置字体大小

# 定义用户数量
users = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

non_two_way_selection = [2.2293183085073675e-08, 1.9977314958081403e-08, 5.437775830006455e-10, 7.112825823364063e-09, 2.880411790916204e-09, 2.588762130453809e-10, 1.0946453223121032e-09, 1.0494406887699566e-10, 2.2227371842100082e-10, 6.971315550047934e-10]

random_selection = [2.5771559140078983e-08, 2.3640125118575817e-08, 1.0446728041432903e-08, 5.242812470656972e-09, 2.4727440052982864e-10, 2.8602242891703488e-09, 3.578420656307037e-10, 5.37534303540467e-10, 1.328042413129349e-10, 1.3202229113432184e-10]

two_way_selection = [0.0014761277556647684, 0.0023774902406499066, 0.014801370376349732, 0.012028337985980794, 0.008331191577156012, 0.006803530549252748, 0.0049404939703355105, 0.0051612308528955165, 0.004041455074781069, 0.00023410065733837294]
# 绘制图表
plt.plot(users, non_two_way_selection, 'r--o', label='non_two_way_selection')  # 红色虚线，圆形标记
plt.plot(users, random_selection, 'b--s', label='random_selection')  # 蓝色虚线，方形标记
plt.plot(users, two_way_selection, 'g--^', label='two_way_selection')  # 绿色虚线，三角形标记

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
