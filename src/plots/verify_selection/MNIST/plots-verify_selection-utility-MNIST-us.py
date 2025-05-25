import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# 设置全局字体为 Times New Roman
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 16  # 同时设置字体大小

# 定义用户数量
users = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

non_two_way_selection = [-1.5923702203624058e-06, -6.228416603846305e-06, -3.569837567790542e-07, -8.946296465218415e-06, -6.059118654030849e-06, -6.410268132552255e-07, -3.8451299482520605e-06, -5.73596440414538e-07, -1.4133709269161056e-06, -5.838047504966128e-06]

random_selection = [-2.4160836693824044e-06, -8.671486943058386e-06, -8.74345716511233e-06, -6.9344932945222855e-06, -5.277275204387495e-07, -8.94923171420994e-06, -1.626605993299317e-06, -3.1637751672750925e-06, -9.363071958411975e-07, -1.1873956062385541e-06]

two_way_selection = [-5.260121008914762e-06, 0.0008573809365079454, -0.3912712373115528, -0.4325966963548159, -0.35581515160098987, -0.3451288696582322, -0.2785617320956707, -0.3499750994557783, -0.2959082797519794, 0.026695730018061026]

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
plt.ylabel(r'$U_s$')

# 强制显示完全
plt.tight_layout()

# 显示网格
plt.grid(True)

# 显示图表
plt.show()