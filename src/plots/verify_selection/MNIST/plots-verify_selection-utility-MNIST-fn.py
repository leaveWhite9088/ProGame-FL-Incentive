import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# 设置全局字体为 Times New Roman
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 16  # 同时设置字体大小

# 定义用户数量
users = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

non_two_way_selection = [0.7673974152039331, 0.697767943531419, 0.739871560408493, 0.773194600760119, 0.8008022818251861, 0.8167405437397406, 0.8008549649379036, 0.7795670780346716, 0.8415913021441158, 0.7942324776528156]

random_selection = [0.5300990838967706, 0.6322358345971153, 0.693718171785043, 0.6088504669035559, 0.6574638203222735, 0.6391478544758442, 0.6602259544291998, 0.6457028741903733, 0.6564849553071853, 0.6862285396074289]

two_way_selection = [0.5454834403648007, 0.6688743105164064, 0.5946592165120189, 0.5561746902008702, 0.6355258816327773, 0.6470209803988087, 0.7200569571084432, 0.6448365376877682, 0.7017754779020116, 0.6926871253153707]

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
plt.ylabel(r'$f_n$')

# 强制显示完全
plt.tight_layout()

# 显示网格
plt.grid(True)

# 显示图表
plt.show()
