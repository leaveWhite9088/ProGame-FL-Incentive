import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# 设置全局字体为 Times New Roman
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 16  # 同时设置字体大小

# 定义用户数量
users = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

utility_com =[-0.16108802169660663, 0.4261408607962416, 0.8060637175441556, 1.5279618840947151, 2.205029942683666, 2.4978474557156716, 2.6413754987425477, 3.5842624395753617, 4.339412296497329, 4.67074824094658]

utility_pgirdfl =[-1.3877787807814457e-17, -0.008364993013729706, 0.0015354897338530071, -0.0063507938380771844, -0.00845145410035087, -0.006868191929079181, -0.004933797692281921, -0.0052052451425088865, -0.003724748209734383, -0.00129257896486979]

utility_fixed = [-0.18582113773165398, -0.1489218724065775, -0.1557963650352039, -0.16820814869227352, -0.143677498550823, -0.16682732548619195, -0.17072604531871394, -0.2002050767666944, -0.16097771483756418, -0.18039590884047485]

utility_random =[0.0, 0.007528475710599447, 0.032754682128633306, -0.05147270671653448, 0.021493383935970528, 0.03921219515861307, 0.025403979781653707, -0.033254756978255084, 0.047045417762795694, 0.03692863801742019]

# 绘制图表
plt.plot(users, utility_com, 'r--o', label='Comparison')  # 红色虚线，圆形标记
plt.plot(users, utility_pgirdfl, 'b--s', label='PGI-RDFL')  # 蓝色虚线，方形标记
plt.plot(users, utility_fixed, 'y--o', label='Fixed')
plt.plot(users, utility_random, 'g--^', label='Random')  # 绿色虚线，三角形标记

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