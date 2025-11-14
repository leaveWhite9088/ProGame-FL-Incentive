import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# 设置全局字体为 Times New Roman
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 16  # 同时设置字体大小

# 定义用户数量
users = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

utility_com = [0.11727782369295085, 0.11667949560130161, 0.07255831921045175, 0.09940635898754797, 0.09118380996705401,
               0.07282785604777216, 0.08706099188965286, 0.0684412985471118, 0.08293199123103015, 0.06637327170098657]

utility_pgirdfl = [0.8376995085023538, 0.42049815408902347, 0.2840040470403707, 0.21296840449140894, 0.1697579959528504,
                   0.14205976102054776, 0.12225086619303718, 0.10641506904586476, 0.09495222520900705,
                   0.08490337478581564]

utility_fixed = [0.050609723577412266, 0.01908126388439124, 0.016916277586250476, 0.010628518668338423,
                 0.009907070610603417, 0.006023785138836667, 0.006065712967554416, 0.004598746373146738,
                 0.004635705351600605, 0.003339973029031975]

utility_random = [0.01183807797710396, 0.005711045649220027, 0.003055193390794655, 0.0011584736674741306,
                  0.001978269838407709, 0.0018546991105881712, 0.0011882126372278303, 0.0010575334225272758,
                  0.0011600681910249016, 0.0006672957989451178]

utility_tiflcsmarp = []

# 绘制图表
plt.plot(users, utility_com, 'r--o', label='Comparison')  # 红色虚线，圆形标记
plt.plot(users, utility_pgirdfl, 'b--s', label='PGI-RDFL')  # 蓝色虚线，方形标记
plt.plot(users, utility_fixed, 'y--o', label='Fixed')
plt.plot(users, utility_random, 'g--^', label='Random')  # 绿色虚线，三角形标记
plt.plot(users, utility_tiflcsmarp, 'b--^', label='TiFLCS-MARP')

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
