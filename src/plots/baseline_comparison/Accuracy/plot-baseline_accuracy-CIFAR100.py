import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# 设置全局字体为 Times New Roman
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 16  # 同时设置字体大小

# 定义轮次
users = range(100)

# utility_baseline =
utility_pgirdfl = [0.016, 0.018, 0.0135, 0.0129, 0.0119, 0.0133, 0.014, 0.0124, 0.022, 0.0221, 0.0255, 0.0269, 0.0313, 0.0349, 0.0358, 0.0395, 0.0423, 0.047, 0.0493, 0.0481, 0.0512, 0.0576, 0.0535, 0.0592, 0.0601, 0.0636, 0.0617, 0.0626, 0.0668, 0.073, 0.0762, 0.0781, 0.0737, 0.0791, 0.079, 0.0866, 0.0904, 0.0963, 0.0903, 0.0901, 0.0929, 0.0916, 0.0946, 0.0942, 0.1006, 0.0987, 0.1028, 0.1006, 0.1059, 0.1035, 0.1083, 0.1053, 0.1121, 0.1142, 0.1117, 0.1155, 0.1145, 0.1144, 0.1124, 0.1154, 0.1176, 0.118, 0.1235, 0.1201, 0.1204, 0.1251, 0.1193, 0.1236, 0.1274, 0.1218, 0.13, 0.1285, 0.1294, 0.1358, 0.1279, 0.126, 0.1349, 0.1351, 0.1375, 0.1319, 0.1352, 0.1316, 0.139, 0.1379, 0.1403, 0.1355, 0.1434, 0.1405, 0.1452, 0.1478, 0.144, 0.1461, 0.1416, 0.1459, 0.147, 0.1441, 0.1449, 0.1411, 0.148, 0.1439]
# utility_fixed =
# utility_random =
# utility_FL_Client_Sampling =

utility_pgirdfl2 = [0.0294, 0.0833, 0.1069, 0.1188, 0.1333, 0.1466, 0.1561, 0.1634, 0.1724, 0.179, 0.1885, 0.1913, 0.2003, 0.2032, 0.208, 0.2138, 0.2169, 0.2213, 0.227, 0.227, 0.227, 0.227, 0.227, 0.2277, 0.2277, 0.2334, 0.2334, 0.2334, 0.2334, 0.2334, 0.2334, 0.2334, 0.2334, 0.2334, 0.2334, 0.2334, 0.2334, 0.2334, 0.2334, 0.2334, 0.2334, 0.2334, 0.2334, 0.2334, 0.2334, 0.2334, 0.2334, 0.2334, 0.2334, 0.2334, 0.2334, 0.2334, 0.2334, 0.2334, 0.2335, 0.2335, 0.2335, 0.2338, 0.2339, 0.2347, 0.2353, 0.2364, 0.2389, 0.2389, 0.2389, 0.2393, 0.2393, 0.2393, 0.2393, 0.2393, 0.2393, 0.2393, 0.2393, 0.2393, 0.2393, 0.2393, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24, 0.24]

# 绘制数据线
# plt.plot(users, utility_baseline, 'r--o', markersize=5, linewidth=2, label='Baseline')
plt.plot(users, utility_pgirdfl, 'b', markersize=5, linewidth=2, label='PGI-RDFL')
# plt.plot(users, utility_fixed, 'y--D', markersize=5, linewidth=2, label='Fixed')
# plt.plot(users, utility_random, 'g--^', markersize=5, linewidth=2, label='Random')
# plt.plot(users, utility_FL_Client_Sampling, 'm--*', markersize=5, linewidth=2, label='Client-Sampling')

plt.plot(users, utility_pgirdfl2, 'g', markersize=5, linewidth=2, label='PGI-RDFL(without force update)')

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
