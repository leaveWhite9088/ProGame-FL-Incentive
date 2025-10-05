import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# 设置全局字体为 Times New Roman
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 16  # 同时设置字体大小

# 定义轮次
users = range(101)

# utility_baseline =
utility_pgirdfl = [0.12, 0.246, 0.2935, 0.3345, 0.3543, 0.3657, 0.38, 0.3876, 0.3918, 0.4051, 0.4041, 0.4144, 0.4174, 0.4224, 0.4266, 0.4255, 0.4235, 0.4229, 0.4298, 0.4406, 0.4326, 0.4348, 0.445, 0.4446, 0.4545, 0.4481, 0.4433, 0.4497, 0.4464, 0.458, 0.4513, 0.459, 0.4527, 0.4487, 0.456, 0.4523, 0.4629, 0.4628, 0.4743, 0.4734, 0.4683, 0.4701, 0.4711, 0.4649, 0.4634, 0.4676, 0.4746, 0.4659, 0.4727, 0.4756, 0.4766, 0.4743, 0.4688, 0.4744, 0.4799, 0.4705, 0.4679, 0.4821, 0.4753, 0.4666, 0.4786, 0.4789, 0.4762, 0.4826, 0.4735, 0.4664, 0.468, 0.4772, 0.4808, 0.4703, 0.4795, 0.4779, 0.467, 0.4787, 0.4745, 0.4691, 0.4677, 0.4709, 0.4826, 0.4681, 0.4839, 0.4792, 0.4834, 0.47, 0.4722, 0.4743, 0.4751, 0.4763, 0.4766, 0.4749, 0.4716, 0.4689, 0.4667, 0.4709, 0.4705, 0.4666, 0.4674, 0.4723, 0.4705, 0.472, 0.4714]
# utility_fixed =
# utility_random =
# utility_FL_Client_Sampling =

utility_pgirdfl2 = [0.12, 0.3528, 0.3951, 0.4103, 0.4517, 0.4517, 0.4517, 0.4517, 0.4517, 0.4517, 0.4517, 0.4517, 0.4517, 0.4517, 0.4517, 0.4517, 0.4517, 0.4522, 0.4564, 0.4626, 0.4795, 0.4848, 0.4868, 0.4934, 0.5007, 0.5146, 0.5201, 0.5269, 0.5269, 0.5269, 0.5312, 0.5312, 0.5332, 0.5332, 0.5332, 0.5332, 0.5333, 0.5333, 0.5333, 0.5333, 0.5333, 0.5415, 0.5486, 0.5486, 0.5486, 0.5486, 0.5486, 0.5486, 0.5486, 0.5486, 0.5486, 0.5486, 0.5486, 0.5486, 0.5486, 0.5486, 0.5486, 0.5486, 0.5486, 0.5486, 0.5486, 0.5486, 0.5493, 0.5493, 0.5493, 0.5493, 0.5511, 0.5511, 0.5511, 0.5511, 0.5511, 0.5511, 0.5511, 0.5511, 0.5511, 0.5511, 0.5531, 0.5531, 0.5531, 0.5531, 0.5531, 0.5531, 0.5531, 0.5531, 0.5531, 0.5531, 0.5531, 0.5531, 0.5531, 0.5534, 0.5534, 0.5534, 0.5534, 0.5534, 0.5534, 0.5534, 0.5534, 0.5534, 0.5534, 0.5534, 0.5558]

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
