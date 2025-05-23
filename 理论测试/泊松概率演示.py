# 该脚本用于演示泊松分布的基本特性。
# 它首先生成一个随机的λ值（泊松分布的平均发生率）矩阵，
# 然后对每个λ值进行泊松采样得到一个样本值，
# 接着计算在该λ值下获得该特定样本值的概率（使用泊松概率质量函数 PMF）。
# 主要输出:
# 1. 控制台输出：
#    - 生成的λ值矩阵。
#    - 对每个λ值进行泊松采样得到的样本值矩阵。
#    - 每个样本值在其对应λ下的发生概率矩阵。
#    - 概率矩阵中所有概率值的乘积。
#    - 概率矩阵中所有概率值的平均值。
# 2. 一个包含四个子图的可视化窗口 (或保存的图像, 如果 plt.show() 被注释掉并替换为 plt.savefig()):
#    - λ值矩阵的热图。
#    - 采样值矩阵的热图。
#    - 概率矩阵的热图。
#    - 显示概率乘积和平均值的条形图（Y轴为对数刻度）。
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 设置随机种子以确保结果可重复
np.random.seed(42)

# 生成4x4的随机lambda值矩阵 (16个数)
lambdas = np.random.uniform(0, 200, (4, 4))

# 对每个lambda值进行泊松采样
samples = np.random.poisson(lambdas)

# 计算每个采样值对应的概率
probabilities = stats.poisson.pmf(samples, lambdas)

# 计算概率矩阵的乘积和平均值
prob_product = np.prod(probabilities)
prob_mean = np.mean(probabilities)

# 打印结果
print("\n=== 泊松分布概率计算结果 ===")
print("\nλ值矩阵:")
print(lambdas)
print("\n采样值矩阵:")
print(samples)
print("\n概率矩阵:")
print(probabilities)
print("\n概率矩阵乘积:", prob_product)
print("概率矩阵平均值:", prob_mean)

# 可视化结果
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

# 显示λ值矩阵
im0 = axes[0].imshow(lambdas, cmap='viridis')
axes[0].set_title('λ值矩阵')
plt.colorbar(im0, ax=axes[0])

# 显示采样值矩阵
im1 = axes[1].imshow(samples, cmap='viridis')
axes[1].set_title('采样值矩阵')
plt.colorbar(im1, ax=axes[1])

# 显示概率矩阵
im2 = axes[2].imshow(probabilities, cmap='viridis')
axes[2].set_title('概率矩阵')
plt.colorbar(im2, ax=axes[2])

# 显示概率统计结果
axes[3].bar(['乘积', '平均值'], [prob_product, prob_mean])
axes[3].set_title('概率统计结果')
axes[3].set_yscale('log')  # 使用对数刻度以更好地显示差异

plt.suptitle('泊松分布采样结果可视化')
plt.tight_layout()

# 保存图像到fig文件夹
fig_dir = os.path.join(SCRIPT_DIR, 'fig')
os.makedirs(fig_dir, exist_ok=True)
plt.savefig(os.path.join(fig_dir, '泊松分布采样结果可视化.png'))
plt.close(fig) # 关闭图像，避免显示
# plt.show() # 如果仍然需要显示，可以取消注释此行，但通常保存后就不需要了 