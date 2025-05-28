import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson
plt.rcParams['font.family'] = ['SimHei']

# 参数设置
n_samples = 50000000
lambda_original = 100  # 原始泊松分布参数
p = 0.25  # 二项重采样概率
lambda_expected = lambda_original * p  # 期望的均值

# 1. 从参数为12的泊松分布采样
poisson_samples = np.random.poisson(lambda_original, n_samples)

# 2. 对每个样本进行二项重采样
binomial_samples = np.random.binomial(poisson_samples, p)

# 3. 从参数为3的泊松分布采样（作为对照）
poisson_expected = np.random.poisson(lambda_expected, n_samples)

# 创建图形
plt.figure(figsize=(15, 5))

# 计算三个分布的x范围（取最大值，保证x轴一致且间隔为1）
x_min = 0
x_max = 120  # 限制x轴范围
x = np.arange(x_min, x_max + 1, 1)

# 1. 绘制原始泊松分布
plt.subplot(131)
counts, _, _ = plt.hist(poisson_samples, bins=np.arange(x_min, x_max + 2, 1), density=True, alpha=0.7, label='采样值', align='left', edgecolor='black')
plt.plot(x, poisson.pmf(x, lambda_original), 'r-', label='理论分布')
plt.title(f'泊松分布 (λ={lambda_original})')
plt.xlabel('计数值')
plt.ylabel('概率密度')
plt.legend()
plt.xticks(np.arange(x_min, x_max + 1, 20))  # 每20个单位显示一个刻度
plt.xlim(x_min, x_max)

# 2. 绘制二项重采样后的分布
plt.subplot(132)
counts, _, _ = plt.hist(binomial_samples, bins=np.arange(x_min, x_max + 2, 1), density=True, alpha=0.7, label='采样值', align='left', edgecolor='black')
plt.plot(x, poisson.pmf(x, lambda_expected), 'r-', label='理论分布')
plt.title(f'二项重采样后 (p={p})')
plt.xlabel('计数值')
plt.ylabel('概率密度')
plt.legend()
plt.xticks(np.arange(x_min, x_max + 1, 20))  # 每20个单位显示一个刻度
plt.xlim(x_min, x_max)

# 3. 绘制参数为3的泊松分布
plt.subplot(133)
counts, _, _ = plt.hist(poisson_expected, bins=np.arange(x_min, x_max + 2, 1), density=True, alpha=0.7, label='采样值', align='left', edgecolor='black')
plt.plot(x, poisson.pmf(x, lambda_expected), 'r-', label='理论分布')
plt.title(f'泊松分布 (λ={lambda_expected})')
plt.xlabel('计数值')
plt.ylabel('概率密度')
plt.legend()
plt.xticks(np.arange(x_min, x_max + 1, 20))  # 每20个单位显示一个刻度
plt.xlim(x_min, x_max)

plt.tight_layout()
plt.savefig('二项重采样测试结果.png', dpi=300, bbox_inches='tight')
plt.show()

# 打印统计信息
print(f"原始泊松分布 (λ={lambda_original}):")
print(f"均值: {np.mean(poisson_samples):.1f}")
print(f"方差: {np.var(poisson_samples):.1f}")
print(f"\n二项重采样后 (p={p}):")
print(f"均值: {np.mean(binomial_samples):.1f}")
print(f"方差: {np.var(binomial_samples):.1f}")
print(f"\n对照泊松分布 (λ={lambda_expected}):")
print(f"均值: {np.mean(poisson_expected):.1f}")
print(f"方差: {np.var(poisson_expected):.1f}")
