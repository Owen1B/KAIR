# 该脚本用于可视化Anscombe变换的效果，特别关注其方差稳定特性以及对数据正态性的影响，同时也会展示变换后再进行0-1归一化的效果。
# 主要输出:
# 1. 'anscombe_transform_analysis.png': 一张包含多个子图的图像，展示：
#    - 不同计数值下，原始方差、变换后方差、归一化后方差的对比。
#    - 几个典型计数值下，原始分布、变换后分布、归一化后分布的核密度估计图。
#    - 中等计数值下，变换后和归一化后数据的Q-Q图。
#    - 不同计数值下，变换后和归一化后数据的偏度和峰度对比。
# 2. 控制台输出详细的统计信息，包括：
#    - 对中等计数值样本进行变换和归一化后的正态性检验结果 (Shapiro-Wilk, D'Agostino's K^2)。
#    - 变换后和归一化后方差、偏度、峰度的范围总结。
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def anscombe_transform(x):
    """Anscombe变换"""
    return 2.0 * np.sqrt(4*np.maximum(x, 0) + 3.0/8.0)

def normalize_to_01(x):
    """归一化到[0,1]范围"""
    return (x - x.min()) / (x.max() - x.min())

def simulate_poisson_samples(mean_counts, n_samples=1000):
    """对给定的计数值进行泊松采样"""
    return np.random.poisson(mean_counts, size=n_samples)

def calculate_statistics(samples):
    """计算样本的统计特性"""
    return {
        'mean': np.mean(samples),
        'std': np.std(samples),
        'var': np.var(samples),
        'cv': np.std(samples) / np.mean(samples) if np.mean(samples) > 0 else 0,
        'skewness': stats.skew(samples),
        'kurtosis': stats.kurtosis(samples)
    }

def test_normality(samples):
    """进行正态性检验"""
    # Shapiro-Wilk检验
    shapiro_test = stats.shapiro(samples)
    # D'Agostino's K^2检验
    k2_test = stats.normaltest(samples)
    return {
        'shapiro_stat': shapiro_test[0],
        'shapiro_p': shapiro_test[1],
        'k2_stat': k2_test[0],
        'k2_p': k2_test[1]
    }

def main():
    # 设置要测试的计数值范围
    count_values = np.linspace(1, 200, 20)  # 20个均匀分布的计数值
    n_samples = 100000  # 每个计数值的采样数
    
    # 存储结果
    original_stats = []
    transformed_stats = []
    normalized_stats = []
    
    # 对每个计数值进行测试
    for count in count_values:
        # 生成泊松采样
        samples = simulate_poisson_samples(count, n_samples)
        # 应用Anscombe变换
        transformed_samples = anscombe_transform(samples)
        # 归一化
        normalized_samples = normalize_to_01(transformed_samples)
        
        # 计算统计特性
        original_stats.append(calculate_statistics(samples))
        transformed_stats.append(calculate_statistics(transformed_samples))
        normalized_stats.append(calculate_statistics(normalized_samples))
    
    # 转换为numpy数组以便绘图
    count_values = np.array(count_values)
    original_means = np.array([s['mean'] for s in original_stats])
    original_vars = np.array([s['var'] for s in original_stats])
    transformed_vars = np.array([s['var'] for s in transformed_stats])
    normalized_vars = np.array([s['var'] for s in normalized_stats])
    
    # 创建图形
    fig = plt.figure(figsize=(15, 15))
    
    # 1. 方差对比图
    ax1 = plt.subplot(321)
    ax1.plot(count_values, original_vars, 'b-', label='原始方差')
    ax1.plot(count_values, transformed_vars, 'r-', label='变换后方差')
    ax1.plot(count_values, normalized_vars, 'g-', label='归一化后方差')
    ax1.set_xlabel('计数值')
    ax1.set_ylabel('方差')
    ax1.set_title('方差对比')
    ax1.legend()
    ax1.grid(True)
    
    # 2. 选择几个典型的计数值，展示分布变化
    selected_counts = [1, 10, 50, 200]
    ax2 = plt.subplot(322)
    for count in selected_counts:
        samples = simulate_poisson_samples(count, n_samples)
        transformed = anscombe_transform(samples)
        normalized = normalize_to_01(transformed)
        sns.kdeplot(samples, ax=ax2, label=f'原始分布 (λ={count})')
        sns.kdeplot(transformed, ax=ax2, label=f'变换后分布 (λ={count})')
        sns.kdeplot(normalized, ax=ax2, label=f'归一化后分布 (λ={count})')
    ax2.set_xlabel('值')
    ax2.set_ylabel('密度')
    ax2.set_title('分布对比')
    ax2.legend()
    
    # 3. Q-Q图对比
    ax3 = plt.subplot(323)
    count = 50  # 选择一个中等计数值
    samples = simulate_poisson_samples(count, n_samples)
    transformed = anscombe_transform(samples)
    normalized = normalize_to_01(transformed)
    stats.probplot(transformed, dist="norm", plot=ax3)
    ax3.set_title('变换后Q-Q图')
    
    ax4 = plt.subplot(324)
    stats.probplot(normalized, dist="norm", plot=ax4)
    ax4.set_title('归一化后Q-Q图')
    
    # 4. 偏度和峰度对比
    ax5 = plt.subplot(325)
    transformed_skew = np.array([s['skewness'] for s in transformed_stats])
    normalized_skew = np.array([s['skewness'] for s in normalized_stats])
    ax5.plot(count_values, transformed_skew, 'r-', label='变换后偏度')
    ax5.plot(count_values, normalized_skew, 'g-', label='归一化后偏度')
    ax5.axhline(y=0, color='k', linestyle='--', label='高斯分布偏度')
    ax5.set_xlabel('计数值')
    ax5.set_ylabel('偏度')
    ax5.set_title('偏度对比')
    ax5.legend()
    ax5.grid(True)
    
    ax6 = plt.subplot(326)
    transformed_kurt = np.array([s['kurtosis'] for s in transformed_stats])
    normalized_kurt = np.array([s['kurtosis'] for s in normalized_stats])
    ax6.plot(count_values, transformed_kurt, 'r-', label='变换后峰度')
    ax6.plot(count_values, normalized_kurt, 'g-', label='归一化后峰度')
    ax6.axhline(y=0, color='k', linestyle='--', label='高斯分布峰度')
    ax6.set_xlabel('计数值')
    ax6.set_ylabel('峰度')
    ax6.set_title('峰度对比')
    ax6.legend()
    ax6.grid(True)
    
    plt.tight_layout()
    plt.savefig('anscombe_transform_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 打印统计信息
    print("\n=== Anscombe变换和归一化效果分析 ===")
    
    # 对中等计数值进行正态性检验
    count = 50
    samples = simulate_poisson_samples(count, n_samples)
    transformed = anscombe_transform(samples)
    normalized = normalize_to_01(transformed)
    
    print("\n1. 正态性检验 (λ=50):")
    print("变换后:")
    print(test_normality(transformed))
    print("\n归一化后:")
    print(test_normality(normalized))
    
    print("\n2. 方差分析:")
    print(f"变换后方差范围: {transformed_vars.min():.2f} - {transformed_vars.max():.2f}")
    print(f"归一化后方差范围: {normalized_vars.min():.2f} - {normalized_vars.max():.2f}")
    
    print("\n3. 偏度和峰度分析:")
    print(f"变换后偏度范围: {transformed_skew.min():.2f} - {transformed_skew.max():.2f}")
    print(f"归一化后偏度范围: {normalized_skew.min():.2f} - {normalized_skew.max():.2f}")
    print(f"变换后峰度范围: {transformed_kurt.min():.2f} - {transformed_kurt.max():.2f}")
    print(f"归一化后峰度范围: {normalized_kurt.min():.2f} - {normalized_kurt.max():.2f}")

if __name__ == '__main__':
    main() 