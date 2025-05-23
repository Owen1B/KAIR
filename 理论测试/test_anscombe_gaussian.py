# 该脚本用于测试Anscombe变换将泊松分布转换为高斯分布的效果，针对不同的泊松分布参数λ值进行验证。
# 主要输出:
# 1. 'anscombe_gaussian_test.png': 对于每个λ值，生成三张图：
#    - 原始泊松分布的直方图
#    - Anscombe变换后分布的直方图，并与理论高斯分布对比
#    - Anscombe变换后分布的Q-Q图
# 2. 控制台输出每个λ值下原始泊松分布、变换后分布及理论高斯分布的均值、标准差、偏度和峰度等统计特性，
#    以及对变换后分布进行的Shapiro-Wilk正态性检验结果。
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def anscombe_transform(x):
    """Anscombe变换"""
    return 2.0 * np.sqrt(np.maximum(x, 0) + 3.0/8.0)

def test_poisson_to_gaussian(lambda_values=[1, 5, 10, 50, 200], n_samples=10000):
    """测试不同λ值下Anscombe变换的效果"""
    fig, axes = plt.subplots(len(lambda_values), 3, figsize=(15, 4*len(lambda_values)))
    
    for i, lambda_val in enumerate(lambda_values):
        # 生成泊松分布样本
        poisson_samples = np.random.poisson(lambda_val, size=n_samples)
        
        # 应用Anscombe变换
        transformed_samples = anscombe_transform(poisson_samples)
        
        # 生成理论高斯分布
        theoretical_mean = 2 * np.sqrt(lambda_val + 3/8)
        theoretical_std = 1.0  # Anscombe变换后的理论标准差
        gaussian_samples = np.random.normal(theoretical_mean, theoretical_std, size=n_samples)
        
        # 1. 直方图对比
        sns.histplot(poisson_samples, ax=axes[i,0], label='泊松分布', stat='density')
        axes[i,0].set_title(f'原始泊松分布 (λ={lambda_val})')
        axes[i,0].legend()
        
        # 2. 变换后的分布
        sns.histplot(transformed_samples, ax=axes[i,1], label='变换后分布', stat='density')
        sns.histplot(gaussian_samples, ax=axes[i,1], label='理论高斯分布', stat='density', alpha=0.5)
        axes[i,1].set_title(f'Anscombe变换后 (λ={lambda_val})')
        axes[i,1].legend()
        
        # 3. Q-Q图
        stats.probplot(transformed_samples, dist="norm", plot=axes[i,2])
        axes[i,2].set_title(f'Q-Q图 (λ={lambda_val})')
        
        # 计算统计量
        poisson_stats = {
            'mean': np.mean(poisson_samples),
            'std': np.std(poisson_samples),
            'skew': stats.skew(poisson_samples),
            'kurtosis': stats.kurtosis(poisson_samples)
        }
        
        transformed_stats = {
            'mean': np.mean(transformed_samples),
            'std': np.std(transformed_samples),
            'skew': stats.skew(transformed_samples),
            'kurtosis': stats.kurtosis(transformed_samples)
        }
        
        gaussian_stats = {
            'mean': np.mean(gaussian_samples),
            'std': np.std(gaussian_samples),
            'skew': stats.skew(gaussian_samples),
            'kurtosis': stats.kurtosis(gaussian_samples)
        }
        
        print(f"\n=== λ = {lambda_val} 的统计特性 ===")
        print("原始泊松分布:")
        print(f"  均值: {poisson_stats['mean']:.2f}")
        print(f"  标准差: {poisson_stats['std']:.2f}")
        print(f"  偏度: {poisson_stats['skew']:.2f}")
        print(f"  峰度: {poisson_stats['kurtosis']:.2f}")
        
        print("\nAnscombe变换后:")
        print(f"  均值: {transformed_stats['mean']:.2f}")
        print(f"  标准差: {transformed_stats['std']:.2f}")
        print(f"  偏度: {transformed_stats['skew']:.2f}")
        print(f"  峰度: {transformed_stats['kurtosis']:.2f}")
        
        print("\n理论高斯分布:")
        print(f"  均值: {gaussian_stats['mean']:.2f}")
        print(f"  标准差: {gaussian_stats['std']:.2f}")
        print(f"  偏度: {gaussian_stats['skew']:.2f}")
        print(f"  峰度: {gaussian_stats['kurtosis']:.2f}")
        
        # 进行正态性检验
        shapiro_test = stats.shapiro(transformed_samples)
        print(f"\nShapiro-Wilk正态性检验:")
        print(f"  统计量: {shapiro_test[0]:.4f}")
        print(f"  p值: {shapiro_test[1]:.4f}")
        print(f"  是否服从正态分布: {'是' if shapiro_test[1] > 0.05 else '否'}")
    
    plt.tight_layout()
    plt.savefig('anscombe_gaussian_test.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    test_poisson_to_gaussian() 