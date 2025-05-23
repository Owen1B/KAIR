# 该脚本用于测试Anscombe变换在不同剂量水平下对SPECT图像的PSNR和SSIM指标的影响。
# 主要输出:
# 1. 'dose_metrics_comparison.png': 包含四张图，分别显示原始域和Anscombe域中，
#    - 不同剂量水平与标准剂量比较的PSNR曲线
#    - 不同剂量水平与标准剂量比较的SSIM曲线
#    - 不同剂量水平下组间PSNR曲线
#    - 不同剂量水平下组间SSIM曲线
# 2. 控制台输出详细的平均指标数据。
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def load_spect_data(file_path):
    """加载SPECT数据文件"""
    try:
        data = np.fromfile(file_path, dtype=np.float32)
        return data.astype(np.float32)
    except Exception as e:
        print(f"Error loading SPECT data: {e}")
        return None

def anscombe_transform(x):
    """Anscombe变换"""
    return 2.0 * np.sqrt(np.maximum(x, 0) + 3.0/8.0)

def inverse_anscombe_transform(x):
    """Anscombe逆变换"""
    return (x/2.0)**2 - 3.0/8.0

def normalize_to_255(img):
    """将图像归一化到0-255范围"""
    img_min = np.min(img)
    img_max = np.max(img)
    return ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)

def calculate_metrics(ideal, noisy, transformed):
    """计算所有指标"""
    ideal_norm = normalize_to_255(ideal)
    noisy_norm = normalize_to_255(noisy)
    transformed_norm = normalize_to_255(transformed)
    
    # 计算理想图像的Anscombe变换
    ideal_transformed = anscombe_transform(ideal)
    ideal_transformed_norm = normalize_to_255(ideal_transformed)
    
    metrics = {
        'original_psnr': psnr(ideal_norm, noisy_norm),
        'original_ssim': ssim(ideal_norm, noisy_norm),
        'anscombe_psnr': psnr(ideal_transformed_norm, transformed_norm),
        'anscombe_ssim': ssim(ideal_transformed_norm, transformed_norm)
    }
    return metrics

def main(size=256, n_samples=10):
    # 加载SPECT数据作为理想图像
    spect_file = 'trainsets/spectH_fwhm7/B001_pure.dat'
    ideal_image = load_spect_data(spect_file)
    if ideal_image is None:
        print("Failed to load SPECT data")
        return
    
    # 定义不同的剂量水平
    dose_levels = [1, 2, 3, 4, 5,6,7,8,9,10]  # 1表示标准剂量，其他表示标准剂量的1/n
    
    # 存储每个剂量水平的指标
    metrics_by_dose = {dose: [] for dose in dose_levels}
    # 存储每个剂量水平的组间指标
    inter_group_metrics = {dose: [] for dose in dose_levels}
    
    for i in range(n_samples):
        # 对标准剂量图像进行两次泊松采样
        noisy_image1 = np.random.poisson(ideal_image)
        noisy_image2 = np.random.poisson(ideal_image)
        
        # 计算标准剂量下的组间指标
        inter_group_metrics[1].append({
            'original_psnr': psnr(normalize_to_255(noisy_image1), normalize_to_255(noisy_image2)),
            'original_ssim': ssim(normalize_to_255(noisy_image1), normalize_to_255(noisy_image2)),
            'anscombe_psnr': psnr(normalize_to_255(anscombe_transform(noisy_image1)), 
                                normalize_to_255(anscombe_transform(noisy_image2))),
            'anscombe_ssim': ssim(normalize_to_255(anscombe_transform(noisy_image1)), 
                                normalize_to_255(anscombe_transform(noisy_image2)))
        })
        
        # 对其他剂量水平进行采样和计算
        for dose in dose_levels[1:]:
            # 生成低剂量图像
            low_dose_image = ideal_image / dose
            # 进行两次泊松采样
            noisy_low_dose1 = np.random.poisson(low_dose_image)
            noisy_low_dose2 = np.random.poisson(low_dose_image)
            
            # 计算与标准剂量图像的指标
            metrics = {
                'original_psnr': psnr(normalize_to_255(noisy_image1), normalize_to_255(noisy_low_dose1)),
                'original_ssim': ssim(normalize_to_255(noisy_image1), normalize_to_255(noisy_low_dose1)),
                'anscombe_psnr': psnr(normalize_to_255(anscombe_transform(noisy_image1)), 
                                    normalize_to_255(anscombe_transform(noisy_low_dose1))),
                'anscombe_ssim': ssim(normalize_to_255(anscombe_transform(noisy_image1)), 
                                    normalize_to_255(anscombe_transform(noisy_low_dose1)))
            }
            metrics_by_dose[dose].append(metrics)
            
            # 计算组间指标
            inter_group_metrics[dose].append({
                'original_psnr': psnr(normalize_to_255(noisy_low_dose1), normalize_to_255(noisy_low_dose2)),
                'original_ssim': ssim(normalize_to_255(noisy_low_dose1), normalize_to_255(noisy_low_dose2)),
                'anscombe_psnr': psnr(normalize_to_255(anscombe_transform(noisy_low_dose1)), 
                                    normalize_to_255(anscombe_transform(noisy_low_dose2))),
                'anscombe_ssim': ssim(normalize_to_255(anscombe_transform(noisy_low_dose1)), 
                                    normalize_to_255(anscombe_transform(noisy_low_dose2)))
            })
    
    # 计算每个剂量水平的平均指标
    mean_metrics = {}
    mean_inter_group_metrics = {}
    for dose in dose_levels:
        if dose == 1:
            # 标准剂量只计算组间指标
            mean_metrics[dose] = {
                'original_psnr': 0,  # 标准剂量与自身比较，PSNR为无穷大
                'original_ssim': 1,  # 标准剂量与自身比较，SSIM为1
                'anscombe_psnr': 0,
                'anscombe_ssim': 1
            }
        else:
            mean_metrics[dose] = {
                'original_psnr': np.mean([m['original_psnr'] for m in metrics_by_dose[dose]]),
                'original_ssim': np.mean([m['original_ssim'] for m in metrics_by_dose[dose]]),
                'anscombe_psnr': np.mean([m['anscombe_psnr'] for m in metrics_by_dose[dose]]),
                'anscombe_ssim': np.mean([m['anscombe_ssim'] for m in metrics_by_dose[dose]])
            }
        mean_inter_group_metrics[dose] = {
            'original_psnr': np.mean([m['original_psnr'] for m in inter_group_metrics[dose]]),
            'original_ssim': np.mean([m['original_ssim'] for m in inter_group_metrics[dose]]),
            'anscombe_psnr': np.mean([m['anscombe_psnr'] for m in inter_group_metrics[dose]]),
            'anscombe_ssim': np.mean([m['anscombe_ssim'] for m in inter_group_metrics[dose]])
        }
    
    # 绘制PSNR和SSIM随剂量变化的曲线
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 与标准剂量的PSNR曲线
    ax1.plot(dose_levels, [mean_metrics[d]['original_psnr'] for d in dose_levels], 'b-o', label='原始域')
    ax1.plot(dose_levels, [mean_metrics[d]['anscombe_psnr'] for d in dose_levels], 'r-o', label='Anscombe域')
    ax1.set_xlabel('剂量水平 (1/n)')
    ax1.set_ylabel('PSNR')
    ax1.set_title('不同剂量水平下与标准剂量的PSNR')
    ax1.legend()
    ax1.grid(True)
    
    # 与标准剂量的SSIM曲线
    ax2.plot(dose_levels, [mean_metrics[d]['original_ssim'] for d in dose_levels], 'b-o', label='原始域')
    ax2.plot(dose_levels, [mean_metrics[d]['anscombe_ssim'] for d in dose_levels], 'r-o', label='Anscombe域')
    ax2.set_xlabel('剂量水平 (1/n)')
    ax2.set_ylabel('SSIM')
    ax2.set_title('不同剂量水平下与标准剂量的SSIM')
    ax2.legend()
    ax2.grid(True)
    
    # 组间PSNR曲线
    ax3.plot(dose_levels, [mean_inter_group_metrics[d]['original_psnr'] for d in dose_levels], 'b-o', label='原始域')
    ax3.plot(dose_levels, [mean_inter_group_metrics[d]['anscombe_psnr'] for d in dose_levels], 'r-o', label='Anscombe域')
    ax3.set_xlabel('剂量水平 (1/n)')
    ax3.set_ylabel('PSNR')
    ax3.set_title('不同剂量水平下组间PSNR')
    ax3.legend()
    ax3.grid(True)
    
    # 组间SSIM曲线
    ax4.plot(dose_levels, [mean_inter_group_metrics[d]['original_ssim'] for d in dose_levels], 'b-o', label='原始域')
    ax4.plot(dose_levels, [mean_inter_group_metrics[d]['anscombe_ssim'] for d in dose_levels], 'r-o', label='Anscombe域')
    ax4.set_xlabel('剂量水平 (1/n)')
    ax4.set_ylabel('SSIM')
    ax4.set_title('不同剂量水平下组间SSIM')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig('dose_metrics_comparison.png', dpi=300, bbox_inches='tight')
    
    # 打印详细数据
    print("\n=== 不同剂量水平下的平均指标 ===")
    for dose in dose_levels:
        print(f"\n剂量水平 1/{dose}:")
        print("与标准剂量比较:")
        print(f"原始域 PSNR: {mean_metrics[dose]['original_psnr']:.2f}")
        print(f"原始域 SSIM: {mean_metrics[dose]['original_ssim']:.4f}")
        print(f"Anscombe域 PSNR: {mean_metrics[dose]['anscombe_psnr']:.2f}")
        print(f"Anscombe域 SSIM: {mean_metrics[dose]['anscombe_ssim']:.4f}")
        print("\n组间比较:")
        print(f"原始域 PSNR: {mean_inter_group_metrics[dose]['original_psnr']:.2f}")
        print(f"原始域 SSIM: {mean_inter_group_metrics[dose]['original_ssim']:.4f}")
        print(f"Anscombe域 PSNR: {mean_inter_group_metrics[dose]['anscombe_psnr']:.2f}")
        print(f"Anscombe域 SSIM: {mean_inter_group_metrics[dose]['anscombe_ssim']:.4f}")

if __name__ == '__main__':
    main() 