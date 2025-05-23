# 该脚本旨在探索当泊松噪声图像经过最小-最大归一化处理后，其统计特性如何变化，
# 以及如何通过比较这些统计特性来判断两幅归一化图像是否源自相同或不同剂量的原始理想图像，
# 并尝试反推未知图像的相对剂量因子。
# 主要步骤包括：
# 1. 生成一个理想图像 (I_orig)。
# 2. 对理想图像进行泊松采样得到 Image 1 (X1_raw)，进行归一化 (N1_norm)，并记录原始范围以重建泊松参数的最佳估计 (I_estimated)。
# 3. 以特定剂量因子 (D_true_for_N2) 缩放理想图像，泊松采样得到 Image 2 (X2_raw)，并归一化 (N2_norm)，模拟未知来源的图像。
# 4. 第一部分：判断N2_norm是否与一个基于I_estimated、剂量为1的参考归一化图像 (N_ref_dose1) 具有相似的统计分布（直方图SSD, KS检验）。
# 5. 第二部分：通过在预设的剂量因子搜索空间 (D_search_space) 中迭代，为每个测试剂量因子 (D_test) 生成模拟的归一化图像，
#    并计算其与N2_norm的直方图SSD。选择SSD最小的D_test作为N2_norm的估计剂量因子 (D_estimated)。
# 主要输出:
# 1. 'dose_estimation_mismatch_curve.png': 一张图，显示测试剂量因子D_test与平均直方图失配度（SSD）之间的关系曲线，
#    并标出真实的D_true_for_N2和估计的最佳D_estimated。
# 2. 控制台输出：
#    - Image 1 和 Image 2 的生成参数和范围信息。
#    - N2_norm 与 N_ref_dose1 (模拟D=1的参考) 的统计比较结果（直方图SSD, 均值、方差、偏度、峰度，KS检验）。
#    - D_search_space 中每个D_test的平均失配度。
#    - 最终对Image 2的真实剂量因子D_true与估计剂量因子D_estimated的比较。
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from skimage.metrics import structural_similarity # 可以备用，但主要用直方图和KS

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 获取脚本所在目录用于保存图像
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def create_ideal_image(size=(128, 128), max_val=200):
    """生成一个简单的理想图像"""
    img = np.zeros(size, dtype=np.float32)
    img[size[0]//4:size[0]*3//4, size[1]//4:size[1]*3//4] = max_val * 0.75
    img[size[0]//8:size[0]*7//8, size[1]//8:size[1]*3//8] = max_val * 0.5
    img[size[0]//2:size[0]*7//8, size[1]//2:size[1]*7//8] = max_val * 0.25
    img[size[0]//16:size[0]*3//16, size[1]//16:size[1]*3//16] = max_val * 0.95
    return np.clip(img, 1, max_val) 

def add_poisson_noise(image):
    """
    向图像添加泊松噪声。
    确保输入图像为非负数代表计数。
    """
    image_counts = np.maximum(image, 0) 
    noisy_image = np.random.poisson(image_counts).astype(np.float32)
    return noisy_image

def poisson_sample(ideal_image):
    """对理想图像进行泊松采样"""
    return np.random.poisson(np.maximum(ideal_image, 0)).astype(np.float32)

def normalize_min_max(image_raw):
    """对原始图像进行最小-最大归一化到[0,1]"""
    min_val = np.min(image_raw)
    max_val = np.max(image_raw)
    if max_val == min_val:
        return np.zeros_like(image_raw), min_val, max_val
    normalized_image = (image_raw - min_val) / (max_val - min_val)
    return normalized_image, min_val, max_val

def reconstruct_from_normalized(normalized_image, min_original, max_original):
    """从归一化图像和原始范围重建原始计数值图像"""
    if max_original == min_original:
        return np.full_like(normalized_image, min_original)
    return normalized_image * (max_original - min_original) + min_original

def get_histogram(image_flat, bins=50, range_hist=(0,1)):
    """计算展平图像的归一化直方图"""
    hist, bin_edges = np.histogram(image_flat, bins=bins, range=range_hist, density=True)
    return hist, bin_edges

def get_descriptive_stats(image_flat):
    """计算展平图像的描述性统计量"""
    mean = np.mean(image_flat)
    var = np.var(image_flat)
    skewness = stats.skew(image_flat)
    kurt = stats.kurtosis(image_flat) 
    return mean, var, skewness, kurt

def compare_images_stats(N_ref_flat, N2_flat, bins=50):
    """比较两张归一化图像的统计特性"""
    print("\n--- 统计特性比较 (N_ref vs N2_norm) ---")
    
    hist_ref, _ = get_histogram(N_ref_flat, bins=bins)
    hist_N2, _ = get_histogram(N2_flat, bins=bins)
    hist_ssd = np.sum((hist_ref - hist_N2)**2) / bins 
    print(f"直方图SSD (越小越相似): {hist_ssd:.4e}")

    stats_ref = get_descriptive_stats(N_ref_flat)
    stats_N2 = get_descriptive_stats(N2_flat)
    print(f" N_ref: Mean={stats_ref[0]:.3f}, Var={stats_ref[1]:.3f}, Skew={stats_ref[2]:.3f}, Kurt={stats_ref[3]:.3f}")
    print(f" N2_norm: Mean={stats_N2[0]:.3f}, Var={stats_N2[1]:.3f}, Skew={stats_N2[2]:.3f}, Kurt={stats_N2[3]:.3f}")

    ks_stat, ks_p_value = stats.ks_2samp(N_ref_flat, N2_flat)
    print(f"KS检验: Statistic={ks_stat:.4f}, P-value={ks_p_value:.4f}")
    if ks_p_value > 0.05:
        print("  KS检验结论: 不能拒绝两样本来自相同分布的假设 (可能同源)。")
    else:
        print("  KS检验结论: 拒绝两样本来自相同分布的假设 (可能不同源或分布有差异)。")
    return hist_ssd, ks_p_value

def histogram_ssd(hist1, hist2):
    """计算两个直方图之间的SSD"""
    if len(hist1) != len(hist2):
        raise ValueError("直方图长度必须相同")
    return np.sum((hist1 - hist2)**2) / len(hist1)

def main():
    image_size = (64, 64) 
    ideal_max_count = 150 
    num_simulations_for_D = 10 

    print("--- 1. 准备图像 ---")
    I_orig = create_ideal_image(size=image_size, max_val=ideal_max_count)
    
    X1_raw = poisson_sample(I_orig)
    N1_norm, min_X1, max_X1 = normalize_min_max(X1_raw)
    X1_reconstructed = reconstruct_from_normalized(N1_norm, min_X1, max_X1)
    I_estimated = np.maximum(X1_reconstructed, 0) 

    print(f"Image 1: 原始理想图像最大值 approx {I_orig.max():.1f}")
    print(f"Image 1: 泊松采样后 X1_raw范围 [{X1_raw.min():.1f}, {X1_raw.max():.1f}]")
    print(f"Image 1: 记录的归一化系数 min_X1={min_X1:.1f}, max_X1={max_X1:.1f}")
    print(f"Image 1: 重建的 X1_reconstructed (Î) 范围 [{I_estimated.min():.1f}, {I_estimated.max():.1f}]")

    D_true_for_N2 = 0.5 
    
    I_for_N2 = I_orig * D_true_for_N2 
    X2_raw = poisson_sample(I_for_N2)
    N2_norm, _, _ = normalize_min_max(X2_raw) 

    print(f"Image 2: 真实剂量因子 D_true={D_true_for_N2}")
    print(f"Image 2: 缩放后理想图像最大值 approx {I_for_N2.max():.1f}")
    print(f"Image 2: 泊松采样后 X2_raw范围 [{X2_raw.min():.1f}, {X2_raw.max():.1f}]")
    print(f"Image 2: 归一化后 N2_norm范围 [{N2_norm.min():.1f}, {N2_norm.max():.1f}]")

    print("\n--- 2. 判断是否来自同一剂量 (D=1的假设) ---")
    X_ref_dose1 = poisson_sample(I_estimated) 
    N_ref_dose1, _, _ = normalize_min_max(X_ref_dose1)

    compare_images_stats(N_ref_dose1.flatten(), N2_norm.flatten())

    print("\n--- 3. 反推 Image 2 的剂量因子 D ---")
    
    D_search_space = np.concatenate([
        np.linspace(0.1, 0.4, 4, endpoint=False),
        np.linspace(0.4, 0.7, 7, endpoint=False), 
        np.linspace(0.7, 1.2, 6, endpoint=False), 
        np.linspace(1.2, 2.0, 5)
    ])

    print(f"搜索D的范围: {D_search_space}")
    best_D_estimated = -1
    min_mismatch_score = float('inf')
    
    mismatch_scores = []
    hist_N2_target, _ = get_histogram(N2_norm.flatten()) # Pre-calculate target histogram

    for D_test in D_search_space:
        print(f"  测试剂量因子 D_test = {D_test:.3f}")
        current_D_mismatches = []
        lambda_model_base = I_estimated * D_test 

        for i_sim in range(num_simulations_for_D):
            X_model_sample = poisson_sample(lambda_model_base)
            N_model_sample, _, _ = normalize_min_max(X_model_sample)
            
            hist_model, _ = get_histogram(N_model_sample.flatten())
            mismatch = histogram_ssd(hist_model, hist_N2_target)
            current_D_mismatches.append(mismatch)
        
        avg_mismatch_for_D_test = np.mean(current_D_mismatches)
        mismatch_scores.append(avg_mismatch_for_D_test)
        print(f"    平均失配度 (直方图SSD): {avg_mismatch_for_D_test:.4e}")

        if avg_mismatch_for_D_test < min_mismatch_score:
            min_mismatch_score = avg_mismatch_for_D_test
            best_D_estimated = D_test
            
    print("\n反推完成。")
    print(f"Image 2 的真实剂量因子 D_true: {D_true_for_N2:.3f}")
    print(f"Image 2 的估计剂量因子 D_estimated: {best_D_estimated:.3f} (基于最小直方图SSD)")

    plt.figure(figsize=(8,5))
    plt.plot(D_search_space, mismatch_scores, 'o-')
    plt.axvline(D_true_for_N2, color='r', linestyle='--', label=f'真实D = {D_true_for_N2:.2f}')
    plt.axvline(best_D_estimated, color='g', linestyle=':', label=f'估计D = {best_D_estimated:.2f}')
    plt.xlabel("测试剂量因子 (D_test)")
    plt.ylabel("平均失配度 (直方图SSD)")
    plt.title("剂量因子估计的失配度曲线")
    plt.legend()
    plt.grid(True)
    
    fig_dir = os.path.join(SCRIPT_DIR, 'fig')
    os.makedirs(fig_dir, exist_ok=True)
    plot_filename = os.path.join(fig_dir, "泊松归一化误差_剂量估计失配度曲线.png") # Updated filename
    plt.savefig(plot_filename)
    print(f"\n失配度曲线已保存为: {plot_filename}")
    plt.close() # Close the figure

if __name__ == '__main__':
    main() 