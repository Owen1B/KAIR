# 该脚本旨在比较两幅经过泊松噪声处理并归一化的图像的统计特性，以判断它们是否可能源自相同剂量水平的同一理想图像。
# 核心方法是通过分析图像局部块的均值-方差关系，使用RANSAC算法估计一个与图像原始最大计数值相关的归一化因子N（理论上 方差 ≈ N * 均值）。
# 脚本首先基于参考理想图像（一个真实的SPECT数据）建立参考N值的经验分布（均值和标准差）。
# 然后，它生成多个测试用例（如同剂量、剂量减半、剂量1/4），计算它们的估计N值，并通过Z分数与参考分布进行比较，从而推断相对剂量。
# 主要输出:
# 1. 'statistical_source_comparison_ransac_spect.png': 一张包含两幅子图的图像，分别显示参考图像样本和第一个测试用例（同剂量）的局部块均值-方差散点图及RANSAC拟合线。
# 2. 控制台输出详细的比较分析，包括：
#    - 参考图像的经验N值分布（均值、标准差）。
#    - 每个测试用例的真实N（如果已知）、RANSAC估算的N。
#    - 估算的N与参考经验N的Z分数。
#    - 基于N估算推断的相对剂量（与参考剂量1相比）及推断的原始最大计数值。
#    - 基于Z分数的剂量相似性判断。
#    - (辅助信息) ROI的变异系数（CV）比较，但主要结论依赖于N的分析。
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import view_as_windows # 用于提取图像块
# from scipy.stats import linregress # 不再直接使用，除非RANSAC中要用
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SPECT_FILE = os.path.join( 'trainsets', 'spectH_raw', 'B001_pure.dat') # 默认SPECT文件路径

def load_spect_anterior_data(file_path):
    """加载SPECT .dat文件并返回前视图。"""
    if not os.path.exists(file_path):
        print(f"错误：SPECT数据文件未找到于 {file_path}")
        return None
    try:
        data = np.fromfile(file_path, dtype=np.float32)
        # 假设数据格式为 [2, 1024, 256] (视图, 高, 宽)
        # 或者根据实际情况调整reshape的维度顺序
        reshaped_data = data.reshape(2, 1024, 256) # 通用格式
        # 如果需要特定方向，例如(视图, 宽, 高) -> (2, 256, 1024)
        # reshaped_data = data.reshape(2, 256, 1024) 
        return reshaped_data[0].astype(np.float32)  # 前视图, 确保为float32
    except Exception as e:
        print(f"加载SPECT数据时发生错误 ({file_path}): {e}")
        return None

def create_base_ideal_image(size=(256, 256), high_dose_max_count=200):
    """生成一个基础的理想图像，包含不同计数值的区域。"""
    image = np.zeros(size, dtype=np.float32)
    half_x, half_y = size[0] // 2, size[1] // 2
    image[0:half_x, 0:half_y] = high_dose_max_count * 0.25  # ROI 0 (左上)
    image[0:half_x, half_y:size[1]] = high_dose_max_count * 0.5   # ROI 1 (右上)
    image[half_x:size[0], 0:half_y] = high_dose_max_count * 0.75  # ROI 2 (左下)
    image[half_x:size[0], half_y:size[1]] = high_dose_max_count * 1.0   # ROI 3 (右下)
    return image

def get_rois(image):
    """从图像中提取四个象限作为ROI。"""
    half_x, half_y = image.shape[0] // 2, image.shape[1] // 2
    roi0 = image[0:half_x, 0:half_y]
    roi1 = image[0:half_x, half_y:image.shape[1]]
    roi2 = image[half_x:image.shape[0], 0:half_y]
    roi3 = image[half_x:image.shape[0], half_y:image.shape[1]]
    return [roi0, roi1, roi2, roi3]

def calculate_cv(patch):
    """计算图像块的变异系数 (CV = std / mean)。"""
    mean = np.mean(patch)
    std = np.std(patch)
    if mean == 0 or np.isclose(mean, 0): # 增加对均值接近0的判断
        return np.nan
    return std / mean

def get_local_patch_stats(image, patch_size=(16, 16), step=16):
    """提取图像的局部块，并计算每个块的均值和方差。"""
    window_shape = patch_size
    patches = view_as_windows(image, window_shape, step=step)
    num_rows, num_cols, _, _ = patches.shape
    patch_stats = []
    for r in range(num_rows):
        for c in range(num_cols):
            patch = patches[r, c, :, :]
            mean = np.mean(patch)
            var = np.var(patch)
            if mean > 1e-6 : # 只考虑均值不为极小值的块，避免除零或不稳定
                patch_stats.append((mean, var))
    return patch_stats

def add_poisson_noise(image):
    """向图像添加泊松噪声。"""
    image_counts = np.maximum(image, 0)
    noisy_image = np.random.poisson(image_counts).astype(np.float32)
    return noisy_image

def normalize_image_by_its_max(img_raw):
    """根据图像自身的最大值将其归一化，返回归一化图像和归一化因子N。"""
    raw_max = img_raw.max()
    if raw_max == 0:
        return np.zeros_like(img_raw), 0 # N=0 表示无法有效归一化或原始图像全黑
    norm_img = img_raw / raw_max
    N = 1.0 / raw_max
    return norm_img, N

def estimate_N_ransac(patch_stats, random_state=None):
    """使用RANSAC从局部块的(均值,方差)对中估计归一化因子N (方差 ≈ N * 均值)。"""
    if not patch_stats or len(patch_stats) < 2:
        return np.nan

    means = np.array([p[0] for p in patch_stats]).reshape(-1, 1)
    variances = np.array([p[1] for p in patch_stats])

    valid_indices = (means.flatten() > 1e-6) & (variances > 1e-9)
    if np.sum(valid_indices) < 2: 
        return np.nan
    
    means_valid = means[valid_indices]
    variances_valid = variances[valid_indices]

    try:
        ransac = RANSACRegressor(random_state=random_state, min_samples=max(2, int(len(means_valid) * 0.1)), residual_threshold=np.percentile(variances_valid, 25))
        ransac.fit(means_valid, variances_valid)
        inlier_mask = ransac.inlier_mask_
        
        if hasattr(ransac.estimator_, 'coef_') and ransac.estimator_.coef_ is not None and len(ransac.estimator_.coef_) > 0:
            estimated_N = ransac.estimator_.coef_[0]
            if estimated_N < 1e-9 :
                ratios = variances_valid[inlier_mask] / means_valid[inlier_mask].flatten()
                return np.nanmedian(ratios) if len(ratios) > 0 else np.nan
            return estimated_N
        else:
            if np.any(inlier_mask) and len(means_valid[inlier_mask]) > 0:
                 ratios = variances_valid[inlier_mask] / means_valid[inlier_mask].flatten()
                 return np.nanmedian(ratios)
            else: 
                 ratios = variances_valid / means_valid.flatten()
                 return np.nanmedian(ratios) if len(ratios) > 0 else np.nan

    except Exception as e:
        ratios = variances_valid / means_valid.flatten()
        return np.nanmedian(ratios) if len(ratios) > 0 else np.nan

def analyze_single_normalized_image(norm_image, patch_size_for_stats, use_ransac=True, random_state=None):
    """分析单张归一化图像，计算其ROI CV和估计的N。"""
    roi_cvs = [calculate_cv(roi) for roi in get_rois(norm_image)]
    patch_stats = get_local_patch_stats(norm_image, patch_size=patch_size_for_stats, step=patch_size_for_stats[0])
    if use_ransac:
        estimated_N = estimate_N_ransac(patch_stats, random_state=random_state)
    return roi_cvs, estimated_N

def print_comparison(case_name, N_true_test, N_estimated_test, CVs_test, 
                     N_ref_true, N_ref_empirical_mean, N_ref_empirical_std, CVs_ref,
                     ref_original_max_count):
    print(f"\n--- 分析案例: {case_name} ---")
    if N_true_test is not None:
        print(f"  测试图像真实 N: {N_true_test:.4e} (最大原始值: {1/N_true_test if N_true_test else np.nan:.1f})")
    print(f"  测试图像估算 N (RANSAC): {N_estimated_test:.4e}")
    print(f"  参考图像真实 N: {N_ref_true:.4e} (原始最大值: {1/N_ref_true if N_ref_true > 0 else np.nan:.1f})")
    print(f"  参考图像经验 N (RANSAC均值): {N_ref_empirical_mean:.4e} (标准差: {N_ref_empirical_std:.2e})")

    if not np.isnan(N_estimated_test) and not np.isnan(N_ref_empirical_mean) and not np.isnan(N_ref_empirical_std):
        z_score = (N_estimated_test - N_ref_empirical_mean) / N_ref_empirical_std if N_ref_empirical_std > 1e-9 else np.inf
        print(f"  估算N与参考经验N的Z分数: {z_score:.2f}")

        relative_dose_estimate = N_ref_empirical_mean / N_estimated_test if N_estimated_test > 1e-9 else np.nan
        print(f"  推断的相对剂量 (与参考剂量1相比): {relative_dose_estimate:.2f}")
        if ref_original_max_count is not None and not np.isnan(relative_dose_estimate):
            inferred_abs_max_count = ref_original_max_count * relative_dose_estimate
            print(f"  (推断的测试图像原始最大计数值约: {inferred_abs_max_count:.1f}) (参考原始最大值: {ref_original_max_count:.1f})")

        if abs(z_score) < 1.0: 
            print("  >> 剂量相似性 (基于N): 非常高 (Z分数 < 1.0)")
        elif abs(z_score) < 2.0: 
            print("  >> 剂量相似性 (基于N): 较高 (1.0 <= Z分数 < 2.0)")
        elif abs(z_score) < 3.0: 
            print("  >> 剂量相似性 (基于N): 中等 (2.0 <= Z分数 < 3.0)")
        else:
            print("  >> 剂量相似性 (基于N): 低 (Z分数 >= 3.0)")
    else:
        print("  >> 剂量相似性 (基于N): 无法进行Z分数比较 (估算值或参考分布缺失)")

    print("\n  ROI CV 比较 (测试 vs 参考) - [注意: ROI CV对复杂图像可能不稳定]:")
    roi_names = ['ROI0', 'ROI1', 'ROI2', 'ROI3']
    cv_diff_ratios_sum = 0
    valid_cv_comparisons = 0
    for i in range(4):
        cv_t = CVs_test[i]
        cv_r = CVs_ref[i]
        if not np.isnan(cv_t) and not np.isnan(cv_r) and cv_r > 1e-6:
            diff_ratio = abs(cv_t - cv_r) / cv_r
            cv_diff_ratios_sum += diff_ratio
            valid_cv_comparisons +=1
            print(f"    {roi_names[i]}: 测试CV={cv_t:.3f}, 参考CV={cv_r:.3f} (相对差异: {diff_ratio*100:.1f}%)")
        else:
            print(f"    {roi_names[i]}: 测试CV={cv_t:.3f}, 参考CV={cv_r:.3f} (一个或多个CV无效)")
    
    if valid_cv_comparisons > 0:
        avg_cv_diff_ratio = cv_diff_ratios_sum / valid_cv_comparisons
        print(f"  平均CV相对差异: {avg_cv_diff_ratio*100:.1f}%")
    print("  >> 初步结论请主要参考基于N估算的Z分数和推断的相对剂量。")

def main():
    patch_analyze_size = (16, 16)
    spect_file_path = DEFAULT_SPECT_FILE
    num_ref_samples_for_N_dist = 30

    print(f"--- 正在加载理想图像: {spect_file_path} ---")
    base_ideal_anterior_raw = load_spect_anterior_data(spect_file_path)
    if base_ideal_anterior_raw is None: return
    print(f"理想图像加载成功，形状: {base_ideal_anterior_raw.shape}, 原始计数范围: [{base_ideal_anterior_raw.min():.2f}, {base_ideal_anterior_raw.max():.2f}]")

    ideal_ref_raw = base_ideal_anterior_raw.copy()
    ref_original_max_count_approx = ideal_ref_raw.max()

    print("\n--- 正在建立参考标准 (基于加载的SPECT图像) ---")
    _temp_noisy_ref = add_poisson_noise(ideal_ref_raw.copy())
    _, N_ref_true_single_instance = normalize_image_by_its_max(_temp_noisy_ref)
    print(f"参考图像的单实例真实N (仅供参考): {N_ref_true_single_instance:.4e}")

    print(f"\n正在通过 {num_ref_samples_for_N_dist} 次采样估算参考图像的N值分布 (RANSAC)... ")
    N_ref_empirical_samples = []
    CVs_ref_samples_roi0 = []

    for i in range(num_ref_samples_for_N_dist):
        noisy_ref_sample = add_poisson_noise(ideal_ref_raw.copy())
        norm_ref_sample, _ = normalize_image_by_its_max(noisy_ref_sample)
        cvs_s, n_s = analyze_single_normalized_image(norm_ref_sample, patch_analyze_size, use_ransac=True, random_state=i)
        if not np.isnan(n_s):
            N_ref_empirical_samples.append(n_s)
        if not np.isnan(cvs_s[0]):
             CVs_ref_samples_roi0.append(cvs_s[0])

    if not N_ref_empirical_samples:
        print("错误: 未能从参考图像中估算出任何有效的N值 (RANSAC)。脚本终止。")
        return
        
    N_ref_empirical_mean = np.mean(N_ref_empirical_samples)
    N_ref_empirical_std = np.std(N_ref_empirical_samples)
    CVs_ref_last_sample = cvs_s if 'cvs_s' in locals() else [np.nan]*4 

    print(f"参考图像经验N (RANSAC): 均值={N_ref_empirical_mean:.4e}, 标准差={N_ref_empirical_std:.2e} (基于{len(N_ref_empirical_samples)}个样本)")

    common_random_state_for_test_ransac = num_ref_samples_for_N_dist

    noisy_test1_raw = add_poisson_noise(ideal_ref_raw.copy()) 
    norm_test1_img, N_test1_true = normalize_image_by_its_max(noisy_test1_raw)
    CVs_test1, N_test1_estimated = analyze_single_normalized_image(norm_test1_img, patch_analyze_size, use_ransac=True, random_state=common_random_state_for_test_ransac)
    print_comparison("测试1 (同剂量、同场景 - 基于SPECT)", 
                     N_test1_true, N_test1_estimated, CVs_test1, 
                     N_ref_true_single_instance, N_ref_empirical_mean, N_ref_empirical_std, CVs_ref_last_sample,
                     ref_original_max_count_approx)

    ideal_test2_raw = ideal_ref_raw.copy() * 0.5 
    noisy_test2_raw = add_poisson_noise(ideal_test2_raw)
    norm_test2_img, N_test2_true = normalize_image_by_its_max(noisy_test2_raw)
    CVs_test2, N_test2_estimated = analyze_single_normalized_image(norm_test2_img, patch_analyze_size, use_ransac=True, random_state=common_random_state_for_test_ransac+1)
    print_comparison("测试2 (剂量减半、同场景 - 基于SPECT)", 
                     N_test2_true, N_test2_estimated, CVs_test2, 
                     N_ref_true_single_instance, N_ref_empirical_mean, N_ref_empirical_std, CVs_ref_last_sample,
                     ref_original_max_count_approx)

    ideal_test3_raw = ideal_ref_raw.copy() * 0.25
    noisy_test3_raw = add_poisson_noise(ideal_test3_raw)
    norm_test3_img, N_test3_true = normalize_image_by_its_max(noisy_test3_raw)
    CVs_test3, N_test3_estimated = analyze_single_normalized_image(norm_test3_img, patch_analyze_size, use_ransac=True, random_state=common_random_state_for_test_ransac+2)
    print_comparison("测试3 (剂量1/4、同场景 - 基于SPECT)", 
                     N_test3_true, N_test3_estimated, CVs_test3, 
                     N_ref_true_single_instance, N_ref_empirical_mean, N_ref_empirical_std, CVs_ref_last_sample,
                     ref_original_max_count_approx)
    
    fig, axs = plt.subplots(1, 2, figsize=(14, 7)) 
    fig.suptitle("均值-方差特性与RANSAC拟合 (基于SPECT数据)", fontsize=16)

    if 'norm_ref_sample' in locals() and norm_ref_sample is not None:
        patch_stats_ref = get_local_patch_stats(norm_ref_sample, patch_size=patch_analyze_size, step=patch_analyze_size[0]//2) 
        if patch_stats_ref:
            means_ref = np.array([p[0] for p in patch_stats_ref]).reshape(-1,1)
            vars_ref = np.array([p[1] for p in patch_stats_ref])
            valid_plot_indices = (means_ref.flatten() > 1e-5) & (vars_ref > 1e-8) & (vars_ref < 0.2) & (means_ref.flatten() < 0.9)
            
            axs[0].scatter(means_ref[valid_plot_indices], vars_ref[valid_plot_indices], alpha=0.2, label='局部块数据', s=10)
            N_ref_plot_fit = estimate_N_ransac(patch_stats_ref, random_state=common_random_state_for_test_ransac-1) 
            if not np.isnan(N_ref_plot_fit) and len(means_ref[valid_plot_indices]) > 0 :
                 x_fit_coords = np.array([np.min(means_ref[valid_plot_indices]), np.max(means_ref[valid_plot_indices])])
                 y_fit_coords = N_ref_plot_fit * x_fit_coords
                 axs[0].plot(x_fit_coords, y_fit_coords, 'r--', label=f'RANSAC拟合 (N={N_ref_plot_fit:.2e})')
        axs[0].set_xlabel("归一化后局部均值"); axs[0].set_ylabel("归一化后局部方差"); 
        axs[0].set_title(f"参考图像样本 (RANSAC N均值 ≈ {N_ref_empirical_mean:.2e})"); axs[0].legend(); axs[0].grid(True)
        axs[0].set_xlim(left=0, right=max(0.1, np.percentile(means_ref[valid_plot_indices].flatten(),99) if np.any(valid_plot_indices) else 0.1))
        axs[0].set_ylim(bottom=0, top=max(0.01, np.percentile(vars_ref[valid_plot_indices],99) if np.any(valid_plot_indices) else 0.01) )

    if norm_test1_img is not None:
        patch_stats_test1 = get_local_patch_stats(norm_test1_img, patch_size=patch_analyze_size, step=patch_analyze_size[0]//2) 
        if patch_stats_test1:
            means_test1 = np.array([p[0] for p in patch_stats_test1]).reshape(-1,1)
            vars_test1 = np.array([p[1] for p in patch_stats_test1])
            valid_plot_indices_t1 = (means_test1.flatten() > 1e-5) & (vars_test1 > 1e-8) & (vars_test1 < 0.2) & (means_test1.flatten() < 0.9)

            axs[1].scatter(means_test1[valid_plot_indices_t1], vars_test1[valid_plot_indices_t1], alpha=0.2, label='局部块数据', s=10, color='green')
            if not np.isnan(N_test1_estimated) and len(means_test1[valid_plot_indices_t1]) > 0:
                 x_fit_coords = np.array([np.min(means_test1[valid_plot_indices_t1]), np.max(means_test1[valid_plot_indices_t1])])
                 y_fit_coords = N_test1_estimated * x_fit_coords 
                 axs[1].plot(x_fit_coords, y_fit_coords, 'r--', label=f'RANSAC拟合 (N={N_test1_estimated:.2e})')
        axs[1].set_xlabel("归一化后局部均值"); axs[1].set_ylabel("归一化后局部方差"); 
        axs[1].set_title(f"测试图像1 (估算N={N_test1_estimated:.2e})"); axs[1].legend(); axs[1].grid(True)
        axs[1].set_xlim(left=0, right=max(0.1, np.percentile(means_test1[valid_plot_indices_t1].flatten(),99) if np.any(valid_plot_indices_t1) else 0.1))
        axs[1].set_ylim(bottom=0, top=max(0.01, np.percentile(vars_test1[valid_plot_indices_t1],99) if np.any(valid_plot_indices_t1) else 0.01))
    
    plt.tight_layout(rect=[0,0,1,0.95])
    # plot_filename = os.path.join(SCRIPT_DIR, 'statistical_source_comparison_ransac_spect.png')
    # Updated save path
    fig_dir = os.path.join(SCRIPT_DIR, 'fig')
    os.makedirs(fig_dir, exist_ok=True)
    plot_filename = os.path.join(fig_dir, 'statistical_source_comparison_ransac_spect.png')
    plt.savefig(plot_filename)
    print(f"\n示例图已保存到: {plot_filename}")
    plt.close(fig)

if __name__ == '__main__':
    main() 