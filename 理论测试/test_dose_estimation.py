# 该脚本旨在通过比较图像的统计特性（主要是归一化直方图）来反推两幅图像之间的相对剂量因子。
# 假设第二幅图像(N2_norm)的剂量相对于第一幅图像(由I_estimated模拟)的剂量未知，脚本尝试通过搜索找到最佳匹配的剂量因子D。
# 主要输出:
# 1. 'dose_estimation_curve_trueD_*.png': 对于每个测试的真实剂量因子D_true，绘制测试剂量因子D_test与平均直方图失配度（SSD）之间的关系曲线图。
# 2. 'summary_D_true_vs_D_estimated.png': 绘制所有测试中真实剂量因子与估计的最佳剂量因子的对比散点图。
# 3. 控制台输出：
#    - Image 2（当前测试剂量）与Image 1同剂量模拟参考图的统计特性对比，包括直方图SSD和KS检验结果。
#    - 每个真实剂量因子D_true的最佳估计剂量因子D_estimated及其对应的最小失配度。
#    - 所有测试结果的总结表格。
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from skimage.metrics import structural_similarity # 可以备用，但主要用直方图和KS
from skimage.transform import resize # 用于缩放SPECT图像
import os # 用于路径处理

# 设置matplotlib中文字体 (如果需要绘图)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# 尝试构建相对于脚本的 trainsets 路径
DEFAULT_SPECT_FILE = os.path.abspath(os.path.join( 'trainsets', 'spectH_raw', 'B001_pure.dat'))

# --- 辅助函数 ---
def load_spect_anterior_data(file_path):
    """加载SPECT .dat文件并返回前视图。"""
    if not os.path.exists(file_path):
        print(f"错误：SPECT数据文件在主路径未找到: {file_path}")
        # 如果脚本在某个子目录，尝试假设项目根目录结构
        # 例如，如果脚本在 KAIR/scripts/test, trainsets 在 KAIR/trainsets
        # SCRIPT_DIR -> KAIR/scripts/test
        # SCRIPT_DIR/../.. -> KAIR
        alt_path = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', 'trainsets', 'spectH_raw', os.path.basename(file_path)))
        print(f"尝试备用项目路径: {alt_path}")
        if os.path.exists(alt_path):
            file_path = alt_path
        else:
            print(f"错误：SPECT数据文件在主路径和备用项目路径中均未找到。请检查路径。")
            return None
    print(f"从路径加载SPECT文件: {file_path}")
    try:
        data = np.fromfile(file_path, dtype=np.float32)
        reshaped_data = data.reshape(2, 1024, 256) # (视图, 高, 宽)
        return reshaped_data[0].astype(np.float32)  # 前视图
    except Exception as e:
        print(f"加载SPECT数据 ({file_path}) 时发生错误: {e}")
        return None



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
    kurt = stats.kurtosis(image_flat) # Fisher's kurtosis (normal ==> 0)
    return mean, var, skewness, kurt

def compare_images_stats(N_ref_flat, N2_flat, D_true_for_N2_being_tested, bins=50):
    print(f"\n--- 对比 Image 2 (真实D={D_true_for_N2_being_tested:.3f}) 与 Image 1 同剂量模拟参考图 (N_ref_dose1) 的统计特性 ---")
    hist_ref, _ = get_histogram(N_ref_flat, bins=bins)
    hist_N2, _ = get_histogram(N2_flat, bins=bins)
    hist_ssd = np.sum((hist_ref - hist_N2)**2) / bins 
    print(f"直方图SSD (越小越相似): {hist_ssd:.4e}")
    stats_ref = get_descriptive_stats(N_ref_flat)
    stats_N2 = get_descriptive_stats(N2_flat)
    print(f" N_ref_dose1 (模拟D=1): Mean={stats_ref[0]:.3f}, Var={stats_ref[1]:.3f}, Skew={stats_ref[2]:.3f}, Kurt={stats_ref[3]:.3f}")
    print(f" N2_norm (真实D={D_true_for_N2_being_tested:.3f}): Mean={stats_N2[0]:.3f}, Var={stats_N2[1]:.3f}, Skew={stats_N2[2]:.3f}, Kurt={stats_N2[3]:.3f}")
    try:
        ks_stat, ks_p_value = stats.ks_2samp(N_ref_flat, N2_flat)
        print(f"KS检验: Statistic={ks_stat:.4f}, P-value={ks_p_value:.4f}")
        if ks_p_value > 0.05:
            print("  KS检验结论: 不能拒绝 'N2_norm' 与 'N_ref_dose1' 来自相同分布 (可能与D=1情况相似)。")
        else:
            print("  KS检验结论: 拒绝 'N2_norm' 与 'N_ref_dose1' 来自相同分布 (可能与D=1情况显著不同)。")
    except Exception as e:
        print(f"KS检验时发生错误: {e}")
        ks_p_value = 0
    return hist_ssd, ks_p_value

def histogram_ssd(hist1, hist2):
    """计算两个直方图之间的SSD"""
    if len(hist1) != len(hist2):
        raise ValueError("直方图长度必须相同")
    return np.sum((hist1 - hist2)**2) / len(hist1)

# --- 主逻辑 ---
def main():
    num_simulations_for_D = 10 
    spect_file_to_load = DEFAULT_SPECT_FILE
    # resize_to_shape = (128, 64) # 将SPECT图像缩放到此尺寸以加快测试 (高, 宽)
    resize_to_shape = None # 如果不想缩放，设为None，但会很慢

    print("--- 1. 准备图像 ---")
    # 1a. 加载并准备 Image 1 的基础理想图像 (I_orig)
    print(f"尝试加载SPECT文件: {spect_file_to_load}")
    I_orig_loaded = load_spect_anterior_data(spect_file_to_load)
    if I_orig_loaded is None:
        print("无法加载SPECT理想图像，脚本终止。")
        return

    if resize_to_shape is not None:
        print(f"将加载的SPECT图像从 {I_orig_loaded.shape} 缩放到 {resize_to_shape}")
        # 注意：resize 输出的可能是 float64，且范围可能不是0-1，需要检查
        I_orig_resized = resize(I_orig_loaded, resize_to_shape, 
                                anti_aliasing=True, preserve_range=True)
        I_orig = I_orig_resized.astype(np.float32)
    else:
        I_orig = I_orig_loaded.astype(np.float32)
    
    # 确保理想图像计数至少为1，以避免泊松参数为0的问题，同时保留其原始动态范围上限
    I_orig = np.maximum(I_orig, 1.0) 
    print(f"处理后的理想图像 I_orig 形状: {I_orig.shape}, 范围: [{I_orig.min():.1f}, {I_orig.max():.1f}] ")
    
    # --- 后续逻辑与之前类似，使用这个 I_orig ---
    X1_raw = poisson_sample(I_orig)
    N1_norm, min_X1, max_X1 = normalize_min_max(X1_raw)
    X1_reconstructed = reconstruct_from_normalized(N1_norm, min_X1, max_X1)
    I_estimated = np.maximum(X1_reconstructed, 0) 

    print(f"Image 1: 原始理想图像(I_orig)最大值 approx {I_orig.max():.1f}")
    print(f"Image 1: 泊松采样后 X1_raw范围 [{X1_raw.min():.1f}, {X1_raw.max():.1f}]")
    print(f"Image 1: 记录的归一化系数 min_X1={min_X1:.1f}, max_X1={max_X1:.1f}")
    print(f"Image 1: 重建的 X1_reconstructed (Î) 范围 [{I_estimated.min():.1f}, {I_estimated.max():.1f}]")

    # --- 测试不同的真实剂量因子 D_true_for_N2 ---
    D_true_values_to_test = [1.0, 0.8, 0.6, 0.5, 0.4, 0.25, 0.1, 0.05] # 您想测试的真实剂量因子
    results_summary = [] # 用于存储 (D_true, D_estimated, min_mismatch)

    # 定义D的搜索空间 (在循环外定义，因为对所有测试都一样)
    D_search_space = np.concatenate([
        np.linspace(0.01, 0.35, 8, endpoint=False), 
        np.linspace(0.35, 0.65, 7, endpoint=False), 
        np.linspace(0.65, 1.25, 7, endpoint=False), 
        np.linspace(1.25, 2.0, 4)                  
    ])
    D_search_space = np.sort(np.unique(D_search_space))
    print(f"\n将用于估计的D搜索空间 (共{len(D_search_space)}个点): {D_search_space}")

    # 为"同剂量比较"准备一个基于I_orig的参考图像 (N_ref_dose1)
    # 这个参考是针对 "Image 1的剂量水平" 进行的模拟，直接使用 I_orig 以确保基准纯净
    X_ref_for_dose1_comparison = poisson_sample(I_orig) # 使用 I_orig 而不是 I_estimated
    N_ref_for_dose1_comparison, _, _ = normalize_min_max(X_ref_for_dose1_comparison)
    N_ref_for_dose1_comparison_flat = N_ref_for_dose1_comparison.flatten()

    for D_idx, D_true_for_N2 in enumerate(D_true_values_to_test):
        print(f"\n=================================================================")
        print(f"开始测试 {D_idx+1}/{len(D_true_values_to_test)}: Image 2 的真实剂量因子 D_true = {D_true_for_N2:.3f}")
        print(f"=================================================================")

        # 1b. 生成 Image 2 (N2_norm) - 当前测试的"未知"来源
        I_for_N2 = I_orig * D_true_for_N2 
        X2_raw = poisson_sample(I_for_N2)
        N2_norm, _, _ = normalize_min_max(X2_raw) 
        N2_norm_flat = N2_norm.flatten()
        hist_N2_target, _ = get_histogram(N2_norm_flat) # 预计算目标直方图

        print(f"Image 2 (当前测试剂量 {D_true_for_N2:.3f}):")
        print(f"  缩放后理想图像(I_for_N2)最大值 approx {I_for_N2.max():.1f}")
        print(f"  归一化后 N2_norm范围 [{N2_norm.min():.1f}, {N2_norm.max():.1f}]")

        # --- 第一部分逻辑：将当前 N2_norm 与 N_ref_for_dose1_comparison 进行比较 ---
        compare_images_stats(N_ref_for_dose1_comparison_flat, N2_norm_flat, D_true_for_N2)

        # --- 第二部分：反推 Image 2 的剂量因子 D ---
        print(f"\n--- 开始为 D_true={D_true_for_N2:.3f} 反推剂量因子 D ---")
        best_D_estimated_current_run = -1
        min_mismatch_score_current_run = float('inf')
        mismatch_scores_current_run = []

        for D_test in D_search_space:
            current_D_mismatches = []
            lambda_model_base = np.maximum(I_estimated * D_test, 0) 
            for i_sim in range(num_simulations_for_D):
                X_model_sample = poisson_sample(lambda_model_base)
                N_model_sample, _, _ = normalize_min_max(X_model_sample)
                hist_model, _ = get_histogram(N_model_sample.flatten())
                mismatch = histogram_ssd(hist_model, hist_N2_target)
                current_D_mismatches.append(mismatch)
            
            avg_mismatch_for_D_test = np.mean(current_D_mismatches)
            mismatch_scores_current_run.append(avg_mismatch_for_D_test)

            if avg_mismatch_for_D_test < min_mismatch_score_current_run:
                min_mismatch_score_current_run = avg_mismatch_for_D_test
                best_D_estimated_current_run = D_test
        
        results_summary.append((D_true_for_N2, best_D_estimated_current_run, min_mismatch_score_current_run))
        print(f"\n反推完成 (对于 D_true = {D_true_for_N2:.3f}):")
        print(f"  估计的最佳剂量因子 D_estimated: {best_D_estimated_current_run:.3f}")
        print(f"  对应的最小平均失配度: {min_mismatch_score_current_run:.4e}")

        # 为当前D_true绘制失配度曲线
        plt.figure(figsize=(10,6))
        plt.plot(D_search_space, mismatch_scores_current_run, 'o-')
        plt.axvline(D_true_for_N2, color='r', linestyle='--', label=f'真实D = {D_true_for_N2:.2f}')
        if best_D_estimated_current_run != -1:
            plt.axvline(best_D_estimated_current_run, color='g', linestyle=':', label=f'估计D = {best_D_estimated_current_run:.2f}')
            plt.plot(best_D_estimated_current_run, min_mismatch_score_current_run, 'gx', markersize=10, label=f'最小失配度点')
        plt.xlabel("测试剂量因子 (D_test)")
        plt.ylabel("平均失配度 (直方图SSD)")
        plt.title(f"剂量因子估计失配度曲线 (真实 D = {D_true_for_N2:.3f})")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plot_filename = f"dose_estimation_curve_trueD_{D_true_for_N2:.3f}.png"
        plt.savefig(plot_filename)
        print(f"失配度曲线已保存为: {plot_filename}")
        plt.close() # 关闭图像，避免过多窗口

    # --- 打印总结 ---    
    print("\n\n===================== 总结所有测试结果 =====================")
    print("真实剂量因子 (D_true) | 估计剂量因子 (D_est) | 最小失配度")
    print("----------------------|------------------------|----------------")
    for d_true, d_est, mismatch_val in results_summary:
        print(f"{d_true:<21.3f} | {d_est:<22.3f} | {mismatch_val:<14.4e}")
    
    # --- 绘制总结图：真实D vs 估计D ---
    if results_summary:
        d_trues = [r[0] for r in results_summary]
        d_ests = [r[1] for r in results_summary]
        plt.figure(figsize=(8,8))
        plt.scatter(d_trues, d_ests, c='blue', label='估计值')
        plt.plot([min(d_trues), max(d_trues)], [min(d_trues), max(d_trues)], 'r--', label='理想情况 (y=x)')
        plt.xlabel("真实剂量因子 (D_true)")
        plt.ylabel("估计剂量因子 (D_estimated)")
        plt.title("真实剂量因子 vs. 估计剂量因子")
        plt.legend()
        plt.grid(True)
        plt.axis('equal') # 确保x和y轴比例相同
        plt.tight_layout()
        summary_plot_filename = "summary_D_true_vs_D_estimated.png"
        plt.savefig(summary_plot_filename)
        print(f"\n总结对比图已保存为: {summary_plot_filename}")
        plt.close()

if __name__ == '__main__':
    main() 