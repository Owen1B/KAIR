# 该脚本用于生成棋盘格图像作为理想图像，
# 对其进行N次泊松采样，然后对理想图像和所有采样图像进行Anscombe变换并归一化。
# 最终计算并分析以下两组图像间的PSNR, SSIM, VIF指标分布：
# 1. N个采样图像两两之间。
# 2. N个采样图像分别与理想图像之间。
# VIF在Anscombe变换后直接计算，PSNR和SSIM在Anscombe变换并用anscombe(70)归一化后计算。
# PSNR, SSIM, VIF 指标均在图像经过Anscombe变换并用anscombe(230)作为参考进行归一化后计算。
# 结果以统计数据和直方图形式输出。

import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
try:
    from sewar.full_ref import vifp
except ImportError:
    print("警告: sewar库未找到。VIF指标将无法计算。请尝试 'pip install sewar'")
    vifp = None # type: ignore

# 设置中文字体以正确显示标题等
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(SCRIPT_DIR, 'fig')

# --- 配置参数 ---
N_SAMPLES = 10  # 泊松采样的次数
# WIN_SIZE_SSIM: SSIM的窗口大小，必须是奇数且小于图像最小维度
# 对于1024x256的图像, 7或11是常用值
WIN_SIZE_SSIM = 7
BLOCK_SIZE = 32  # VIF计算时的分块大小

def generate_checkerboard(height=1024, width=256, block_size=2, black_val=200.0, white_val=100.0):
    """生成棋盘格图像，黑色区域值为black_val，白色区域值为white_val"""
    rows = height // block_size
    cols = width // block_size
    board = np.zeros((rows, cols), dtype=np.float32)
    board[::2, ::2] = 1
    board[1::2, 1::2] = 1
    # 放大到目标尺寸
    board = np.repeat(np.repeat(board, block_size, axis=0), block_size, axis=1)
    # 裁剪到目标尺寸
    board = board[:height, :width]
    # 根据棋盘格的0/1值分配black_val和white_val
    return np.where(board == 1, black_val, white_val)

def anscombe_transform(image):
    """应用Anscombe变换: 2 * sqrt(x + 3/8)"""
    return 2 * np.sqrt(np.maximum(image, 0) + 3.0/8.0)

def normalize_by_anscombe_ref(image, ref_anscombe_val):
    """使用参考Anscombe值进行归一化。"""
    if ref_anscombe_val == 0:
        print("警告: 参考Anscombe值为0，不进行归一化。")
        return image
    normalized = image / ref_anscombe_val
    # 裁剪到0-1范围
    return normalized

def calculate_block_vif(img1, img2, block_size=16):
    """计算分块VIF"""
    if vifp is None:
        return np.nan
    
    h, w = img1.shape
    vif_values = []
    
    # 计算中间从上往下5块的位置
    center_j = w // 2 - block_size // 2
    start_i = h // 2 - (5 * block_size) // 2
    
    # 只计算中间5块
    for i in range(start_i, start_i + 5 * block_size, block_size):
        # 确保不会超出图像边界
        end_i = min(i + block_size, h)
        end_j = center_j + block_size
        
        # 提取块
        block1 = img1[i:end_i, center_j:end_j]
        block2 = img2[i:end_i, center_j:end_j]
        
        try:
            vif_block = vifp(block1, block2)
            if not np.isnan(vif_block):
                vif_values.append(vif_block)
        except Exception as e:
            print(f"计算块VIF时出错 ({i},{center_j}): {e}")
            continue
    
    if not vif_values:
        return np.nan
    return np.mean(vif_values)

def plot_and_save_histogram(data, title, xlabel, filename):
    """绘制数据直方图并保存。"""
    plt.figure(figsize=(8, 6))
    plt.hist(data, bins=15, color='skyblue', edgecolor='black')
    mean_val = np.mean(data)
    std_val = np.std(data)
    plt.title(f"{title}\n均值: {mean_val:.4f}, 标准差: {std_val:.4f}")
    plt.xlabel(xlabel)
    plt.ylabel('频数')
    plt.grid(axis='y', alpha=0.75)
    
    os.makedirs(FIG_DIR, exist_ok=True)
    save_path = os.path.join(FIG_DIR, filename)
    plt.savefig(save_path)
    print(f"直方图已保存到: {save_path}")
    plt.close()

def main():
    print(f"开始处理脚本: {os.path.basename(__file__)}")
    print(f"泊松采样次数 N_SAMPLES = {N_SAMPLES}")
    speed_factors = [1,2,3,4,5,6,7,8] # 倍率因子 n 从 1 到 8

    # 生成棋盘格理想图像
    print("正在生成棋盘格理想图像...")
    I_ideal_raw_original = generate_checkerboard(1024, 256, 32, 230.0)
    print(f"理想图像生成成功。形状: {I_ideal_raw_original.shape}, 数据类型: {I_ideal_raw_original.dtype}")

    # 计算Anscombe变换和归一化参数
    anscombe_230_ref_val = anscombe_transform(np.array([230.0]))[0]
    anscombe_230_ref_val = 1000.414
    print(f"Anscombe变换的归一化参考值 (A(230.0)): {anscombe_230_ref_val:.4f}")

    # 对原始理想图像进行Anscombe变换和归一化 (只做一次)
    I_ideal_ans_norm_original = normalize_by_anscombe_ref(anscombe_transform(I_ideal_raw_original), anscombe_230_ref_val)
    print(f"原始理想图像 (Anscombe变换后用A(230)归一化) 范围: [{I_ideal_ans_norm_original.min():.4f}, {I_ideal_ans_norm_original.max():.4f}]")

    all_sets_raw_images = {} # 存储每个倍率的原始采样图像集
    all_sets_ans_norm_images = {} # 存储每个倍率的Anscombe变换并归一化后的图像集

    # --- 生成所有倍率的采样图像 ---
    for n_factor in speed_factors:
        print(f"\n--- 正在处理倍率因子 n = {n_factor} ---")
        I_temp = I_ideal_raw_original / n_factor
        
        current_set_raw = []
        current_set_ans_norm = []
        
        print(f"  正在为 n={n_factor} 生成 {N_SAMPLES} 个采样图像...")
        for i in range(N_SAMPLES):
            S_raw_temp = np.random.poisson(I_temp).astype(np.float64)
            S_n_raw = S_raw_temp# 恢复计数
            current_set_raw.append(S_n_raw)
            
            S_n_ans = anscombe_transform(S_n_raw)
            S_n_ans_norm = normalize_by_anscombe_ref(S_n_ans, anscombe_230_ref_val)
            current_set_ans_norm.append(S_n_ans_norm)
            
            if (i+1) % max(1, (N_SAMPLES//4)) == 0:
                print(f"    已生成 {i+1}/{N_SAMPLES} 个 (n={n_factor}) 采样图像...")
        
        all_sets_raw_images[n_factor] = current_set_raw
        all_sets_ans_norm_images[n_factor] = current_set_ans_norm
        print(f"  完成 n={n_factor} 的图像生成和预处理。")

    # 获取基准 Set_1 (Anscombe变换并归一化后)
    Set_1_ans_norm_images = all_sets_ans_norm_images[1]

    # --- VIF 计算 ---
    results_vif = {} # 存储VIF结果: results_vif[n_factor]['type'] = [values]

    print("\n--- 开始VIF计算 ---")
    for n_factor in speed_factors:
        print(f"  计算 VIF for n = {n_factor}")
        results_vif[n_factor] = {
            'vif_set_vs_set': [],      # Set_n vs Set_n
            'vif_set_vs_ideal': [],  # Set_n vs I_ideal_raw_original (processed)
            'vif_set_vs_set1': []    # Set_n vs Set_1 (only for n > 1)
        }
        
        current_set_ans_norm = all_sets_ans_norm_images[n_factor]

        # a) VIF(Set_n vs Set_n)
        if N_SAMPLES >= 2:
            for i in range(N_SAMPLES):
                for j in range(i + 1, N_SAMPLES):
                    img1 = current_set_ans_norm[i]
                    img2 = current_set_ans_norm[j]
                    try:
                        vif_val = calculate_block_vif(img1, img2, BLOCK_SIZE)
                        results_vif[n_factor]['vif_set_vs_set'].append(vif_val)
                    except Exception as e:
                        print(f"    VIF(S{n_factor}[{i}] vs S{n_factor}[{j}]) 错误: {e}")
                        results_vif[n_factor]['vif_set_vs_set'].append(np.nan)
        
        # b) VIF(Set_n vs I_ideal_raw_original)
        for i in range(N_SAMPLES):
            img_s = current_set_ans_norm[i]
            try:
                vif_val = calculate_block_vif(I_ideal_ans_norm_original, img_s, BLOCK_SIZE)
                results_vif[n_factor]['vif_set_vs_ideal'].append(vif_val)
            except Exception as e:
                print(f"    VIF(S{n_factor}[{i}] vs Ideal) 错误: {e}")
                results_vif[n_factor]['vif_set_vs_ideal'].append(np.nan)

        # c) VIF(Set_n vs Set_1) - for n > 1
        if n_factor > 1:
            for i in range(N_SAMPLES): # Iterating through Set_n
                for k in range(N_SAMPLES): # Iterating through Set_1
                    img_set_n = current_set_ans_norm[i]
                    img_set_1 = Set_1_ans_norm_images[k]
                    try:
                        vif_val = calculate_block_vif(img_set_1,img_set_n, BLOCK_SIZE)
                        results_vif[n_factor]['vif_set_vs_set1'].append(vif_val)
                    except Exception as e:
                        print(f"    VIF(S{n_factor}[{i}] vs S1[{k}]) 错误: {e}")
                        results_vif[n_factor]['vif_set_vs_set1'].append(np.nan)
        print(f"  完成 VIF for n = {n_factor}")

    # --- 结果统计与打印 ---
    print("\n--- VIF 指标统计结果 ---")
    base_filename_prefix = "checkerboard_MultiSpeedAnalysis"

    summary_stats = {} # To store mean/std for plotting

    for n_factor in speed_factors:
        print(f"\n倍率因子 n = {n_factor}:")
        summary_stats[n_factor] = {}
        
        for vif_type in ['vif_set_vs_set', 'vif_set_vs_ideal', 'vif_set_vs_set1']:
            if not results_vif[n_factor][vif_type] and vif_type == 'vif_set_vs_set1' and n_factor == 1:
                # Skip VIF(Set1 vs Set1) if n=1 and this specific type as it's not applicable by definition here
                continue
            if not results_vif[n_factor][vif_type]:
                 print(f"  {vif_type}: 无数据。")
                 summary_stats[n_factor][vif_type] = {'mean': np.nan, 'std': np.nan}
                 continue

            valid_data = [x for x in results_vif[n_factor][vif_type] if not np.isnan(x)]
            if not valid_data:
                print(f"  {vif_type}: 无有效数据 (均为NaN)。")
                summary_stats[n_factor][vif_type] = {'mean': np.nan, 'std': np.nan}
                continue
            
            mean_val = np.mean(valid_data)
            std_val = np.std(valid_data)
            count = len(valid_data)
            
            title_map = {
                'vif_set_vs_set': f'VIF (Set_{n_factor} vs Set_{n_factor})',
                'vif_set_vs_ideal': f'VIF (Set_{n_factor} vs Ideal)',
                'vif_set_vs_set1': f'VIF (Set_{n_factor} vs Set_1)'
            }
            print(f"  {title_map.get(vif_type, vif_type)}:")
            print(f"    数量: {count}")
            print(f"    均值: {mean_val:.4f}")
            print(f"    标准差: {std_val:.4f}")
            summary_stats[n_factor][vif_type] = {'mean': mean_val, 'std': std_val}

            # Plot histogram for each n and each VIF type
            hist_title = f"{title_map.get(vif_type, vif_type)}\n(n={n_factor}, Anscombe+A(230) Norm)"
            hist_xlabel = "VIF 值"
            hist_filename = f"{base_filename_prefix}_n{n_factor}_{vif_type}_hist.png"
            if count > 0 : # only plot if there's data
                 plot_and_save_histogram(valid_data, hist_title, hist_xlabel, hist_filename)
    
    # --- 绘制汇总图表 ---
    print("\n正在生成汇总图表...")
    plt.figure(figsize=(12, 8))
    
    n_factors_list = list(speed_factors)
    
    # Plot VIF(Set_n vs Set_n)
    means_set_vs_set = [summary_stats[n]['vif_set_vs_set']['mean'] for n in n_factors_list]
    stds_set_vs_set = [summary_stats[n]['vif_set_vs_set']['std'] for n in n_factors_list]
    plt.errorbar(n_factors_list, means_set_vs_set, yerr=stds_set_vs_set, marker='o', linestyle='-', label='均值 VIF (Set_n vs Set_n)')
    
    # Plot VIF(Set_n vs Ideal)
    means_set_vs_ideal = [summary_stats[n]['vif_set_vs_ideal']['mean'] for n in n_factors_list]
    stds_set_vs_ideal = [summary_stats[n]['vif_set_vs_ideal']['std'] for n in n_factors_list]
    plt.errorbar(n_factors_list, means_set_vs_ideal, yerr=stds_set_vs_ideal, marker='s', linestyle='--', label='均值 VIF (Set_n vs Ideal)')
    
    # Plot VIF(Set_n vs Set_1) - starts from n=2, so adjust x-axis data
    n_factors_for_set1_comp = [n for n in n_factors_list if n > 1]
    if n_factors_for_set1_comp: # only plot if there are factors > 1
        means_set_vs_set1 = [summary_stats[n]['vif_set_vs_set1']['mean'] for n in n_factors_for_set1_comp]
        stds_set_vs_set1 = [summary_stats[n]['vif_set_vs_set1']['std'] for n in n_factors_for_set1_comp]
        plt.errorbar(n_factors_for_set1_comp, means_set_vs_set1, yerr=stds_set_vs_set1, marker='^', linestyle=':', label='均值 VIF (Set_n vs Set_1)')

    plt.xlabel('倍率因子 n (模拟扫描速度)')
    plt.ylabel('平均 VIF 值')
    plt.title('不同模拟扫描速度下的平均VIF比较\n(图像预处理: Anscombe * sqrt(2), then normalize with A(230))')
    plt.xticks(n_factors_list)
    plt.legend()
    plt.grid(True)
    
    summary_plot_filename = os.path.join(FIG_DIR, f"{base_filename_prefix}_summary_vif_vs_speed.png")
    os.makedirs(FIG_DIR, exist_ok=True)
    plt.savefig(summary_plot_filename)
    print(f"汇总图表已保存到: {summary_plot_filename}")
    plt.close()

    print("\n脚本执行完毕。")

    # Visualizing some example images (optional, can be expanded)
    # For example, show ideal, one from Set_1, one from Set_4, one from Set_8
    print("\n正在生成示例图像可视化...")
    images_to_plot_examples = []
    titles_examples = []

    images_to_plot_examples.append(I_ideal_ans_norm_original)
    titles_examples.append('理想图像\n(Anscombe+A(230) Norm)')

    if 1 in all_sets_ans_norm_images and all_sets_ans_norm_images[1]:
        images_to_plot_examples.append(all_sets_ans_norm_images[1][0]) # First sample from Set_1
        titles_examples.append('Set_1 采样 0\n(Anscombe+A(230) Norm)')
    
    example_n_factors = [1,2,3,4,5,6,7,8]
    for n_ex in example_n_factors:
        if n_ex in all_sets_ans_norm_images and all_sets_ans_norm_images[n_ex]:
             images_to_plot_examples.append(all_sets_ans_norm_images[n_ex][0]) # First sample
             titles_examples.append(f'Set_{n_ex} 采样 0\n(Anscombe+A(230) Norm)')

    num_images_to_show = len(images_to_plot_examples)
    if num_images_to_show > 0:
        global_vmin = np.min([img.min() for img in images_to_plot_examples if img is not None])
        global_vmax = np.max([img.max() for img in images_to_plot_examples if img is not None])
        
        fig_compare, axes_compare = plt.subplots(1, num_images_to_show, figsize=(5.5 * num_images_to_show, 5.5))
        if num_images_to_show == 1: axes_compare = [axes_compare]

        for i in range(num_images_to_show):
            ax = axes_compare[i]
            if images_to_plot_examples[i] is not None:
                im = ax.imshow(images_to_plot_examples[i], cmap='gray', vmin=global_vmin, vmax=global_vmax)
                fig_compare.colorbar(im, ax=ax, shrink=0.7)
            ax.set_title(titles_examples[i])
            ax.axis('off')
        
        fig_compare.suptitle('示例图像对比 (Anscombe变换后, 用A(230)归一化, 共享颜色范围)', fontsize=14)
        plt.tight_layout(rect=[0, 0.05, 1, 0.93])
        
        os.makedirs(FIG_DIR, exist_ok=True)
        comparison_filename = os.path.join(FIG_DIR, f'{base_filename_prefix}_example_images_comparison.png')
        plt.savefig(comparison_filename)
        print(f"示例对比图像已保存到: {comparison_filename}")
        plt.close(fig_compare)
    else:
        print("没有足够的图像进行可视化对比。")

if __name__ == '__main__':
    main() 