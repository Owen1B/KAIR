"""
分析不同噪声水平下图像质量评估指标的变化

该脚本用于分析SPECT图像在不同噪声水平下的图像质量评估指标（LPIPS、PSNR、SSIM、MS-SSIM、FID）的变化。
主要功能：
1. 对比原始带噪图像与理想图像在不同噪声水平下的质量差异
2. 对比BM3D降噪后的图像与原始带噪图像在不同噪声水平下的质量差异
3. 生成对比图表，展示LPIPS、PSNR、SSIM、MS-SSIM、FID随噪声水平的变化趋势

使用方法：
1. 确保已安装所需依赖：torch, lpips, numpy, matplotlib, tqdm, scikit-image, pytorch-fid
2. 配置数据目录路径
3. 运行脚本：python analyze_lpips_vs_noise_level.py

"""

import os
import sys
# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import torch
import lpips
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("警告: pandas未安装，将跳过CSV数据保存功能")

# 设置 matplotlib 使用支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300  # 提高图像质量

# 1. LPIPS Model Initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
    loss_fn_alex = lpips.LPIPS(net='alex').to(device)
    print(f"LPIPS模型将在 {device} 上运行")
except Exception as e:
    print(f"无法初始化LPIPS模型: {e}. 请确保lpips已正确安装并且可以访问预训练模型.")
    exit()

# 2. Helper Functions
def load_dat_file(file_path: str) -> np.ndarray | None:
    """加载并重塑.dat文件.
    返回 (2, 1024, 256) 的 numpy 数组, 如果失败则返回 None.
    """
    try:
        data = np.fromfile(file_path, dtype=np.float32)
        if data.size != 2 * 1024 * 256:
            print(f"警告: 文件 {os.path.basename(file_path)} 的大小 ({data.size}) 不是预期的 2*1024*256。")
            return None
        return data.reshape(2, 1024, 256)
    except FileNotFoundError:
        print(f"错误: 文件 {file_path} 未找到.")
        return None
    except Exception as e:
        print(f"读取或重塑文件 {os.path.basename(file_path)} 时出错: {e}")
        return None

def prepare_for_lpips(img_np: np.ndarray, data_range: float, device: torch.device) -> torch.Tensor:
    """将NumPy图像转换为LPIPS所需的PyTorch张量格式。"""
    img_normalized = img_np / data_range * 2.0 - 1.0 # 归一化到 [-1, 1]
    img_tensor = torch.from_numpy(img_normalized.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    return img_tensor.to(device)

def analyze_single_file_lpips_vs_noise(
    original_noisy_file_path: str,
    reference_file_path: str,
    output_plot_dir: str,
    noise_scale_factors: list[float],
    num_repetitions: int,
    comparison_name: str
):
    """对单个文件进行分析并绘图.
    
    Args:
        original_noisy_file_path: 原始带噪图像路径
        reference_file_path: 参考图像路径（理想图或BM3D降噪图）
        output_plot_dir: 输出图表目录
        noise_scale_factors: 噪声缩放因子列表
        num_repetitions: 每个噪声水平的重复次数
        comparison_name: 对比名称（用于图表标题和文件名）
    """
    original_filename = os.path.basename(original_noisy_file_path)
    print(f"\n开始处理文件: {original_filename}")

    original_noisy_data = load_dat_file(original_noisy_file_path)
    reference_data = load_dat_file(reference_file_path)

    if original_noisy_data is None or reference_data is None:
        print(f"无法加载所需数据文件，跳过 {original_filename}.")
        return

    view_names = ['Anterior']

    for view_idx, view_name in enumerate(view_names):
        print(f"  处理视图: {view_name}")
        original_view = original_noisy_data[view_idx].astype(np.float32)
        reference_view = reference_data[view_idx].astype(np.float32)

        # 确定LPIPS归一化的data_range (基于原始带噪图像)
        data_r = np.max(original_view)
        if data_r == 0: # 防止除以零
            print(f"    警告: 视图 {view_name} 的原始图像最大值为0，跳过此视图的LPIPS计算.")
            continue
        
        original_view_lpips_tensor = prepare_for_lpips(original_view, data_r, device)
        
        avg_lpips_scores_for_view = []
        avg_psnr_scores_for_view = []
        avg_ssim_scores_for_view = []
        
        # 添加标准差列表
        std_lpips_scores_for_view = []
        std_psnr_scores_for_view = []
        std_ssim_scores_for_view = []
        
        for scale_factor in tqdm(noise_scale_factors, desc=f"    {view_name} - 缩放因子"):
            lambda_img = np.maximum(0, reference_view * scale_factor)
            current_scale_lpips_values = []
            current_scale_psnr_values = []
            current_scale_ssim_values = []
            
            for _ in range(num_repetitions):
                renoised_img = np.random.poisson(lambda_img).astype(np.float32)/scale_factor
                renoised_img_lpips_tensor = prepare_for_lpips(renoised_img, data_r, device)
                
                with torch.no_grad():
                    lpips_score = loss_fn_alex(original_view_lpips_tensor, renoised_img_lpips_tensor).item()
                current_scale_lpips_values.append(lpips_score)
                
                # 计算PSNR、SSIM
                psnr_score = psnr(original_view, renoised_img, data_range=data_r)
                ssim_score = ssim(original_view, renoised_img, data_range=data_r)
                
                current_scale_psnr_values.append(psnr_score)
                current_scale_ssim_values.append(ssim_score)
            
            # 计算平均值和标准差
            avg_lpips_scores_for_view.append(np.mean(current_scale_lpips_values))
            avg_psnr_scores_for_view.append(np.mean(current_scale_psnr_values))
            avg_ssim_scores_for_view.append(np.mean(current_scale_ssim_values))
            
            std_lpips_scores_for_view.append(np.std(current_scale_lpips_values))
            std_psnr_scores_for_view.append(np.std(current_scale_psnr_values))
            std_ssim_scores_for_view.append(np.std(current_scale_ssim_values))

        # 创建四个子图
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 24))
        
        # 设置x轴刻度
        x_ticks = noise_scale_factors
        x_labels = [f'{x:.2f}' for x in x_ticks]
        
        # LPIPS图（带误差条）
        ax1.errorbar(noise_scale_factors, avg_lpips_scores_for_view, 
                    yerr=std_lpips_scores_for_view, marker='o', linestyle='-', 
                    linewidth=2, capsize=5, capthick=2, elinewidth=1)
        ax1.set_xscale('log')
        ax1.set_title(f'LPIPS vs. Noise Scale Factor - {view_name} View\n{comparison_name}\nFile: {original_filename}', pad=20)
        ax1.set_xlabel('Noise Scale Factor', labelpad=10)
        ax1.set_ylabel('Average LPIPS ± Std', labelpad=10)
        ax1.grid(True, which='both', linestyle='--', alpha=0.7)
        ax1.set_xticks(x_ticks)
        ax1.set_xticklabels(x_labels, rotation=45)
        
        # PSNR图（带误差条）
        ax2.errorbar(noise_scale_factors, avg_psnr_scores_for_view, 
                    yerr=std_psnr_scores_for_view, marker='o', linestyle='-', 
                    color='green', linewidth=2, capsize=5, capthick=2, elinewidth=1)
        ax2.set_xscale('log')
        ax2.set_title('PSNR vs. Noise Scale Factor', pad=20)
        ax2.set_xlabel('Noise Scale Factor', labelpad=10)
        ax2.set_ylabel('Average PSNR (dB) ± Std', labelpad=10)
        ax2.grid(True, which='both', linestyle='--', alpha=0.7)
        ax2.set_xticks(x_ticks)
        ax2.set_xticklabels(x_labels, rotation=45)
        
        # SSIM图（带误差条）
        ax3.errorbar(noise_scale_factors, avg_ssim_scores_for_view, 
                    yerr=std_ssim_scores_for_view, marker='o', linestyle='-', 
                    color='red', linewidth=2, capsize=5, capthick=2, elinewidth=1)
        ax3.set_xscale('log')
        ax3.set_title('SSIM vs. Noise Scale Factor', pad=20)
        ax3.set_xlabel('Noise Scale Factor', labelpad=10)
        ax3.set_ylabel('Average SSIM ± Std', labelpad=10)
        ax3.grid(True, which='both', linestyle='--', alpha=0.7)
        ax3.set_xticks(x_ticks)
        ax3.set_xticklabels(x_labels, rotation=45)
        
        plt.tight_layout()
        
        plot_filename = f"metrics_vs_noise_scale_{view_name.lower()}_{comparison_name}_{os.path.splitext(original_filename)[0]}.png"
        plot_filepath = os.path.join(output_plot_dir, plot_filename)
        try:
            plt.savefig(plot_filepath, bbox_inches='tight', dpi=300)
            print(f"    图表已保存到: {plot_filepath}")
        except Exception as e:
            print(f"    保存图表 {plot_filepath} 失败: {e}")
        plt.close()
        
        # 保存数值数据到CSV文件
        csv_filename = f"metrics_data_{view_name.lower()}_{comparison_name}_{os.path.splitext(original_filename)[0]}.csv"
        csv_filepath = os.path.join(output_plot_dir, csv_filename)
        
        if PANDAS_AVAILABLE:
            try:
                data_dict = {
                    'Noise_Scale_Factor': noise_scale_factors,
                    'LPIPS_Mean': avg_lpips_scores_for_view,
                    'LPIPS_Std': std_lpips_scores_for_view,
                    'PSNR_Mean': avg_psnr_scores_for_view,
                    'PSNR_Std': std_psnr_scores_for_view,
                    'SSIM_Mean': avg_ssim_scores_for_view,
                    'SSIM_Std': std_ssim_scores_for_view
                }
                
                df = pd.DataFrame(data_dict)
                df.to_csv(csv_filepath, index=False, float_format='%.6f')
                print(f"    数值数据已保存到: {csv_filepath}")
                
            except Exception as e:
                print(f"    保存CSV数据失败: {e}")
        
        # 打印部分关键数据点（无论是否有pandas）
        print(f"    关键数据点预览 ({view_name}):")
        print(f"    {'Scale':<8} {'LPIPS':<12} {'PSNR':<12} {'SSIM':<12}")
        print(f"    {'':<8} {'Mean±Std':<12} {'Mean±Std':<12} {'Mean±Std':<12}")
        print("    " + "-" * 60)
        
        # 显示几个关键点的数据
        key_indices = [0, len(noise_scale_factors)//4, len(noise_scale_factors)//2, 
                      3*len(noise_scale_factors)//4, -1]
        for i in key_indices:
            if i < len(noise_scale_factors):
                scale = noise_scale_factors[i]
                lpips_val = f"{avg_lpips_scores_for_view[i]:.3f}±{std_lpips_scores_for_view[i]:.3f}"
                psnr_val = f"{avg_psnr_scores_for_view[i]:.2f}±{std_psnr_scores_for_view[i]:.2f}"
                ssim_val = f"{avg_ssim_scores_for_view[i]:.3f}±{std_ssim_scores_for_view[i]:.3f}"
                print(f"    {scale:<8.2f} {lpips_val:<12} {psnr_val:<12} {ssim_val:<12}")
        
        print()

def main():
    # 3. Configuration
    config = {
        # 'original_noisy_dir': "SPECTdatasets/spectH_clinical",
        # 'ideal_dir': "SPECTdatasets/spectH_XCAT_ideal_1x",
        # 'bm3d_dir': "SPECTdatasets/spectH_clinical_bm3d_1x",
        'original_noisy_dir': "SPECTdatasets/spectH_XCAT_poisson_1x",
        'ideal_dir': "SPECTdatasets/spectH_XCAT_ideal_1x",
        'bm3d_dir': "SPECTdatasets/spectH_XCAT_bm3d_1x",


        'output_plot_dir': "理论测试/fig/lpips_noise_analysis_results",
        'num_repetitions': 100,  # 每个噪声水平重复次数
        'noise_scale_factors': [0.1, 0.125, 0.25, 0.5, 0.75, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4, 5, 8, 10]  # 噪声水平（缩放因子）
    }

    os.makedirs(config['output_plot_dir'], exist_ok=True)
    print(f"输出图表将保存到: {config['output_plot_dir']}")

    # 文件选择逻辑
    try:
        noisy_files = sorted([f for f in os.listdir(config['original_noisy_dir']) if f.lower().endswith('.dat')])
    except FileNotFoundError:
        print(f"错误: 原始带噪数据目录 '{config['original_noisy_dir']}' 未找到.")
        return

    if not noisy_files:
        print(f"在目录 '{config['original_noisy_dir']}' 中未找到 .dat 文件.")
        return

    # 只处理第一个文件
    first_noisy_file_name = noisy_files[0]
    
    # 分析理想图对比
    original_noisy_file_path = os.path.join(config['original_noisy_dir'], first_noisy_file_name)
    ideal_file_path = os.path.join(config['ideal_dir'], first_noisy_file_name)
    
    if os.path.exists(ideal_file_path):
        analyze_single_file_lpips_vs_noise(
            original_noisy_file_path,
            ideal_file_path,
            config['output_plot_dir'],
            config['noise_scale_factors'],
            config['num_repetitions'],
            "Ideal_Comparison"
        )
    else:
        print(f"警告: 理想图文件 '{ideal_file_path}' 未找到，跳过理想图对比分析。")
    
    # 分析BM3D对比
    bm3d_file_path = os.path.join(config['bm3d_dir'], first_noisy_file_name)
    if os.path.exists(bm3d_file_path):
        analyze_single_file_lpips_vs_noise(
            original_noisy_file_path,
            bm3d_file_path,
            config['output_plot_dir'],
            config['noise_scale_factors'],
            config['num_repetitions'],
            "BM3D_Comparison"
        )
    else:
        print(f"警告: BM3D降噪文件 '{bm3d_file_path}' 未找到，跳过BM3D对比分析。")

    print("\n分析完成.")

if __name__ == "__main__":
    main() 