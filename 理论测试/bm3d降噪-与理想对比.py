import os
import numpy as np
import matplotlib.pyplot as plt
import bm3d
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import torch
import lpips

# 设置 matplotlib 使用支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 初始化LPIPS模型 (全局，以避免重复加载)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loss_fn_alex = lpips.LPIPS(net='alex').to(device)
print(f"LPIPS模型将在 {device} 上运行")

def anscombe_transform(x: np.ndarray) -> np.ndarray:
    """Anscombe变换: 2 * sqrt(x + 3/8)"""
    return 2 * np.sqrt(np.maximum(0, x) + 3/8)

def inverse_anscombe_transform(y: np.ndarray) -> np.ndarray:
    """Anscombe反变换: (y/2)^2 - 3/8"""
    term = y / 2.0
    return np.maximum(0, term**2 - 3/8)

def prepare_for_lpips(img_np: np.ndarray, data_range: float, device: torch.device) -> torch.Tensor:
    """将NumPy图像转换为LPIPS所需的PyTorch张量格式。"""
    # LPIPS期望输入范围是 [-1, 1]
    # 确保data_range有效，避免除以零
    current_data_range = data_range if data_range > 1e-5 else 1.0 
    img_normalized = img_np / current_data_range * 2 - 1
    img_tensor = torch.from_numpy(img_normalized.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    return img_tensor.to(device)

def process_and_visualize_ideal_dat_file(ideal_dat_file_path: str, output_base_dir: str):
    """
    处理单个理想DAT文件，模拟噪声、执行降噪、计算指标并可视化。
    """
    filename = os.path.basename(ideal_dat_file_path)
    print(f"\n--- 开始处理理想文件: {filename} ---") # 增加换行和分隔符

    try:
        data = np.fromfile(ideal_dat_file_path, dtype=np.float32)
        if data.size != 2 * 1024 * 256:
            print(f"警告: 文件 {filename} 的大小不是预期的 2*1024*256。跳过此文件。")
            return
        data = data.reshape(2, 1024, 256)
    except Exception as e:
        print(f"读取或重塑文件 {filename} 时出错: {e}。跳过此文件。")
        return

    view_names = ['前位 (Anterior)', '后位 (Posterior)']
    
    for i, view_name_full in enumerate(view_names):
        print(f"\n  --- 视图: {view_name_full} ---")
        view_name_short = 'anterior' if 'Anterior' in view_name_full else 'posterior'
        ideal_view = data[i]

        # 1. 添加泊松噪声到理想图像
        noisy_view = np.random.poisson(np.maximum(0, ideal_view)).astype(np.float32)

        # 2. Anscombe 变换 (对带噪图像)
        anscombe_noisy_view = anscombe_transform(noisy_view)

        # 3. BM3D 降噪
        sigma_psd = 1.0
        denoised_anscombe_view = bm3d.bm3d(
            anscombe_noisy_view.astype(np.float32),
            sigma_psd=sigma_psd,
            stage_arg=bm3d.BM3DStages.ALL_STAGES,
            profile='np'
        )

        # 4. 反 Anscombe 变换
        final_denoised_view = inverse_anscombe_transform(denoised_anscombe_view)

        # 5. 给最终降噪图像再次添加泊松噪声
        positive_denoised_view = np.maximum(0, final_denoised_view)
        denoised_plus_poisson_view = np.random.poisson(positive_denoised_view).astype(np.float32)

        # --- 指标计算 --- 
        vmax_ideal = np.max(ideal_view)
        data_r_ideal = vmax_ideal if vmax_ideal > 1e-5 else 1.0

        print("  指标评估:")
        # A. noisy_view vs ideal_view
        psnr_noisy_vs_ideal = peak_signal_noise_ratio(ideal_view, noisy_view, data_range=data_r_ideal)
        ssim_noisy_vs_ideal = structural_similarity(ideal_view, noisy_view, data_range=data_r_ideal, win_size=7, channel_axis=None)
        ideal_lpips_tensor = prepare_for_lpips(ideal_view.astype(np.float32), data_r_ideal, device)
        noisy_lpips_tensor_vs_ideal = prepare_for_lpips(noisy_view.astype(np.float32), data_r_ideal, device)
        with torch.no_grad():
            lpips_noisy_vs_ideal = loss_fn_alex(ideal_lpips_tensor, noisy_lpips_tensor_vs_ideal).item()
        print(f"    1. Ideal+Poisson vs Ideal:")
        print(f"       PSNR: {psnr_noisy_vs_ideal:.4f} dB, SSIM: {ssim_noisy_vs_ideal:.4f}, LPIPS: {lpips_noisy_vs_ideal:.4f}")

        # B. final_denoised_view vs ideal_view
        psnr_denoised_vs_ideal = peak_signal_noise_ratio(ideal_view, final_denoised_view, data_range=data_r_ideal)
        ssim_denoised_vs_ideal = structural_similarity(ideal_view, final_denoised_view, data_range=data_r_ideal, win_size=7, channel_axis=None)
        denoised_lpips_tensor_vs_ideal = prepare_for_lpips(final_denoised_view.astype(np.float32), data_r_ideal, device)
        with torch.no_grad():
            lpips_denoised_vs_ideal = loss_fn_alex(ideal_lpips_tensor, denoised_lpips_tensor_vs_ideal).item()
        print(f"    2. Denoised vs Ideal:")
        print(f"       PSNR: {psnr_denoised_vs_ideal:.4f} dB, SSIM: {ssim_denoised_vs_ideal:.4f}, LPIPS: {lpips_denoised_vs_ideal:.4f}")

        # C. denoised_plus_poisson_view vs ideal_view
        psnr_denoisedP_vs_ideal = peak_signal_noise_ratio(ideal_view, denoised_plus_poisson_view, data_range=data_r_ideal)
        ssim_denoisedP_vs_ideal = structural_similarity(ideal_view, denoised_plus_poisson_view, data_range=data_r_ideal, win_size=7, channel_axis=None)
        denoisedP_lpips_tensor_vs_ideal = prepare_for_lpips(denoised_plus_poisson_view.astype(np.float32), data_r_ideal, device)
        with torch.no_grad():
            lpips_denoisedP_vs_ideal = loss_fn_alex(ideal_lpips_tensor, denoisedP_lpips_tensor_vs_ideal).item()
        print(f"    3. Denoised+Poisson vs Ideal:")
        print(f"       PSNR: {psnr_denoisedP_vs_ideal:.4f} dB, SSIM: {ssim_denoisedP_vs_ideal:.4f}, LPIPS: {lpips_denoisedP_vs_ideal:.4f}")

        # D. denoised_plus_poisson_view vs noisy_view (Ideal+Poisson)
        vmax_noisy = np.max(noisy_view)
        data_r_noisy = vmax_noisy if vmax_noisy > 1e-5 else 1.0
        psnr_denoisedP_vs_noisy = peak_signal_noise_ratio(noisy_view, denoised_plus_poisson_view, data_range=data_r_noisy)
        ssim_denoisedP_vs_noisy = structural_similarity(noisy_view, denoised_plus_poisson_view, data_range=data_r_noisy, win_size=7, channel_axis=None)
        noisy_lpips_tensor_vs_noisy = prepare_for_lpips(noisy_view.astype(np.float32), data_r_noisy, device)
        denoisedP_lpips_tensor_vs_noisy = prepare_for_lpips(denoised_plus_poisson_view.astype(np.float32), data_r_noisy, device)
        with torch.no_grad():
            lpips_denoisedP_vs_noisy = loss_fn_alex(noisy_lpips_tensor_vs_noisy, denoisedP_lpips_tensor_vs_noisy).item()
        print(f"    4. Denoised+Poisson vs Ideal+Poisson:")
        print(f"       PSNR: {psnr_denoisedP_vs_noisy:.4f} dB, SSIM: {ssim_denoisedP_vs_noisy:.4f}, LPIPS: {lpips_denoisedP_vs_noisy:.4f}")
        # --- 指标计算结束 ---

        # --- 可视化 (1x4) ---
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        fig.suptitle(f"理想图降噪模拟: {filename} - 视图: {view_name_full}", fontsize=16)

        # 子图1: 原始理想图像
        im0 = axes[0].imshow(ideal_view, cmap='gray', vmax=vmax_ideal)
        axes[0].set_title(f"理想图像 (Ideal)\n总计数: {np.sum(ideal_view):.0f}")
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        # 子图2: 理想图 + 泊松噪声
        im1 = axes[1].imshow(noisy_view, cmap='gray', vmax=vmax_ideal)
        title_noisy = (
            f"理想图+泊松噪声 (Ideal+Poisson)\n"
            f"总计数: {np.sum(noisy_view):.0f}\n"
            f"PSNR: {psnr_noisy_vs_ideal:.2f} dB, SSIM: {ssim_noisy_vs_ideal:.4f}\n"
            f"LPIPS: {lpips_noisy_vs_ideal:.4f}"
        )
        axes[1].set_title(title_noisy)
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        # 子图3: 最终降噪图像
        im2 = axes[2].imshow(final_denoised_view, cmap='gray', vmax=vmax_ideal)
        title_denoised = (
            f"最终降噪图像 (Denoised)\n"
            f"总计数: {np.sum(final_denoised_view):.0f}\n"
            f"PSNR: {psnr_denoised_vs_ideal:.2f} dB, SSIM: {ssim_denoised_vs_ideal:.4f}\n"
            f"LPIPS: {lpips_denoised_vs_ideal:.4f}"
        )
        axes[2].set_title(title_denoised)
        fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

        # 子图4: 降噪后 + 泊松噪声
        im3 = axes[3].imshow(denoised_plus_poisson_view, cmap='gray', vmax=vmax_ideal)
        title_denoised_P = (
            f"降噪+泊松 (Denoised+Poisson)\n"
            f"总计数: {np.sum(denoised_plus_poisson_view):.0f}\n-- vs Ideal --\n"
            f"PSNR: {psnr_denoisedP_vs_ideal:.2f} dB, SSIM: {ssim_denoisedP_vs_ideal:.4f}, LPIPS: {lpips_denoisedP_vs_ideal:.4f}\n"
            f"-- vs Ideal+Poisson --\n"
            f"PSNR: {psnr_denoisedP_vs_noisy:.2f} dB, SSIM: {ssim_denoisedP_vs_noisy:.4f}, LPIPS: {lpips_denoisedP_vs_noisy:.4f}"
        )
        axes[3].set_title(title_denoised_P)
        fig.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)
        
        for ax_col in axes:
            ax_col.axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.93])
        
        output_filename = f"{os.path.splitext(filename)[0]}_{view_name_short}_ideal_sim_denoised_ext.png"
        output_path = os.path.join(output_base_dir, output_filename)
        try:
            plt.savefig(output_path, dpi=200)
            print(f"  已保存图像: {output_path}") # 增加缩进
        except Exception as e:
            print(f"保存图像 {output_path} 时出错: {e}")
        plt.close(fig)
        # --- 可视化结束 ---
    print(f"--- 文件 {filename} 处理完成 ---") # 增加文件处理结束标记

def main():
    # 输入理想数据目录 (例如由 preprocess_spect_ideal.py 生成的 spectH_XCAT_ideal_1x)
    ideal_data_input_dir = os.path.join("SPECTdatasets", "spectH_XCAT_ideal_1x")
    
    # 输出结果目录
    output_dir = "bm3d_ideal_simulation_results"
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.isdir(ideal_data_input_dir):
        print(f"错误: 理想数据输入目录 '{ideal_data_input_dir}' 不存在。请先运行 preprocess_spect_ideal.py 生成理想数据或指定正确路径。")
        return
        
    # 获取目录中所有的.dat文件 (可以限制数量进行测试，例如 [:5])
    dat_files_all = [os.path.join(ideal_data_input_dir, f) for f in os.listdir(ideal_data_input_dir) if f.lower().endswith(".dat")]
    
    if not dat_files_all:
        print(f"在目录 '{ideal_data_input_dir}' 中未找到.dat文件。")
        return
        
    dat_files_to_process = dat_files_all[:1] # 修改这里，只处理第一个文件
    
    print(f"总共找到 {len(dat_files_all)} 个理想.dat文件，将处理其中的 {len(dat_files_to_process)} 个。")

    for ideal_dat_file_path in dat_files_to_process:
        process_and_visualize_ideal_dat_file(ideal_dat_file_path, output_dir)

    print(f"\n所有选定文件的处理完成。结果保存在目录: {output_dir}")

if __name__ == "__main__":
    main() 