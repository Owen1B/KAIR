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
# 尝试使用GPU，如果可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loss_fn_alex = lpips.LPIPS(net='alex').to(device)
print(f"LPIPS模型将在 {device} 上运行")

def anscombe_transform(x: np.ndarray) -> np.ndarray:
    """
    Anscombe变换: 2 * sqrt(x + 3/8)
    将泊松噪声近似转化为方差为1的高斯噪声
    """
    return 2 * np.sqrt(np.maximum(0, x) + 3/8) # x应为非负

def inverse_anscombe_transform(y: np.ndarray) -> np.ndarray:
    """
    Anscombe反变换: (y/2)^2 - 3/8
    """
    term = y / 2.0
    return np.maximum(0, term**2 - 3/8) # 确保输出非负

def prepare_for_lpips(img_np: np.ndarray, data_range: float, device: torch.device) -> torch.Tensor:
    """将NumPy图像转换为LPIPS所需的PyTorch张量格式。"""
    # LPIPS期望输入范围是 [-1, 1]
    img_normalized = img_np / data_range * 2 - 1
    # 转换为张量并添加批次和通道维度 (B, C, H, W)
    img_tensor = torch.from_numpy(img_normalized.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    return img_tensor.to(device)

def process_and_visualize_dat_file(dat_file_path: str, output_base_dir: str):
    """
    处理单个DAT文件，执行Anscombe变换、BM3D降噪、反Anscombe变换，计算指标并可视化。
    """
    filename = os.path.basename(dat_file_path)
    print(f"正在处理文件: {filename}...")

    try:
        # 读取SPECT数据 (2个视图, 每个视图 1024x256)
        data = np.fromfile(dat_file_path, dtype=np.float32)
        if data.size != 2 * 1024 * 256:
            print(f"警告: 文件 {filename} 的大小不是预期的 2*1024*256。跳过此文件。")
            return
        data = data.reshape(2, 1024, 256)
    except Exception as e:
        print(f"读取或重塑文件 {filename} 时出错: {e}。跳过此文件。")
        return

    view_names = ['前位 (Anterior)', '后位 (Posterior)']
    
    for i, view_name_full in enumerate(view_names):
        view_name_short = 'anterior' if 'Anterior' in view_name_full else 'posterior'
        original_view = data[i]

        # 1. Anscombe 变换
        anscombe_view = anscombe_transform(original_view)

        # 2. BM3D 降噪
        # Anscombe变换使噪声方差近似为1，所以 sigma_psd = 1.0
        sigma_psd = 1.0
        
        # BM3D处理。使用NORMAL配置。
        # BM3D可能对输入类型敏感，确保是float。np.sqrt默认返回float64。
        denoised_anscombe_view = bm3d.bm3d(
            anscombe_view.astype(np.float32), # 确保为float32
            sigma_psd=sigma_psd,
            stage_arg=bm3d.BM3DStages.ALL_STAGES, # 执行所有BM3D阶段
            profile='np' # 使用标准配置
        )

        # 3. 反 Anscombe 变换
        final_denoised_view = inverse_anscombe_transform(denoised_anscombe_view)

        # 5. 给最终降噪图像添加泊松噪声
        # 确保 final_denoised_view 是非负的，np.random.poisson 的 lambda 参数必须 >= 0
        # inverse_anscombe_transform 已经处理了非负性，但再次确认或剪裁是安全的
        positive_final_denoised_view = np.maximum(0, final_denoised_view)
        renoised_view = np.random.poisson(positive_final_denoised_view)

        # 计算三个图像的总计数
        total_counts_original = np.sum(original_view)
        total_counts_denoised = np.sum(final_denoised_view)
        total_counts_renoised = np.sum(renoised_view)

        # 4. 可视化 (现在是 1x3) 和 指标计算
        vmax_val = np.max(original_view)
        data_r = vmax_val if vmax_val > 0 else 1.0 # 确保data_range非零

        # 计算指标 vs 原始图像
        # 对于 final_denoised_view
        psnr_denoised = peak_signal_noise_ratio(original_view, final_denoised_view, data_range=data_r)
        ssim_denoised = structural_similarity(original_view, final_denoised_view, data_range=data_r, win_size=7, channel_axis=None)
        
        # 对于 renoised_view
        psnr_renoised = peak_signal_noise_ratio(original_view, renoised_view, data_range=data_r)
        ssim_renoised = structural_similarity(original_view, renoised_view, data_range=data_r, win_size=7, channel_axis=None)

        # LPIPS 计算 (确保图像是float32)
        original_lpips_tensor = prepare_for_lpips(original_view.astype(np.float32), data_r, device)
        denoised_lpips_tensor = prepare_for_lpips(final_denoised_view.astype(np.float32), data_r, device)
        renoised_lpips_tensor = prepare_for_lpips(renoised_view.astype(np.float32), data_r, device)

        with torch.no_grad(): # 在评估模式下运行LPIPS，不计算梯度
            lpips_denoised = loss_fn_alex(original_lpips_tensor, denoised_lpips_tensor).item()
            lpips_renoised = loss_fn_alex(original_lpips_tensor, renoised_lpips_tensor).item()

        fig, axes = plt.subplots(1, 3, figsize=(7, 7)) # 调整为 1x3 布局
        fig.suptitle(f"文件: {filename} - 视图: {view_name_full}", fontsize=16)

        # 子图1: 原始图像
        im0 = axes[0].imshow(original_view, cmap='gray', vmax=vmax_val)
        title_original = (
            f"原始图像 (Original)\n"
            f"总计数: {total_counts_original:.0f}"
        )
        axes[0].set_title(title_original)
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        # 子图2: 最终降噪图像
        im1 = axes[1].imshow(final_denoised_view, cmap='gray', vmax=vmax_val)
        title_denoised = (
            f"最终降噪图像 (Denoised)\n"
            f"总计数: {total_counts_denoised:.0f}\n"
            f"PSNR: {psnr_denoised:.2f} dB\n"
            f"SSIM: {ssim_denoised:.4f}\n"
            f"LPIPS: {lpips_denoised:.4f}"
        )
        axes[1].set_title(title_denoised)
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        # 子图3: 降噪后+泊松噪声
        im2 = axes[2].imshow(renoised_view, cmap='gray', vmax=vmax_val)
        title_renoised = (
            f"降噪后+泊松噪声 (Denoised + Poisson)\n"
            f"总计数: {total_counts_renoised:.0f}\n"
            f"PSNR: {psnr_renoised:.2f} dB\n"
            f"SSIM: {ssim_renoised:.4f}\n"
            f"LPIPS: {lpips_renoised:.4f}"
        )
        axes[2].set_title(title_renoised)
        fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
        
        for ax_col in axes:
            ax_col.axis('off') # 关闭坐标轴显示

        plt.tight_layout(rect=[0, 0.03, 1, 0.93]) # 为总标题和底部留出空间
        
        # 保存图像
        output_filename = f"{os.path.splitext(filename)[0]}_{view_name_short}_anscombe_bm3d.png"
        output_path = os.path.join(output_base_dir, output_filename)
        try:
            plt.savefig(output_path, dpi=200) # dpi可以根据需要调整
            print(f"已保存图像: {output_path}")
        except Exception as e:
            print(f"保存图像 {output_path} 时出错: {e}")
        plt.close(fig)


def main():
    # 输入数据目录
    # 注意：用户提供了SPECTdatasets/spectH_clinical作为目标
    data_base_dir = "SPECTdatasets" 
    clinical_subdir = "spectH_XCAT_poisson_1x"
    input_data_dir = os.path.join(data_base_dir, clinical_subdir)

    # 输出结果目录
    output_dir = "bm3d_real_denoising_results"
    os.makedirs(output_dir, exist_ok=True)

    # 获取目录中所有的.dat文件
    if not os.path.isdir(input_data_dir):
        print(f"错误: 输入目录 '{input_data_dir}' 不存在。")
        return
        
    dat_files = [os.path.join(input_data_dir, f) for f in os.listdir(input_data_dir) if f.lower().endswith(".dat")][:5]

    if not dat_files:
        print(f"在目录 '{input_data_dir}' 中未找到.dat文件。")
        return
    
    print(f"找到 {len(dat_files)} 个.dat文件进行处理。")

    for dat_file_path in dat_files:
        process_and_visualize_dat_file(dat_file_path, output_dir)

    print(f"\n所有处理完成。结果保存在目录: {output_dir}")

if __name__ == "__main__":
    main()
