import os
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import sys
from scipy import stats
from tqdm import tqdm
from skimage.metrics import structural_similarity
# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False     # 用来正常显示负号

# 获取脚本所在目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 对数变换的小常数
LOG_EPS = 1e-10 

def load_spect_data(file_path):
    """加载SPECT .dat文件并返回前视图和后视图"""
    data = np.fromfile(file_path, dtype=np.float32)
    reshaped_data = data.reshape(2, 1024, 256)
    return reshaped_data[0], reshaped_data[1]  # 前视图, 后视图

def apply_gaussian_blur(image, fwhm):
    """应用高斯模糊"""
    sigma = fwhm / 2.355
    return scipy.ndimage.gaussian_filter(image, sigma)

def add_poisson_noise(image):
    """添加泊松噪声"""
    image_non_negative = np.maximum(image, 0)
    return np.random.poisson(image_non_negative).astype(np.float32)

def apply_binomial_sampling(image, speed):
    """通过二项采样模拟低剂量SPECT"""
    image_counts = np.maximum(image, 0).astype(np.int32)
    sampled_image = np.random.binomial(n=image_counts, p=1.0/speed) * speed
    return sampled_image.astype(np.float32)

def get_log_image(image_raw):
    """应用对数变换（以10为底）"""
    return np.log(image_raw + LOG_EPS) / np.log(10)

def normalize_log_image_globally(log_image, v_min_log_ref, v_max_log_ref):
    """使用全局对数最小/最大值参考将对数变换图像归一化到[0,1]"""
    if (v_max_log_ref - v_min_log_ref) == 0:
        return np.zeros_like(log_image)
    normalized_log_img = (log_image - v_min_log_ref) / (v_max_log_ref - v_min_log_ref)
    return np.clip(normalized_log_img, 0, 1)

def normalize_log_image_individually(log_image):
    """使用图像自身的最大/最小值将对数变换图像归一化到[0,1]"""
    log_image_min = np.min(log_image)
    log_image_max = np.max(log_image)
    if log_image_max > log_image_min:
        return (log_image - log_image_min) / (log_image_max - log_image_min)
    return np.zeros_like(log_image)

def prepare_for_vgg(normalized_image_0_1, device):
    """将[0,1]归一化的numpy图像转换为VGG输入张量"""
    img_pil = Image.fromarray((normalized_image_0_1 * 255).astype(np.uint8))
    transform_vgg = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform_vgg(img_pil.convert('RGB')).unsqueeze(0)
    return img_tensor.to(device)

def calculate_perceptual_loss(img1_raw, img2_raw, vgg_model, v_min_log_ref, v_max_log_ref, device):
    """使用全局对数归一化参考计算感知损失"""
    log_img1 = get_log_image(img1_raw)
    log_img2 = get_log_image(img2_raw)
    
    norm_log_img1 = normalize_log_image_globally(log_img1, v_min_log_ref, v_max_log_ref)
    norm_log_img2 = normalize_log_image_globally(log_img2, v_min_log_ref, v_max_log_ref)

    tensor1 = prepare_for_vgg(norm_log_img1, device)
    tensor2 = prepare_for_vgg(norm_log_img2, device)
    
    with torch.no_grad():
        feat1 = vgg_model(tensor1)
        feat2 = vgg_model(tensor2)
    return torch.nn.MSELoss()(feat1, feat2).item()

def calculate_psnr_log(img1_raw, img2_raw):
    """计算对数变换和最小-最大归一化图像的PSNR"""
    log_img1 = get_log_image(img1_raw)
    log_img2 = get_log_image(img2_raw)

    norm_log_img1 = normalize_log_image_individually(log_img1)
    norm_log_img2 = normalize_log_image_individually(log_img2)
    
    mse = np.mean((norm_log_img1 - norm_log_img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(1.0 / np.sqrt(mse))

def calculate_ssim_log(img1_raw, img2_raw, win_size):
    """计算对数变换和最小-最大归一化图像的SSIM"""
    log_img1 = get_log_image(img1_raw)
    log_img2 = get_log_image(img2_raw)

    norm_log_img1 = normalize_log_image_individually(log_img1)
    norm_log_img2 = normalize_log_image_individually(log_img2)

    min_dim = min(norm_log_img1.shape)
    # Ensure win_size is odd, >=3 and <= min_dim
    if win_size % 2 == 0: win_size -=1
    win_size = max(3, win_size)
    win_size = min(win_size, min_dim if min_dim % 2 != 0 else min_dim -1)
    
    if win_size < 3 or win_size > min_dim : # check if win_size is valid after adjustments
        # print(f"警告: 图像维度 ({norm_log_img1.shape}) 对于SSIM窗口大小 ({win_size}) 无效。将返回NaN。")
        return np.nan

    return structural_similarity(
        norm_log_img1.astype(np.float32),
        norm_log_img2.astype(np.float32),
        data_range=1.0,
        win_size=win_size,
        gaussian_weights=True,
        channel_axis=None # For 2D grayscale images
    )

def main():
    # 可调参数
    ideal_image_path = 'trainsets/spectH_raw/B001_pure.dat'
    fwhm_blur = 7.0
    speed_factors_start = 1.0
    speed_factors_end = 8.0
    speed_factors_step = 0.5
    n_instances = 10
    vgg_feature_end_layer_idx = 29
    ssim_window_size = 7
    
    speeds = np.arange(speed_factors_start, speed_factors_end + speed_factors_step / 2, speed_factors_step)

    print("开始分析脚本，使用 Pytorch {} 和 Matplotlib {}".format(torch.__version__, plt.matplotlib.__version__))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载VGG19模型
    vgg19_full_features = models.vgg19(pretrained=True).features
    vgg_model = vgg19_full_features[:vgg_feature_end_layer_idx + 1].eval().to(device)
    print(f"已加载VGG19模型，使用到层: {vgg_feature_end_layer_idx}")

    # 加载理想图像
    ideal_anterior, _ = load_spect_data(ideal_image_path)
    print(f"成功加载理想图像，形状: {ideal_anterior.shape}")
    ideal_image_raw = ideal_anterior.copy()

    # --- 第1a部分: 标准剂量实例间感知损失 (对应150万计数标准扫描) ---
    print("\n第1a部分: 标准剂量实例间感知损失")
    blurred_ideal_raw = apply_gaussian_blur(ideal_image_raw, fwhm_blur)
    std_poisson_instances_raw = [add_poisson_noise(blurred_ideal_raw.copy()) for _ in range(n_instances)]
    
    log_std_poisson_instances = [get_log_image(inst) for inst in std_poisson_instances_raw]
    vmin_log_stand = np.min([np.min(log_inst) for log_inst in log_std_poisson_instances])
    vmax_log_stand = np.max([np.max(log_inst) for log_inst in log_std_poisson_instances])
    print(f"标准剂量对数域全局归一化参考: 最小值={vmin_log_stand:.4f}, 最大值={vmax_log_stand:.4f}")

    inter_losses_std = []
    if n_instances > 1:
        for i in tqdm(range(n_instances), desc="标准剂量实例间损失"):
            for j in range(i + 1, n_instances):
                loss = calculate_perceptual_loss(std_poisson_instances_raw[i], std_poisson_instances_raw[j],
                                               vgg_model, vmin_log_stand, vmax_log_stand, device)
                inter_losses_std.append(loss)
    
    mean_inter_loss_std, std_inter_loss_std = np.nan, np.nan # Initialize
    if inter_losses_std:
        mean_inter_loss_std = np.mean(inter_losses_std)
        std_inter_loss_std = np.std(inter_losses_std)
        print(f"平均感知损失 (标准剂量实例间): {mean_inter_loss_std:.6f} ± {std_inter_loss_std:.6f}")
        
        # 此部分原图绘制逻辑将被新的四合一图取代或调整，此处仅计算
        # plt.figure(figsize=(10,6))
        # plt.hist(inter_losses_std, bins=min(30, len(inter_losses_std)//2 if len(inter_losses_std)>4 else 5), 
        #         density=True, alpha=0.7, label='感知损失分布')
        # plt.axvline(mean_inter_loss_std, color='r', linestyle='dashed', label=f'平均值: {mean_inter_loss_std:.4f}')
        # xmin_hist, xmax_hist = plt.xlim()
        # x_pdf = np.linspace(xmin_hist, xmax_hist, 100)
        # pdf = stats.norm.pdf(x_pdf, mean_inter_loss_std, std_inter_loss_std)
        # plt.plot(x_pdf, pdf, 'k', linewidth=2, label='正态分布拟合')
        # plt.title('感知损失分布 (标准剂量实例间)')
        # plt.xlabel('感知损失'); plt.ylabel('密度'); plt.legend(); plt.grid(True)
        # plt.savefig(os.path.join(SCRIPT_DIR, 'pl_dist_std_dose.png')); plt.close()
        # print("已保存图像: pl_dist_std_dose.png")
    else:
        print("标准剂量实例数不足，无法计算实例间损失的均值和标准差。")


    # --- 第1b部分: 不同模拟剂量下的标准扫描实例之间的感知损失 ---
    # (理想图像 / speed_factor 再加泊松噪声，模拟不同剂量下的标准扫描)
    print("\n第1b部分: 不同模拟剂量下的标准扫描实例之间的感知损失")
    mean_inter_losses_low_dose_vs_speed = []
    std_inter_losses_low_dose_vs_speed = [] # 新增：存储每个剂量下的标准差

    for speed_factor in tqdm(speeds, desc="不同剂量实例间损失 (按speed因子模拟剂量)"):
        # speed_factor 此处用于模拟剂量变化：1/speed_factor 为剂量比例
        base_for_low_dose_raw = ideal_image_raw / speed_factor 
        blurred_low_dose_raw = apply_gaussian_blur(base_for_low_dose_raw, fwhm_blur)
        # 为当前模拟剂量生成n个泊松噪声实例
        low_poisson_instances_raw_s = [add_poisson_noise(blurred_low_dose_raw.copy()) for _ in range(n_instances)]
        
        # 为当前剂量的实例计算全局归一化参数
        log_low_poisson_instances_s = [get_log_image(inst) for inst in low_poisson_instances_raw_s]
        vmin_log_s = np.min([np.min(log_inst) for log_inst in log_low_poisson_instances_s] if log_low_poisson_instances_s else [0])
        vmax_log_s = np.max([np.max(log_inst) for log_inst in log_low_poisson_instances_s] if log_low_poisson_instances_s else [0])

        current_speed_inter_losses = []
        if n_instances > 1:
            for i in range(n_instances):
                for j in range(i + 1, n_instances):
                    loss = calculate_perceptual_loss(low_poisson_instances_raw_s[i], low_poisson_instances_raw_s[j],
                                                   vgg_model, vmin_log_s, vmax_log_s, device)
                    current_speed_inter_losses.append(loss)
        
        if current_speed_inter_losses:
            mean_inter_losses_low_dose_vs_speed.append(np.mean(current_speed_inter_losses))
            std_inter_losses_low_dose_vs_speed.append(np.std(current_speed_inter_losses))
        else:
            mean_inter_losses_low_dose_vs_speed.append(np.nan)
            std_inter_losses_low_dose_vs_speed.append(np.nan)
    
    # 此部分原图绘制逻辑将被新的四合一图的子图2取代
    # if any(np.isfinite(m) for m in mean_inter_losses_low_dose_vs_speed):
    #     plt.figure(figsize=(10,6))
    #     plt.plot(speeds, mean_inter_losses_low_dose_vs_speed, 'o-', label='平均实例间感知损失 (不同剂量基准)') # 标签更新
    #     plt.title('不同模拟剂量下实例间平均感知损失 vs. Speed Factor (模拟剂量)') # 标题更新
    #     plt.xlabel('Speed Factor (理想图像缩放比例 1/Speed)'); plt.ylabel('平均感知损失'); plt.legend(); plt.grid(True)
    #     plt.savefig(os.path.join(SCRIPT_DIR, 'pl_inter_low_dose_vs_speed.png')); plt.close()
    #     print("已保存图像: pl_inter_low_dose_vs_speed.png")

    # --- 第2部分: 150万剂量下不同扫描速度降质图像与标准扫描的比较 ---
    # (标准扫描实例 vs 二项采样降质版本)
    print("\n第2部分: 150万剂量下，标准扫描 vs 不同扫描速度降质")
    
    losses_std_vs_degraded = []
    psnrs_std_vs_degraded = []
    ssims_std_vs_degraded = []
    # degraded_images_for_plot = [] # 这个列表在新图中不直接使用，可以注释掉

    if not std_poisson_instances_raw:
        print("错误: 未生成标准泊松实例 (std_poisson_instances_raw为空)。无法进行第2部分比较。")
    else:
        # 使用第一个标准泊松实例作为参考进行降质
        ref_std_poisson_instance_raw = std_poisson_instances_raw[0]
        
        for speed_factor in tqdm(speeds, desc="标准vs降质 (不同扫描速度)"):
            # speed_factor 此处用于模拟扫描速度变化，导致降质
            degraded_raw = apply_binomial_sampling(ref_std_poisson_instance_raw.copy(), speed_factor)
            # degraded_images_for_plot.append(degraded_raw)

            # 感知损失使用Part 1a中计算的标准剂量全局归一化参数 (vmin_log_stand, vmax_log_stand)
            loss = calculate_perceptual_loss(ref_std_poisson_instance_raw, degraded_raw,
                                           vgg_model, vmin_log_stand, vmax_log_stand, device)
            losses_std_vs_degraded.append(loss)
            psnrs_std_vs_degraded.append(calculate_psnr_log(ref_std_poisson_instance_raw, degraded_raw))
            ssims_std_vs_degraded.append(calculate_ssim_log(ref_std_poisson_instance_raw, degraded_raw, ssim_window_size))
        
        # 此部分原图绘制逻辑将被新的四合一图取代
        # fig_part2, axs_part2 = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
        # axs_part2[0].plot(speeds, losses_std_vs_degraded, 'b-o', label='感知损失 (标准vs降质)')
        # # ... (原PSNR, SSIM绘图)
        # plt.savefig(os.path.join(SCRIPT_DIR, 'metrics_std_vs_degraded.png')); plt.close(fig_part2)
        # print("已保存图像: metrics_std_vs_degraded.png")

        # 原示例图像绘制逻辑也可以移除或注释，因为新的四合一图专注于指标
        # if degraded_images_for_plot:
        # # ... (原 degrading images 绘图)
        # plt.savefig(os.path.join(SCRIPT_DIR, 'imgs_std_vs_degraded.png'))
        # plt.close(fig_imgs_p2)
        # print("已保存图像: imgs_std_vs_degraded.png")

    # --- 新的四合一图表 --- 
    print("\n正在生成四合一图表: metrics_vs_speed_anscombe.png")
    fig_anscombe, axs_anscombe = plt.subplots(4, 1, figsize=(12, 20), sharex=True)

    # 子图1: 150万剂量下不同扫描速度的感知损失
    if losses_std_vs_degraded:
        axs_anscombe[0].plot(speeds, losses_std_vs_degraded, 'bo-', label='降质图像 vs 标准实例 PL', markersize=5)
        if not np.isnan(mean_inter_loss_std):
            axs_anscombe[0].axhline(mean_inter_loss_std, color='r', linestyle='--', label=f'标准实例间平均PL: {mean_inter_loss_std:.4f}')
            if not np.isnan(std_inter_loss_std):
                axs_anscombe[0].fill_between(speeds, mean_inter_loss_std - std_inter_loss_std, 
                                         mean_inter_loss_std + std_inter_loss_std, 
                                         color='r', alpha=0.2, label=f'标准实例间PL标准差范围')
        axs_anscombe[0].set_ylabel('感知损失')
        axs_anscombe[0].set_title('子图1: 150万计数标准剂量下, 不同扫描速度降质图像与标准扫描的感知损失')
        axs_anscombe[0].legend()
        axs_anscombe[0].grid(True)
    else:
        axs_anscombe[0].text(0.5, 0.5, '无数据', horizontalalignment='center', verticalalignment='center', transform=axs_anscombe[0].transAxes)
        axs_anscombe[0].set_title('子图1: 数据不足')

    # 子图2: 不同剂量下的标准扫描的实例之间的感知损失
    if mean_inter_losses_low_dose_vs_speed and std_inter_losses_low_dose_vs_speed:
        # 确保 speeds, means, stds 长度一致且非空
        valid_indices = [i for i, (m,s) in enumerate(zip(mean_inter_losses_low_dose_vs_speed, std_inter_losses_low_dose_vs_speed)) if not (np.isnan(m) or np.isnan(s))]
        if valid_indices:
            plot_speeds = speeds[valid_indices]
            plot_means = np.array(mean_inter_losses_low_dose_vs_speed)[valid_indices]
            plot_stds = np.array(std_inter_losses_low_dose_vs_speed)[valid_indices]
            axs_anscombe[1].errorbar(plot_speeds, plot_means, yerr=plot_stds, fmt='go-', 
                                 label='实例间平均PL', capsize=5, markersize=5)
        else:
             axs_anscombe[1].text(0.5, 0.5, '无有效数据点', ha='center', va='center', transform=axs_anscombe[1].transAxes)
        axs_anscombe[1].set_ylabel('平均感知损失')
        axs_anscombe[1].set_title('子图2: 不同模拟剂量下, 标准扫描实例间的感知损失 (误差棒为标准差)')
        axs_anscombe[1].legend()
        axs_anscombe[1].grid(True)
    else:
        axs_anscombe[1].text(0.5, 0.5, '无数据', horizontalalignment='center', verticalalignment='center', transform=axs_anscombe[1].transAxes)
        axs_anscombe[1].set_title('子图2: 数据不足')

    # 子图3: 150万剂量下不同扫描速度与标准扫描的PSNR
    if psnrs_std_vs_degraded:
        axs_anscombe[2].plot(speeds, psnrs_std_vs_degraded, 'gs-', label='降质图像 vs 标准实例 PSNR', markersize=5)
        axs_anscombe[2].set_ylabel('PSNR (dB)')
        axs_anscombe[2].set_title('子图3: 150万计数标准剂量下, 不同扫描速度降质图像与标准扫描的PSNR')
        axs_anscombe[2].legend()
        axs_anscombe[2].grid(True)
    else:
        axs_anscombe[2].text(0.5, 0.5, '无数据', horizontalalignment='center', verticalalignment='center', transform=axs_anscombe[2].transAxes)
        axs_anscombe[2].set_title('子图3: 数据不足')

    # 子图4: 150万剂量下不同扫描速度与标准扫描的SSIM
    if ssims_std_vs_degraded:
        axs_anscombe[3].plot(speeds, ssims_std_vs_degraded, 'm^-', label='降质图像 vs 标准实例 SSIM', markersize=5)
        axs_anscombe[3].set_ylabel('SSIM')
        axs_anscombe[3].set_title('子图4: 150万计数标准剂量下, 不同扫描速度降质图像与标准扫描的SSIM')
        axs_anscombe[3].legend()
        axs_anscombe[3].grid(True)
    else:
        axs_anscombe[3].text(0.5, 0.5, '无数据', horizontalalignment='center', verticalalignment='center', transform=axs_anscombe[3].transAxes)
        axs_anscombe[3].set_title('子图4: 数据不足')

    # 通用X轴标签
    axs_anscombe[3].set_xlabel('Speed Factor (子图1,3,4: 扫描速度模拟 / 子图2: 剂量模拟因数 1/Speed)')

    fig_anscombe.suptitle('不同剂量及扫描速度下降质对图像质量指标影响分析', fontsize=16)
    fig_anscombe.tight_layout(rect=[0, 0.03, 1, 0.96]) # 调整布局以适应总标题
    save_path_anscombe = os.path.join(SCRIPT_DIR, 'metrics_vs_speed_anscombe.png')
    plt.savefig(save_path_anscombe)
    plt.close(fig_anscombe)
    print(f"已保存四合一图表: {save_path_anscombe}")

    print("\n分析完成。")

if __name__ == '__main__':
    main()