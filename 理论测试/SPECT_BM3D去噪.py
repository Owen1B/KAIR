# 该脚本用于对SPECT（单光子发射计算机断层成像）图像进行BM3D降噪处理，
# 特别是针对泊松噪声，它首先应用Anscombe变换（VST）将泊松噪声近似为高斯噪声，
# 然后在变换域使用BM3D，最后通过逆Anscombe变换恢复图像。
# 脚本还模拟了低剂量（25%）情况，并比较了标准剂量和低剂量图像降噪后的结果。
# 主要输出:
# 1. 'spect_bm3d_denoising_comparison.png': 一张包含三幅子图的图像，显示：
#    - 原始SPECT图像（归一化后）
#    - 标准剂量图像经过BM3D降噪后的结果
#    - 25%剂量图像经过BM3D降噪后的结果，并标注其与标准剂量降噪结果的PSNR和SSIM。
# 2. 控制台输出标准剂量降噪图像与25%剂量降噪图像之间的PSNR和SSIM值。
import numpy as np
import matplotlib.pyplot as plt
import os
import bm3d
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# Setup Matplotlib for Chinese characters if available
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
except Exception:
    print("SimHei font not found, using default matplotlib fonts.")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Path to the SPECT file
DEFAULT_SPECT_FILE = "testsets/spectH/A03_20151119111344.dat"

def load_spect_anterior_data(file_path_param=DEFAULT_SPECT_FILE):
    """Loads the SPECT .dat file and returns the anterior view."""
    file_path = file_path_param
    if not os.path.exists(file_path):
        # Try to construct path relative to SCRIPT_DIR, assuming SCRIPT_DIR is in KAIR/理论测试
        # and DEFAULT_SPECT_FILE is relative to KAIR root
        parts = file_path_param.split('/')
        constructed_path = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', *parts))
        if os.path.exists(constructed_path):
            file_path = constructed_path
        else:
            print(f"Error: SPECT data file not found at {file_path_param} or {constructed_path}")
            return None
                
    print(f"Loading SPECT file from: {file_path}")
    try:
        data = np.fromfile(file_path, dtype=np.float32)
        reshaped_data = data.reshape(2, 1024, 256)
        return reshaped_data[0].astype(np.float32)  # Return anterior view
    except Exception as e:
        print(f"Error loading SPECT data: {e}")
        return None

def normalize_to_01(image):
    """Normalizes an image to the [0, 1] range."""
    min_val = np.min(image)
    max_val = np.max(image)
    scale = max_val - min_val
    if scale < 1e-9:
        return np.zeros_like(image, dtype=float) if min_val < 1e-9 else np.full_like(image, 0.5, dtype=float)
    return (image - min_val) / scale

def anscombe_transform(image_counts):
    """Applies Anscombe variance stabilizing transform."""
    return 2.0 * np.sqrt(np.maximum(image_counts, 0) + 3.0/8.0)

def inverse_anscombe_transform(image_vst):
    """Applies a basic inverse Anscombe transform."""
    return (image_vst / 2.0)**2 - 3.0/8.0

def main():
    print("--- BM3D Denoising for SPECT (Poisson Noise using VST) ---")

    original_image = load_spect_anterior_data()
    if original_image is None:
        print("Failed to load SPECT image. Exiting.")
        return
    
    original_image = np.maximum(original_image, 1e-6)
    print(f"Loaded original image. Shape: {original_image.shape}, Min: {original_image.min():.2e}, Max: {original_image.max():.2f}")

    p = 0.25  
    quarter_dose_image = np.random.binomial(original_image.astype(int), p).astype(np.float32)
    print(f"Generated quarter dose image. Min: {quarter_dose_image.min():.2f}, Max: {quarter_dose_image.max():.2f}")

    original_vst = anscombe_transform(original_image)
    quarter_vst = anscombe_transform(quarter_dose_image)

    sigma_bm3d_vst = 1.0
    print("Performing BM3D denoising...")
    try:
        original_denoised_vst = bm3d.bm3d(original_vst, sigma_psd=sigma_bm3d_vst)
        quarter_denoised_vst = bm3d.bm3d(quarter_vst, sigma_psd=sigma_bm3d_vst)
    except Exception as e:
        print(f"Error during BM3D processing: {e}")
        return

    original_denoised = np.maximum(inverse_anscombe_transform(original_denoised_vst), 0)
    quarter_denoised = np.maximum(inverse_anscombe_transform(quarter_denoised_vst), 0)

    original_denoised_norm = normalize_to_01(original_denoised)
    quarter_denoised_norm = normalize_to_01(quarter_denoised)

    denoised_psnr = psnr(original_denoised_norm, quarter_denoised_norm, data_range=1.0)
    denoised_ssim = ssim(original_denoised_norm, quarter_denoised_norm, data_range=1.0)

    print("\n--- Quality Metrics Between Denoised Images ---")
    print(f"PSNR: {denoised_psnr:.2f}dB")
    print(f"SSIM: {denoised_ssim:.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(10, 10))
    
    axes[0].imshow(normalize_to_01(original_image), cmap='gray')
    axes[0].set_title('原始图像')
    axes[0].axis('off')

    axes[1].imshow(original_denoised_norm, cmap='gray')
    axes[1].set_title('标准扫描降噪')
    axes[1].axis('off')

    axes[2].imshow(quarter_denoised_norm, cmap='gray')
    axes[2].set_title(f'25%剂量降噪\nPSNR: {denoised_psnr:.2f}dB, SSIM: {denoised_ssim:.4f}')
    axes[2].axis('off')

    plt.suptitle('SPECT图像 BM3D降噪对比', fontsize=16)
    plt.tight_layout()
    
    fig_dir = os.path.join(SCRIPT_DIR, 'fig')
    os.makedirs(fig_dir, exist_ok=True)
    plot_filename = os.path.join(fig_dir, "SPECT_BM3D降噪对比图.png") # Changed filename
    plt.savefig(plot_filename)
    print(f"Visualization saved to: {plot_filename}")
    plt.close(fig) # Close the figure to prevent display

if __name__ == '__main__':
    main() 