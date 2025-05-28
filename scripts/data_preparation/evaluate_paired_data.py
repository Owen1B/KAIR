import os
import sys
# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch
import lpips
from tqdm import tqdm
import utils.utils_image as util
from utils import utils_spect
import bm3d
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
plt.rcParams['font.family'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
def anscombe_transform(img):
    """Anscombe变换"""
    return 2 * np.sqrt(img + 3/8)

def inverse_anscombe_transform(img):
    """反Anscombe变换"""
    return (img/2)**2 - 3/8

def add_poisson_noise(img, scale=1.0):
    """添加泊松噪声，考虑原始图像强度"""
    # 将图像转换回原始强度范围
    img_original = img.astype(np.float32) * 150 / 255.0
    
    # 添加泊松噪声
    img_noisy = np.random.poisson(img_original * scale) / scale
    
    # 确保值在合理范围内
    img_noisy = np.clip(img_noisy, 0, 150)
    
    return img_noisy

def process_image_original(img):
    """处理原始图像：Anscombe -> BM3D -> 反Anscombe"""
    # Anscombe变换
    img_anscombe = anscombe_transform(img)
    
    # BM3D降噪
    img_denoised = bm3d.bm3d(img_anscombe.astype(np.float32), sigma_psd=1, stage_arg=bm3d.BM3DStages.ALL_STAGES)
    
    # 反Anscombe变换
    img_inverse = inverse_anscombe_transform(img_denoised)
    
    return img_inverse

def process_image_8x(img):
    """处理8x图像：除以8 -> Anscombe -> BM3D -> 反Anscombe -> 乘以8"""
    # 除以8
    img = img / 8.0
    
    # Anscombe变换
    img_anscombe = anscombe_transform(img)
    
    # BM3D降噪
    img_denoised = bm3d.bm3d(img_anscombe.astype(np.float32), sigma_psd=1, stage_arg=bm3d.BM3DStages.ALL_STAGES)
    
    # 反Anscombe变换
    img_inverse = inverse_anscombe_transform(img_denoised)
    
    # 乘以8
    img_restored = img_inverse * 8.0
    
    return img_restored

def normalize_to_255(img):
    """将图像归一化到0-255范围，使用固定值150"""
    img = np.clip(img, 0, 150)
    img = (img / 150 * 255).astype(np.uint8)
    return img

def calculate_metrics(img1, img2):
    """计算两个图像之间的PSNR、SSIM和LPIPS"""
    # 计算PSNR (使用0-1范围)
    img1_float = img1.astype(np.float32) / 255.0
    img2_float = img2.astype(np.float32) / 255.0
    mse = np.mean((img1_float - img2_float) ** 2)
    psnr = -10 * np.log10(mse + 1e-8)
    
    # 计算SSIM (使用0-255范围)
    ssim = util.calculate_ssim(img1, img2)
    
    # 计算LPIPS (使用0-1范围)
    img1_tensor = torch.from_numpy(img1_float).unsqueeze(0).unsqueeze(0).float()
    img2_tensor = torch.from_numpy(img2_float).unsqueeze(0).unsqueeze(0).float()
    loss_fn = lpips.LPIPS(net='alex')
    lpips_value = loss_fn(img1_tensor, img2_tensor).item()
    
    return psnr, ssim, lpips_value

def visualize_results(images, titles, metrics, save_path):
    """可视化结果"""
    fig = plt.figure(figsize=(9, 24))
    gs = GridSpec(2, 3, figure=fig)
    
    # 显示图像
    for i, (img, title) in enumerate(zip(images, titles)):
        ax = fig.add_subplot(gs[i//3, i%3])
        ax.imshow(img, cmap='gray',vmin=0,vmax=150)
        ax.set_title(title)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    # 设置路径
    high_dir = 'SPECTdatasets/spectH_clinical'
    low_dir = 'SPECTdatasets/spectL_clinical_binomial_8x'
    results_dir = 'evaluation_results'
    os.makedirs(results_dir, exist_ok=True)
    
    # 获取所有文件
    high_files = sorted([f for f in os.listdir(high_dir) if f.endswith('.dat')])
    low_files = sorted([f for f in os.listdir(low_dir) if f.endswith('.dat')])
    
    # 确保文件配对
    assert len(high_files) == len(low_files), "文件数量不匹配"
    for h, l in zip(high_files, low_files):
        assert h == l, f"文件不匹配: {h} vs {l}"
    
    # 初始化指标列表
    all_metrics = []
    
    # 遍历所有文件对
    for high_file, low_file in tqdm(zip(high_files, low_files), total=len(high_files), desc="处理文件"):
        # 读取数据
        high_data = np.fromfile(os.path.join(high_dir, high_file), dtype=np.float32)
        low_data = np.fromfile(os.path.join(low_dir, low_file), dtype=np.float32)
        
        # 重塑数据
        high_data = high_data.reshape(2, 1024, 256)
        low_data = low_data.reshape(2, 1024, 256)
        
        # 处理前位和后位投影
        for proj_idx, proj_name in enumerate(['前位', '后位']):
            # 获取投影数据
            high_proj = high_data[proj_idx]
            low_proj = low_data[proj_idx]
            
            # 处理图像
            # 1. 原始图
            img1 = normalize_to_255(high_proj)
            
            # 2. 原始图处理
            img2 = process_image_original(high_proj)
            img2 = normalize_to_255(img2)
            
            # 3. 原始图处理后加噪声
            img3 = add_poisson_noise(img2, 1.0)
            img3 = normalize_to_255(img3)
            
            # 4. 8x图
            img4 = normalize_to_255(low_proj)
            
            # 5. 8x图处理
            img5 = process_image_8x(low_proj)
            img5 = normalize_to_255(img5)
            
            # 6. 8x图处理后加噪声
            img6 = add_poisson_noise(img5, 1.0)
            img6 = normalize_to_255(img6)
            
            # 计算指标
            metrics = []
            images = [img1, img2, img3, img4, img5, img6]
            titles = ['原始图', '原始图处理', '原始图处理后加噪声', 
                     '8x图', '8x图处理', '8x图处理后加噪声']
            
            for img in images[1:]:
                psnr, ssim, lpips_value = calculate_metrics(img1, img)
                metrics.append((psnr, ssim, lpips_value))
            
            # 可视化结果
            save_path = os.path.join(results_dir, f'{high_file}_{proj_name}.png')
            visualize_results(images, titles, metrics, save_path)
            
            # 存储指标
            all_metrics.append({
                'file': high_file,
                'projection': proj_name,
                'metrics': metrics
            })
            
            print(f"\n文件: {high_file}, {proj_name}投影:")
            for i, (title, (psnr, ssim, lpips)) in enumerate(zip(titles[1:], metrics)):
                print(f"\n{title} vs 原始图:")
                print(f"PSNR: {psnr:.2f} dB")
                print(f"SSIM: {ssim:.4f}")
                print(f"LPIPS: {lpips:.4f}")
    
    # 计算平均指标
    avg_metrics = np.zeros((5, 3))  # 5种处理方式，3个指标
    count = 0
    
    for result in all_metrics:
        for i, (psnr, ssim, lpips) in enumerate(result['metrics']):
            avg_metrics[i] += np.array([psnr, ssim, lpips])
        count += 1
    
    avg_metrics /= count
    
    # 保存总体结果
    with open(os.path.join(results_dir, 'overall_results.txt'), 'w') as f:
        f.write("总体评估结果\n")
        f.write("============\n\n")
        for i, title in enumerate(titles[1:]):
            f.write(f"{title} vs 原始图:\n")
            f.write(f"平均 PSNR: {avg_metrics[i,0]:.2f} dB\n")
            f.write(f"平均 SSIM: {avg_metrics[i,1]:.4f}\n")
            f.write(f"平均 LPIPS: {avg_metrics[i,2]:.4f}\n\n")
    
    print("\n总体统计:")
    for i, title in enumerate(titles[1:]):
        print(f"\n{title} vs 原始图:")
        print(f"平均 PSNR: {avg_metrics[i,0]:.2f} dB")
        print(f"平均 SSIM: {avg_metrics[i,1]:.4f}")
        print(f"平均 LPIPS: {avg_metrics[i,2]:.4f}")

if __name__ == '__main__':
    main() 