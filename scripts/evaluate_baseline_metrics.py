import sys
import os
import warnings
warnings.filterwarnings("ignore")
# 将项目根目录 (KAIR) 添加到 Python 的模块搜索路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import math
import matplotlib.pyplot as plt

from utils import utils_image as util
from utils import utils_option as option
from data.select_dataset import define_Dataset

'''
# --------------------------------------------
# 计算测试数据集中输入(L)和目标(H)图像间的评价指标
# (支持全局归一化和局部归一化两种方式)
# --------------------------------------------
'''

def evaluate_dataset(dataset_opt, device):
    """评估单个数据集的指标"""
    test_set = define_Dataset(dataset_opt)
    test_loader = DataLoader(test_set, batch_size=1,
                           shuffle=False, num_workers=1,
                           drop_last=False, pin_memory=True)
    
    # 初始化指标累加器
    metrics_sum_global = {'psnr': 0, 'ssim': 0, 'lpips': 0}
    metrics_sum_local = {'psnr': 0, 'ssim': 0, 'lpips': 0}
    count = 0
    
    # 存储所有图像数据
    all_imgs = {'L': [], 'H': []}
    image_names = []
    
    # 遍历：收集所有图像数据
    for test_data in test_loader:
        L_tensor = test_data['L'].squeeze(0).cpu().float()
        H_tensor = test_data['H'].squeeze(0).cpu().float()
        
        L_img_hwc = np.transpose(L_tensor.numpy(), (1, 2, 0))
        H_img_hwc = np.transpose(H_tensor.numpy(), (1, 2, 0))
        
        image_name_ext = os.path.basename(test_data['L_path'][0])
        image_name = os.path.splitext(image_name_ext)[0]
        image_names.append(image_name)
        
        all_imgs['L'].append(L_img_hwc)
        all_imgs['H'].append(H_img_hwc)
    
    if not all_imgs['L']:
        return None, [], []
    
    # 计算全局最大最小值
    all_H_values = np.concatenate([img.flatten() for img in all_imgs['H']])
    max_val_global = np.max(all_H_values) * 1  # 设置为H最大值的110%
    min_val_global = 0  # 固定最小值为0
    
    # 计算每张图片的指标并生成可视化
    visuals_list = []
    
    for idx, (L_img, H_img) in enumerate(zip(all_imgs['L'], all_imgs['H'])):
        count += 1
        image_name = image_names[idx]
        
        # --- 全局归一化处理 ---
        # 先clip到[0, max_val_global]
        L_img_clipped = np.clip(L_img, 0, max_val_global)
        H_img_clipped = np.clip(H_img, 0, max_val_global)
        # 归一化到[0,255]
        L_img_255 = (L_img_clipped / max_val_global * 255).astype(np.uint8)
        H_img_255 = (H_img_clipped / max_val_global * 255).astype(np.uint8)
        
        # 计算全局指标
        for ch in range(L_img.shape[2]):
            L_ch_255 = L_img_255[:, :, ch]
            H_ch_255 = H_img_255[:, :, ch]
            
            # 扩展为3通道RGB图像以供度量函数使用
            L_rgb = np.stack([L_ch_255]*3, axis=2)
            H_rgb = np.stack([H_ch_255]*3, axis=2)
            
            psnr_global = util.calculate_psnr(L_rgb, H_rgb)
            ssim_global = util.calculate_ssim(L_rgb, H_rgb)
            lpips_global = util.calculate_lpips(L_rgb, H_rgb)
            
            metrics_sum_global['psnr'] += psnr_global
            metrics_sum_global['ssim'] += ssim_global
            metrics_sum_global['lpips'] += lpips_global
        
        # --- 局部归一化处理 ---
        max_val_local = np.max(H_img) * 1  # 设置为当前H最大值的110%
        min_val_local = 0  # 固定最小值为0
        
        for ch in range(L_img.shape[2]):
            # 先clip到[0, max_val_local]
            L_ch_clipped = np.clip(L_img[:,:,ch], 0, max_val_local)
            H_ch_clipped = np.clip(H_img[:,:,ch], 0, max_val_local)
            
            # 归一化到[0,255]
            L_ch_local = (L_ch_clipped / max_val_local * 255).astype(np.uint8)
            H_ch_local = (H_ch_clipped / max_val_local * 255).astype(np.uint8)
            
            # 扩展为3通道RGB图像以供度量函数使用
            L_rgb_local = np.stack([L_ch_local]*3, axis=2)
            H_rgb_local = np.stack([H_ch_local]*3, axis=2)
            
            psnr_local = util.calculate_psnr(L_rgb_local, H_rgb_local)
            ssim_local = util.calculate_ssim(L_rgb_local, H_rgb_local)
            lpips_local = util.calculate_lpips(L_rgb_local, H_rgb_local)
            
            metrics_sum_local['psnr'] += psnr_local
            metrics_sum_local['ssim'] += ssim_local
            metrics_sum_local['lpips'] += lpips_local
        
        # 创建可视化图像
        fig = plt.figure(figsize=(8, 20))
        gs = plt.GridSpec(2, 4, height_ratios=[1, 1], width_ratios=[1, 1, 1, 0.05])
        
        titles = {
            'L': 'Input (L)',
            'H': 'Ground Truth (H)'
        }
        
        sample_imgs_for_vis = {'L': L_img, 'H': H_img}
        vmax_vis = max_val_local
        vmin_vis = min_val_local
        
        # 添加大标题显示PSNR和SSIM
        plt.suptitle(f'PSNR: {psnr_local:.2f}dB, SSIM: {ssim_local:.4f}', fontsize=16)
        
        for row, view in enumerate(['Anterior', 'Posterior']):
            for col, (key, title) in enumerate(titles.items()):
                ax = plt.subplot(gs[row, col])
                im = ax.imshow(sample_imgs_for_vis[key][:,:,row], cmap='gray', vmin=vmin_vis, vmax=vmax_vis)
                ax.set_title(f'{title} - {view}')
                ax.axis('off')
        
        cax = plt.subplot(gs[:, 3])
        plt.colorbar(im, cax=cax)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        fig.canvas.draw()
        img_array = np.array(fig.canvas.renderer.buffer_rgba())
        visuals_list.append(img_array)
        plt.close(fig)
    
    # 计算平均值
    total_channels = count * L_img.shape[2] if count > 0 and L_img.ndim == 3 else count
    if total_channels == 0:
        metrics_avg = {
            'psnr_global': 0, 'ssim_global': 0, 'lpips_global': 0,
            'psnr_local': 0, 'ssim_local': 0, 'lpips_local': 0
        }
    else:
        metrics_avg = {
            'psnr_global': metrics_sum_global['psnr'] / total_channels,
            'ssim_global': metrics_sum_global['ssim'] / total_channels,
            'lpips_global': metrics_sum_global['lpips'] / total_channels,
            'psnr_local': metrics_sum_local['psnr'] / total_channels,
            'ssim_local': metrics_sum_local['ssim'] / total_channels,
            'lpips_local': metrics_sum_local['lpips'] / total_channels
        }
    
    return metrics_avg, visuals_list, image_names

def main(json_path='SPECToptions/train_rrdbnet_psnr_8x.json'):
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default=json_path, help='Path to option JSON file.')
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=False)
    
    device = torch.device('cuda' if opt.get('gpu_ids') is not None else 'cpu')

    # 评估所有数据集
    datasets = ['test', 'val_1', 'val_2']
    all_results = {}
    
    print("\n=== Evaluation Results ===\n")
    
    for dataset_name in datasets:
        if dataset_name in opt['datasets']:
            dataset_opt = opt['datasets'][dataset_name]
            metrics_avg, visuals_list, image_names = evaluate_dataset(dataset_opt, device)
            
            if metrics_avg is None:
                continue
                
            all_results[dataset_name] = {
                'metrics': metrics_avg,
                'visuals': visuals_list,
                'names': image_names
            }
            
            # 打印结果
            print(f"Dataset: {dataset_name}")
            print("Global Normalization:")
            print(f"  PSNR:  {metrics_avg['psnr_global']:.2f} dB")
            print(f"  SSIM:  {metrics_avg['ssim_global']:.4f}")
            print(f"  LPIPS: {metrics_avg['lpips_global']:.4f}")
            
            print("Local Normalization:")
            print(f"  PSNR:  {metrics_avg['psnr_local']:.2f} dB")
            print(f"  SSIM:  {metrics_avg['ssim_local']:.4f}")
            print(f"  LPIPS: {metrics_avg['lpips_local']:.4f}")
            print()

if __name__ == '__main__':
    main() 