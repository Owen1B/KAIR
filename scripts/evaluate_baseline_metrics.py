import sys
import os

# 将项目根目录 (KAIR) 添加到 Python 的模块搜索路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import math

from utils import utils_image as util
from utils import utils_option as option
from utils import utils_spect # For denormalization
from data.select_dataset import define_Dataset

'''
# --------------------------------------------
# 计算测试数据集中输入(L)和目标(H)图像间的评价指标
# (支持全局归一化和局部归一化两种方式)
# --------------------------------------------
'''

def main(json_path='SPECToptions/train_drunet.json'):
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default=json_path, help='Path to option JSON file.')
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=False) # is_train=False as we are only testing

    print(f"Starting evaluation of L vs H metrics for: {args.opt}")
    if 'datasets' in opt and 'test' in opt['datasets']:
        print(option.dict2str(opt['datasets']['test']))
    else:
        print("ERROR: Test dataset configuration not found or incomplete in options file.")
        return

    device = torch.device('cuda' if opt.get('gpu_ids') is not None else 'cpu')
    print(f"Using device: {device}")

    dataset_opt = opt['datasets'].get('test')
    if not dataset_opt: # Ensure dataset_opt is not None before passing to define_Dataset
        print("ERROR: Test dataset configuration (dataset_opt) is None.")
        return

    test_set = define_Dataset(dataset_opt)
    test_loader = DataLoader(test_set, batch_size=1,
                             shuffle=False, num_workers=1,
                             drop_last=False, pin_memory=True)
    print(f"Number of test images: {len(test_set)}")

    # --- Pass 1: Collect all denormalized images and their names ---
    all_L_images_denorm = []
    all_H_images_denorm = []
    all_image_names = []
    norm_params = opt['datasets']['test'].get('normalization')
    if not norm_params:
        print("ERROR: Normalization parameters (opt['datasets']['test']['normalization']) not found. Cannot proceed with denormalization.")
        return

    print("\n--- Pass 1: Collecting and denormalizing all test images ---")
    for i, test_data in enumerate(test_loader):
        image_name_ext = os.path.basename(test_data['L_path'][0])
        image_name = os.path.splitext(image_name_ext)[0]
        all_image_names.append(image_name)
        print(f"Collecting image {i+1}/{len(test_set)}: {image_name}")

        L_tensor = test_data['L'].squeeze(0).cpu().float()
        H_tensor = test_data['H'].squeeze(0).cpu().float()

        L_img_hwc = np.transpose(L_tensor.numpy(), (1, 2, 0))
        H_img_hwc = np.transpose(H_tensor.numpy(), (1, 2, 0))
        
        L_img_denorm = utils_spect.denormalize_spect(L_img_hwc, norm_params)
        H_img_denorm = utils_spect.denormalize_spect(H_img_hwc, norm_params)
        
        all_L_images_denorm.append(L_img_denorm)
        all_H_images_denorm.append(H_img_denorm)

    if not all_L_images_denorm:
        print("No images were collected. Exiting.")
        return

    # --- Calculate Global Min/Max for Normalization ---
    print("\n--- Calculating global min/max for normalization ---")
    # Consider only L and H for global range, similar to model_base.py
    all_pixel_values_for_global_norm = []
    for l_img, h_img in zip(all_L_images_denorm, all_H_images_denorm):
        all_pixel_values_for_global_norm.append(l_img.flatten())
        all_pixel_values_for_global_norm.append(h_img.flatten())
    
    if not all_pixel_values_for_global_norm:
        print("ERROR: No pixel values collected for global normalization. Exiting.")
        return
        
    global_min_val = np.min(np.concatenate(all_pixel_values_for_global_norm))
    global_max_val = np.max(np.concatenate(all_pixel_values_for_global_norm))
    print(f"Global Min: {global_min_val:.4f}, Global Max: {global_max_val:.4f}")

    if global_max_val == global_min_val:
        print("WARNING: Global min and max are equal. Global metrics will likely be Inf/NaN or 0. Proceeding with caution.")
        # Avoid division by zero later by making them slightly different if they are equal.
        # This is a simple fix; a more robust solution might involve skipping global metrics.
        if global_max_val == global_min_val:
            global_max_val += 1e-6 # Add a small epsilon if they are exactly the same and non-zero
            if global_max_val == 0 and global_min_val == 0: # if both are zero
                 global_max_val = 1.0 # or some other non-zero range


    # --- Pass 2: Calculate metrics using both global and local normalization ---
    print("\n--- Pass 2: Calculating metrics ---")
    metrics_sum_global = {'psnr': 0, 'ssim': 0, 'lpips': 0}
    metrics_sum_local = {'psnr': 0, 'ssim': 0, 'lpips': 0}
    total_channels_processed_global = 0
    total_channels_processed_local = 0
    processed_images_count = 0

    for idx, (L_img_denorm, H_img_denorm) in enumerate(zip(all_L_images_denorm, all_H_images_denorm)):
        image_name = all_image_names[idx]
        processed_images_count +=1
        print(f"Processing metrics for image {idx+1}/{len(all_L_images_denorm)}: {image_name}")
        
        num_channels = L_img_denorm.shape[2]

        # Local normalization parameters for this image pair
        local_min_val = min(np.min(L_img_denorm), np.min(H_img_denorm))
        local_max_val = max(np.max(L_img_denorm), np.max(H_img_denorm))
        
        local_norm_valid = True
        if local_max_val == local_min_val:
            print(f"  WARNING: Local min and max are equal for {image_name}. Skipping local metrics for this image.")
            local_norm_valid = False
            # Define local_max_val_calc to a safe value.
            # This value won't be used for normalization if local_norm_valid is False.
            local_max_val_calc = local_max_val + 1e-6 
            if local_max_val == 0 and local_min_val == 0: 
                 local_max_val_calc = 1.0
        else:
            # If local_max_val != local_min_val, local_norm_valid remains True.
            # local_max_val_calc is local_max_val, ensuring denominator is non-zero.
            local_max_val_calc = local_max_val


        for ch_idx in range(num_channels):
            L_ch_original = L_img_denorm[:, :, ch_idx]
            H_ch_original = H_img_denorm[:, :, ch_idx]

            # Calculate Global Metrics
            if global_max_val > global_min_val: # Proceed only if global range is valid
                L_ch_255_global = np.clip(((L_ch_original - global_min_val) / (global_max_val - global_min_val) * 255), 0, 255).astype(np.uint8)
                H_ch_255_global = np.clip(((H_ch_original - global_min_val) / (global_max_val - global_min_val) * 255), 0, 255).astype(np.uint8)
                L_rgb_global = np.stack([L_ch_255_global]*3, axis=2)
                H_rgb_global = np.stack([H_ch_255_global]*3, axis=2)
                try:
                    metrics_sum_global['psnr'] += util.calculate_psnr(L_rgb_global, H_rgb_global)
                    metrics_sum_global['ssim'] += util.calculate_ssim(L_rgb_global, H_rgb_global)
                    metrics_sum_global['lpips'] += util.calculate_lpips(L_rgb_global, H_rgb_global)
                    total_channels_processed_global += 1
                except Exception as e:
                    print(f"  ERROR (Global): Calculating metrics for {image_name}, channel {ch_idx}: {e}")
            
            # Calculate Local Metrics
            if local_norm_valid:
                # Use local_max_val_calc to avoid division by zero if local_max_val == local_min_val
                # but we already checked local_norm_valid, so this is more of a safeguard or if we change logic
                L_ch_255_local = np.clip(((L_ch_original - local_min_val) / (local_max_val_calc - local_min_val) * 255), 0, 255).astype(np.uint8)
                H_ch_255_local = np.clip(((H_ch_original - local_min_val) / (local_max_val_calc - local_min_val) * 255), 0, 255).astype(np.uint8)
                L_rgb_local = np.stack([L_ch_255_local]*3, axis=2)
                H_rgb_local = np.stack([H_ch_255_local]*3, axis=2)
                try:
                    metrics_sum_local['psnr'] += util.calculate_psnr(L_rgb_local, H_rgb_local)
                    metrics_sum_local['ssim'] += util.calculate_ssim(L_rgb_local, H_rgb_local)
                    metrics_sum_local['lpips'] += util.calculate_lpips(L_rgb_local, H_rgb_local)
                    total_channels_processed_local += 1
                except Exception as e:
                    print(f"  ERROR (Local): Calculating metrics for {image_name}, channel {ch_idx}: {e}")

    # --- Averaging and Printing Results ---
    print("\n--- Average Metrics (L vs H) ---")
    print(f"Processed {processed_images_count} images in total for metrics calculation.")

    if total_channels_processed_global > 0:
        avg_psnr_global = metrics_sum_global['psnr'] / total_channels_processed_global
        avg_ssim_global = metrics_sum_global['ssim'] / total_channels_processed_global
        avg_lpips_global = metrics_sum_global['lpips'] / total_channels_processed_global
        print("\nGlobal Normalization Metrics:")
        print(f"  Average PSNR:  {avg_psnr_global:.2f} dB ({total_channels_processed_global} channels)")
        print(f"  Average SSIM:  {avg_ssim_global:.4f} ({total_channels_processed_global} channels)")
        print(f"  Average LPIPS: {avg_lpips_global:.4f} (Lower is better) ({total_channels_processed_global} channels)")
    else:
        print("\nGlobal Normalization Metrics: No channels processed or global range was invalid.")

    if total_channels_processed_local > 0:
        avg_psnr_local = metrics_sum_local['psnr'] / total_channels_processed_local
        avg_ssim_local = metrics_sum_local['ssim'] / total_channels_processed_local
        avg_lpips_local = metrics_sum_local['lpips'] / total_channels_processed_local
        print("\nLocal (Self) Normalization Metrics:")
        print(f"  Average PSNR:  {avg_psnr_local:.2f} dB ({total_channels_processed_local} channels)")
        print(f"  Average SSIM:  {avg_ssim_local:.4f} ({total_channels_processed_local} channels)")
        print(f"  Average LPIPS: {avg_lpips_local:.4f} (Lower is better) ({total_channels_processed_local} channels)")
    else:
        print("\nLocal (Self) Normalization Metrics: No channels processed or all images had zero local range.")

if __name__ == '__main__':
    main() 