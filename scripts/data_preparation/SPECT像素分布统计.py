#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# ============================================================================
# SPECT像素分布统计分析工具
# ============================================================================
# 
# 脚本功能：
#   - 统计SPECT数据的像素值分布特征
#   - 比较原始数据与添加泊松噪声后的分布差异
#   - 分析图像整体像素和的统计特性
#   - 生成对比直方图用于可视化分析
# 
# 主要用途：
#   - 数据预处理质量评估
#   - 噪声特性分析
#   - 数据集统计特征记录
#   - 为模型训练提供数据分布参考
# 
# 数据格式：
#   - 输入：.dat文件，形状为(2, 1024, 256)的float32数组
#   - 输出：统计信息打印和分布对比图
# 
# 作者：KAIR项目组
# 创建时间：2024年
# ============================================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 设置原始数据目录
raw_data_dir = 'trainsets/spectH_ideal'  # 根据实际情况修改
# raw_data_dir = 'testsets/spectH'

# 获取所有.dat文件路径
file_list = [os.path.join(raw_data_dir, f) for f in os.listdir(raw_data_dir) if f.endswith('.dat')]

if not file_list:
    print(f"未在 {raw_data_dir} 找到 .dat 文件！")
    sys.exit(1)

all_pixels = []
all_pixels_noisy = []
image_sums = []
image_sums_noisy = []

for file_path in tqdm(file_list, desc='统计像素值'):
    # 读取原始数据
    data = np.fromfile(file_path, dtype=np.float32)
    all_pixels.append(data)
    image_sums.append(np.sum(data))
    
    # 添加泊松噪声
    noisy_data = np.random.poisson(data)
    all_pixels_noisy.append(noisy_data)
    image_sums_noisy.append(np.sum(noisy_data))

all_pixels = np.concatenate(all_pixels)
all_pixels_noisy = np.concatenate(all_pixels_noisy)
image_sums = np.array(image_sums)
image_sums_noisy = np.array(image_sums_noisy)

# 原始数据统计信息
print(f"原始像素值统计：")
print(f"最小值: {np.min(all_pixels)}")
print(f"最大值: {np.max(all_pixels)}")
print(f"均值: {np.mean(all_pixels)}")
print(f"标准差: {np.std(all_pixels)}")

# 添加噪声后统计信息
print(f"\n添加泊松噪声后像素值统计：")
print(f"最小值: {np.min(all_pixels_noisy)}")
print(f"最大值: {np.max(all_pixels_noisy)}")
print(f"均值: {np.mean(all_pixels_noisy)}")
print(f"标准差: {np.std(all_pixels_noisy)}")

# 每张图像素和统计
print("\n每张图像的像素值之和统计：")
print(f"原始数据 - 数量: {len(image_sums)}, 最小和: {np.min(image_sums)}, 最大和: {np.max(image_sums)}, 均值: {np.mean(image_sums)}, 标准差: {np.std(image_sums)}")
print(f"添加噪声 - 数量: {len(image_sums_noisy)}, 最小和: {np.min(image_sums_noisy)}, 最大和: {np.max(image_sums_noisy)}, 均值: {np.mean(image_sums_noisy)}, 标准差: {np.std(image_sums_noisy)}")

# 绘制像素值直方图对比
plt.figure(figsize=(12,6))
plt.hist(all_pixels, bins=100, alpha=0.5, color='skyblue', label='原始数据', edgecolor='black')
plt.hist(all_pixels_noisy, bins=100, alpha=0.5, color='red', label='添加泊松噪声', edgecolor='black')
plt.title('SPECT数据像素值分布对比')
plt.xlabel('像素值')
plt.ylabel('频数')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('spect_pixel_distribution_comparison.png', dpi=200)
plt.show()

# 绘制每张图像素和直方图对比
plt.figure(figsize=(12,6))
plt.hist(image_sums, bins=30, alpha=0.5, color='orange', label='原始数据', edgecolor='black')
plt.hist(image_sums_noisy, bins=30, alpha=0.5, color='green', label='添加泊松噪声', edgecolor='black')
plt.title('每张SPECT图像像素值之和分布对比')
plt.xlabel('像素值之和')
plt.ylabel('频数')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('spect_image_sum_distribution_comparison.png', dpi=200)
plt.show()