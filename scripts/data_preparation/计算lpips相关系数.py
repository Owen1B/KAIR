#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# ============================================================================
# LPIPS相关性分析工具
# ============================================================================
# 
# 脚本功能：
#   - 处理多个CSV文件中的指标数据
#   - 计算每个CSV文件中指标之间的皮尔逊相关系数和斯皮尔曼等级相关系数
#   - 合并所有CSV文件的数据并计算整体相关性
# 
# 主要用途：
#   - 比较不同数据集之间的指标相关性
#   - 评估指标在不同数据集上的一致性
#   - 为模型优化提供统计学依据
# 
# 输入输出：
#   - 输入：多个包含训练指标的CSV文件
#   - 输出：打印相关系数分析结果
# 
# 作者：KAIR项目组
# 创建时间：2024年
# ============================================================================
"""

import pandas as pd
import numpy as np
from scipy import stats
import os

def calculate_correlations(csv_paths):
    """
    处理多个CSV文件并计算相关系数
    
    参数:
        csv_paths: CSV文件路径列表
    """
    all_results = {}
    all_data = {}
    
    # 获取所有列名（除了'step'列）
    first_df = pd.read_csv(csv_paths[0])
    columns = [col for col in first_df.columns if col != 'step']
    reference_col = 'val_2_poisson_lpips_local'
    
    print(f"参考列: {reference_col}")
    print(f"所有列: {columns}")
    
    # 处理每个CSV文件
    for i, csv_path in enumerate(csv_paths):
        print(f"\n处理文件 {i+1}: {csv_path}")
        df = pd.read_csv(csv_path)
        
        
        # 创建结果DataFrame
        results = pd.DataFrame(index=columns, columns=['Pearson', 'Pearson_p', 'Spearman', 'Spearman_p'])
        
        # 计算每列与reference_col的相关系数
        for col in columns:
            if col != reference_col:
                # 计算皮尔逊相关系数和p值
                pearson_corr, pearson_p = stats.pearsonr(df[reference_col], df[col])
                
                # 计算斯皮尔曼等级相关系数和p值
                spearman_corr, spearman_p = stats.spearmanr(df[reference_col], df[col])
                
                # 存储结果
                results.loc[col, 'Pearson'] = pearson_corr
                results.loc[col, 'Pearson_p'] = pearson_p
                results.loc[col, 'Spearman'] = spearman_corr
                results.loc[col, 'Spearman_p'] = spearman_p
        
        # 使用唯一键保存结果 - 修复：使用索引避免键重复
        file_key = f"file_{i+1}_{os.path.basename(csv_path)}"
        all_results[file_key] = results
        
        # 存储数据用于后续合并计算 - 修复：使用索引避免键重复
        all_data[file_key] = df
        
        # 打印单个文件的结果
        print(f"\n文件 {i+1} 的相关系数计算结果:")
        print("=" * 50)
        print(results.round(4))
    
    # 合并所有数据并计算整体相关性
    print("\n计算合并后的整体相关性...")
    combined_results = pd.DataFrame(index=columns, columns=['Pearson', 'Pearson_p', 'Spearman', 'Spearman_p'])
    

    
    # 合并所有数据集 - 修复：使用正确的键
    dfs_to_combine = []
    for i, path in enumerate(csv_paths):
        file_key = f"file_{i+1}_{os.path.basename(path)}"
        df = all_data[file_key][columns].copy()
        dfs_to_combine.append(df)
    
    combined_data = pd.concat(dfs_to_combine, ignore_index=True)

    
    # 计算合并后的相关系数
    for col in columns:
        if col != reference_col:
            # 确保数据中没有NaN值
            valid_data = combined_data[[reference_col, col]].dropna()
            if len(valid_data) > 0:
                # 计算皮尔逊相关系数和p值
                pearson_corr, pearson_p = stats.pearsonr(valid_data[reference_col], valid_data[col])
                
                # 计算斯皮尔曼等级相关系数和p值
                spearman_corr, spearman_p = stats.spearmanr(valid_data[reference_col], valid_data[col])
                
                # 存储结果
                combined_results.loc[col, 'Pearson'] = pearson_corr
                combined_results.loc[col, 'Pearson_p'] = pearson_p
                combined_results.loc[col, 'Spearman'] = spearman_corr
                combined_results.loc[col, 'Spearman_p'] = spearman_p


    
    # 打印合并后的结果
    print("\n合并后的整体相关系数计算结果:")
    print("=" * 50)
    print(combined_results.round(4))

if __name__ == '__main__':
    # 示例：处理多个CSV文件
    csv_paths = [
        "SPECTdenoising/drunet_psnr_8x_linear_96数据集_vgg/metrics_correlation.csv",
        "SPECTdenoising/drunet_psnr_8x_linear_96数据集_targetbm3d_vgg/metrics_correlation.csv"
    ]
    calculate_correlations(csv_paths)