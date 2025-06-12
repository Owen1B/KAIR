#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# ============================================================================
# LPIPS相关性分析工具
# ============================================================================
# 
# 脚本功能：
#   - 计算CSV文件中各列数据与参考列的皮尔逊相关系数
#   - 计算斯皮尔曼等级相关系数及其显著性p值
#   - 分析模型训练过程中不同指标之间的相关性
# 
# 主要用途：
#   - 分析训练指标与验证指标的相关性
#   - 评估不同评价指标之间的一致性
#   - 为模型优化提供统计学依据
# 
# 输入输出：
#   - 输入：包含训练指标的CSV文件
#   - 输出：相关系数分析结果的CSV文件
# 
# 作者：KAIR项目组
# 创建时间：2024年
# ============================================================================
"""

import pandas as pd
import numpy as np
from scipy import stats

def calculate_correlations(csv_path):
    """
    计算CSV文件中数据的皮尔逊相关系数和斯皮尔曼等级相关系数，以及对应的p值
    
    参数:
        csv_path: CSV文件路径
    """
    # 读取CSV文件
    df = pd.read_csv(csv_path)
    
    # 获取所有列名（除了'step'列）
    columns = [col for col in df.columns if col != 'step']
    
    # 创建结果DataFrame
    results = pd.DataFrame(index=columns, columns=['Pearson', 'Pearson_p', 'Spearman', 'Spearman_p'])
    
    # 计算每列与test_lpips_local的相关系数
    reference_col = 'val_2_poisson_lpips_local'
    
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
    
    # 打印结果
    print("\n相关系数计算结果:")
    print("=" * 50)
    print(results.round(4))
    
    # 保存结果到CSV
    output_path = csv_path.replace('.csv', '_correlations.csv')
    results.to_csv(output_path)
    print(f"\n结果已保存到: {output_path}")

if __name__ == '__main__':
    csv_path = "SPECTdenoising/drunet_psnr_8x_linear_96数据集/metrics_correlation.csv"
    calculate_correlations(csv_path)