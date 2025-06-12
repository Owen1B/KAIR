#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# ============================================================================
# SPECT数据BM3D降噪处理工具
# ============================================================================
# 
# 脚本功能：
#   - 对含噪声的SPECT数据进行BM3D降噪处理
#   - 采用Anscombe变换 + BM3D降噪 + 逆Anscombe变换流程
#   - 独立处理前位和后位投影图像
#   - 支持批量处理多个.dat文件
# 
# 主要用途：
#   - 临床SPECT数据降噪预处理
#   - 为深度学习模型提供降噪后的对比数据
#   - 评估传统降噪算法效果
# 
# 数据格式：
#   - 输入：含噪.dat文件，形状为(2, 1024, 256)的float32数组
#   - 输出：降噪后的.dat文件，保持相同格式
# 
# 作者：KAIR项目组
# 创建时间：2024年
# ============================================================================
"""

import os
import numpy as np
from tqdm import tqdm
import sys
import bm3d

'''
# --------------------------------------------
# SPECT含噪图 BM3D降噪脚本
# --------------------------------------------
# 该脚本用于处理含噪声的SPECT .dat 文件，应用Anscombe变换，
# 然后使用BM3D进行降噪，最后应用逆Anscombe变换。
# 
# 主要功能：
# 1. 读取输入目录中的每个.dat文件 (假设为前后位数据)。
# 2. 对前后位图像独立进行以下处理：
#    a. Anscombe 变换
#    b. BM3D 降噪 (sigma_psd 通常设为1.0)
#    c. 逆 Anscombe 变换
# 3. 将处理后的数据保存到输出目录，文件名与原文件相同。
# --------------------------------------------
# 数据格式：
# - 输入：含噪 .dat 文件，形状为 (2, 1024, 256) 的 float32 数组
# - 输出：降噪后的 .dat 文件，保持相同格式
# --------------------------------------------
'''

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

# 尝试导入 get_image_paths，如果失败则提供备选方案或提示
try:
    from utils import utils_image as util
    print("成功从 utils.utils_image 导入 util.")
except ImportError:
    print("警告: 无法从 utils.utils_image 导入 util. " \
          "如果脚本无法找到 .dat 文件, 请确保 utils 模块在 Python 路径中, " \
          "或者修改脚本以使用其他文件查找方法.")
    # 提供一个简单的备选文件查找函数，如果导入失败
    class util:
        @staticmethod
        def get_image_paths(directory, extensions=['.dat']):
            paths = []
            if not os.path.isdir(directory):
                print(f"错误: 目录 '{directory}' 不存在.")
                return paths
            for root, _, files in os.walk(directory):
                for file_name in files:
                    if any(file_name.lower().endswith(ext) for ext in extensions):
                        paths.append(os.path.join(root, file_name))
            return paths

def _anscombe_transform(image):
    """直接实现Anscombe变换"""
    return 2.0 * np.sqrt(np.maximum(0, image) + 3.0/8.0)

def _inverse_anscombe_transform(image_ansc):
    """直接实现逆Anscombe变换"""
    return (image_ansc / 2.0)**2 - 3.0/8.0

def process_single_noisy_file(file_path, output_dir, base_name, bm3d_sigma_psd):
    """
    处理单个含噪SPECT文件 (Anscombe -> BM3D -> InvAnscombe)

    参数:
        file_path (str): 输入的含噪.dat文件路径.
        output_dir (str): 保存降噪后文件的目录.
        base_name (str): 输出文件的名称 (通常与输入文件相同).
        bm3d_sigma_psd (float): BM3D算法的噪声标准差参数.
    """
    try:
        data = np.fromfile(file_path, dtype=np.float32)
        if data.size != 2 * 1024 * 256:
            print(f"警告: 文件 {file_path} 的大小 ({data.size}) "
                  f"与预期的 2*1024*256 不符. 跳过此文件.")
            return
        data = data.reshape(2, 1024, 256)
    except FileNotFoundError:
        print(f"错误: 文件 {file_path} 未找到. 跳过此文件.")
        return
    except ValueError as ve:
        print(f"错误: 重塑文件 {file_path} 内容时出错: {ve}. 跳过此文件.")
        return
    except Exception as e:
        print(f"错误: 读取或重塑文件 {file_path} 失败: {e}. 跳过此文件.")
        return

    anterior_noisy = data[0]
    posterior_noisy = data[1]

    # 处理前位图像
    anterior_ansc = _anscombe_transform(anterior_noisy)
    anterior_bm3d_denoised_ansc = bm3d.bm3d(anterior_ansc.astype(np.float64), 
                                            sigma_psd=bm3d_sigma_psd, 
                                            profile='np') # 使用 np profile
    anterior_denoised = _inverse_anscombe_transform(anterior_bm3d_denoised_ansc)
    
    # 处理后位图像
    posterior_ansc = _anscombe_transform(posterior_noisy)
    posterior_bm3d_denoised_ansc = bm3d.bm3d(posterior_ansc.astype(np.float64), 
                                             sigma_psd=bm3d_sigma_psd, 
                                             profile='np') # 使用 np profile
    posterior_denoised = _inverse_anscombe_transform(posterior_bm3d_denoised_ansc)
    
    denoised_data = np.stack([anterior_denoised, posterior_denoised], axis=0)

    output_file_path = os.path.join(output_dir, base_name)
    try:
        denoised_data.astype(np.float32).tofile(output_file_path)
    except Exception as e:
        print(f"错误: 写入文件 {output_file_path} 失败: {e}")

def main():
    # =========================================
    # 配置参数
    # =========================================
    config = {
        'input_dir': "SPECTdatasets/spectH_clinical",  # 输入含噪数据目录
        'output_dir': "SPECTdatasets/spectH_clinical_bm3d_1x", # 输出降噪数据目录
        'bm3d_sigma_psd': 1.0  # BM3D的sigma_psd参数, 对于Anscombe变换后的数据通常为1.0
    }

    print(f"脚本开始执行...")
    print(f"输入目录 (含噪数据): {config['input_dir']}")
    print(f"输出目录 (降噪数据): {config['output_dir']}")
    print(f"BM3D sigma_psd: {config['bm3d_sigma_psd']}")

    if not os.path.exists(config['output_dir']):
        try:
            os.makedirs(config['output_dir'], exist_ok=True)
            print(f"已创建输出目录: {config['output_dir']}")
        except OSError as e:
            print(f"错误: 创建输出目录 {config['output_dir']} 失败: {e}")
            return
    else:
        print(f"输出目录已存在: {config['output_dir']}")

    file_paths = util.get_image_paths(config['input_dir'])
    if not file_paths:
        print(f"错误: 在目录 '{config['input_dir']}' 中未找到任何 .dat 文件.")
        return

    print(f"在输入目录中找到 {len(file_paths)} 个 .dat 文件准备处理.")
    
    for file_path in tqdm(file_paths, desc='对含噪数据应用BM3D降噪'):
        base_name = os.path.basename(file_path)
        process_single_noisy_file(file_path, 
                                  config['output_dir'], 
                                  base_name, 
                                  config['bm3d_sigma_psd'])

    print(f"\n处理完成！")
    try:
        actual_output_files = len([name for name in os.listdir(config['output_dir']) if name.endswith('.dat')])
        print(f"输入文件总数: {len(file_paths)}")
        print(f"实际在输出目录 '{config['output_dir']}' 中找到的 .dat 文件数: {actual_output_files}")
        if len(file_paths) != actual_output_files:
            print("警告: 输入与输出文件数量不符, 请检查是否有错误信息.")
    except FileNotFoundError:
        print(f"无法验证输出文件数量, 因为输出目录 '{config['output_dir']}' 未找到或无法访问.")
    except Exception as e:
        print(f"在验证输出文件数量时发生错误: {e}")

if __name__ == '__main__':
    main() 