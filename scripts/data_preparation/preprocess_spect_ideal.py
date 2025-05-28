import os
import numpy as np
from tqdm import tqdm
import sys
import bm3d # 导入bm3d
'''
# --------------------------------------------
# SPECT理想图预处理脚本
# --------------------------------------------
# 该脚本用于处理SPECT理想图数据，生成不同剂量和噪声水平的图像
# 主要功能：
# 1. 从理想图生成不同剂量水平：
#    - 理想图1x (原始理想图)
#    - 理想图0.25x (原始理想图/4)
#    - 理想图0.125x (原始理想图/8)
# 2. 对每个剂量水平添加泊松噪声：
#    - 1x poisson
#    - 4x poisson
#    - 8x poisson
# 3. 对1x poisson进行二项重采样：
#    - 4x binomial (0.25倍采样)
#    - 8x binomial (0.125倍采样)
# 4. 对1x poisson进行Anscombe+BM3D+Inverse Anscombe处理:
#    - bm3d_1x (前后位拼接后处理)
# --------------------------------------------
# 数据格式：
# - 输入：原始 .dat 文件，形状为 (2, 1024, 256) 的 float32 数组
# - 输出：处理后的 .dat 文件，保持相同格式
# --------------------------------------------
'''

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

import utils.utils_image as util

def _anscombe_transform(image):
    """直接实现Anscombe变换"""
    return 2.0 * np.sqrt(np.maximum(0, image) + 3.0/8.0)

def _inverse_anscombe_transform(image_ansc):
    """直接实现逆Anscombe变换"""
    return (image_ansc / 2.0)**2 - 3.0/8.0

def process_single_file(file_path, output_dirs, base_name):
    """
    处理单个SPECT理想图文件

    参数:
        file_path: 输入原始文件路径
        output_dirs: 输出目录字典，包含所有输出路径
        base_name: 输出文件的基本名称
    """
    # 读取.dat文件
    data = np.fromfile(file_path, dtype=np.float32)
    data = data.reshape(2, 1024, 256)
    data = data * 1  # 新增：整体缩放
    anterior = data[0]
    posterior = data[1]

    # 1. 生成不同剂量水平的理想图
    ideal_1x = np.stack([anterior, posterior], axis=0)  # 原始理想图
    ideal_025x = np.stack([anterior/4, posterior/4], axis=0)  # 0.25倍理想图
    ideal_0125x = np.stack([anterior/8, posterior/8], axis=0)  # 0.125倍理想图

    # 2. 对每个剂量水平添加泊松噪声
    # 1x poisson
    poisson_1x_original = np.stack([
        np.random.poisson(np.maximum(0, anterior)).astype(np.float32),
        np.random.poisson(np.maximum(0, posterior)).astype(np.float32)
    ], axis=0)

    # 4x poisson (0.25倍理想图 + 泊松噪声)
    poisson_4x = np.stack([
        np.random.poisson(np.maximum(0, anterior/4)).astype(np.float32) * 4,
        np.random.poisson(np.maximum(0, posterior/4)).astype(np.float32) * 4
    ], axis=0)

    # 8x poisson (0.125倍理想图 + 泊松噪声)
    poisson_8x = np.stack([
        np.random.poisson(np.maximum(0, anterior/8)).astype(np.float32) * 8,
        np.random.poisson(np.maximum(0, posterior/8)).astype(np.float32) * 8
    ], axis=0)

    # 3. 对1x poisson进行二项重采样
    # 4x binomial (0.25倍采样)
    binomial_4x = np.stack([
        np.random.binomial(poisson_1x_original[0].astype(np.int32), 0.25).astype(np.float32) * 4,
        np.random.binomial(poisson_1x_original[1].astype(np.int32), 0.25).astype(np.float32) * 4
    ], axis=0)

    # 8x binomial (0.125倍采样)
    binomial_8x = np.stack([
        np.random.binomial(poisson_1x_original[0].astype(np.int32), 0.125).astype(np.float32) * 8,
        np.random.binomial(poisson_1x_original[1].astype(np.int32), 0.125).astype(np.float32) * 8
    ], axis=0)

    # 4. 对poisson_1x_original进行Anscombe+BM3D+Inverse Anscombe处理 (前后位拼接)
    H_anterior_raw_poisson = poisson_1x_original[0, ...]
    H_posterior_raw_poisson = poisson_1x_original[1, ...]
    
    # --- 修改开始: 独立处理前后位图像的BM3D降噪 ---
    # 处理前位图像
    H_anterior_ansc = _anscombe_transform(H_anterior_raw_poisson)
    H_anterior_bm3d_denoised = bm3d.bm3d(H_anterior_ansc.astype(np.float64), sigma_psd=1.0, profile='np') # 添加 profile='np' 保持一致性
    H_anterior_denoised_bm3d = _inverse_anscombe_transform(H_anterior_bm3d_denoised)
    
    # 处理后位图像
    H_posterior_ansc = _anscombe_transform(H_posterior_raw_poisson)
    H_posterior_bm3d_denoised = bm3d.bm3d(H_posterior_ansc.astype(np.float64), sigma_psd=1.0, profile='np') # 添加 profile='np' 保持一致性
    H_posterior_denoised_bm3d = _inverse_anscombe_transform(H_posterior_bm3d_denoised)
    # --- 修改结束 ---
    
    # 重新堆叠处理后的H图像
    bm3d_1x = np.stack([H_anterior_denoised_bm3d, H_posterior_denoised_bm3d], axis=0)

    # 保存所有处理后的数据
    
    # 保存理想图
    ideal_1x.astype(np.float32).tofile(os.path.join(output_dirs['ideal_1x'], base_name))
    ideal_025x.astype(np.float32).tofile(os.path.join(output_dirs['ideal_4x'], base_name))
    ideal_0125x.astype(np.float32).tofile(os.path.join(output_dirs['ideal_8x'], base_name))
    
    # 保存泊松噪声图
    poisson_1x_original.astype(np.float32).tofile(os.path.join(output_dirs['poisson_1x'], base_name))
    poisson_4x.astype(np.float32).tofile(os.path.join(output_dirs['poisson_4x'], base_name))
    poisson_8x.astype(np.float32).tofile(os.path.join(output_dirs['poisson_8x'], base_name))
    
    # 保存二项重采样图
    binomial_4x.astype(np.float32).tofile(os.path.join(output_dirs['binomial_4x'], base_name))
    binomial_8x.astype(np.float32).tofile(os.path.join(output_dirs['binomial_8x'], base_name))

    bm3d_1x.astype(np.float32).tofile(os.path.join(output_dirs['bm3d_1x'], base_name)) # 保存新的BM3D处理数据

def main():
    # =========================================
    # 配置参数
    # =========================================
    config = {
        'input_dir': 'SPECTdatasets/raw_data',  # 输入数据目录（理想图）
        'output_base_dir': 'SPECTdatasets',  # 输出基础目录
        'expand_factor': 1,  # 扩大倍数，默认为1（不扩大）
    }

    # 创建所有输出目录
    output_dirs = {
        'ideal_1x': os.path.join(config['output_base_dir'], 'spectH_XCAT_ideal_1x'),
        'ideal_4x': os.path.join(config['output_base_dir'], 'spectL_XCAT_ideal_4x'),
        'ideal_8x': os.path.join(config['output_base_dir'], 'spectL_XCAT_ideal_8x'),
        'poisson_1x': os.path.join(config['output_base_dir'], 'spectH_XCAT_poisson_1x'),
        'poisson_4x': os.path.join(config['output_base_dir'], 'spectL_XCAT_poisson_4x'),
        'poisson_8x': os.path.join(config['output_base_dir'], 'spectL_XCAT_poisson_8x'),
        'binomial_4x': os.path.join(config['output_base_dir'], 'spectL_XCAT_binomial_4x'),
        'binomial_8x': os.path.join(config['output_base_dir'], 'spectL_XCAT_binomial_8x'),
        'bm3d_1x': os.path.join(config['output_base_dir'], 'spectH_XCAT_bm3d_1x'), # 新增BM3D输出目录
    }

    # 创建所有输出目录
    for dir_path in output_dirs.values():
        os.makedirs(dir_path, exist_ok=True)

    # 获取所有.dat文件
    file_paths = util.get_image_paths(config['input_dir'])
    if not file_paths:
        print(f"错误：在 '{config['input_dir']}' 中未找到任何 .dat 文件。请检查路径。")
        return

    # 处理每个文件
    for file_path in tqdm(file_paths, desc='处理数据'):
        # 对每个文件处理expand_factor次
        for i in range(config['expand_factor']):
            # 生成新的文件名，如果是复制的话添加后缀
            if config['expand_factor'] > 1:
                base_name = os.path.basename(file_path).split('_')[0]
                suffix = f'_{i+1}' if i > 0 else ''
                new_base_name = f'{base_name}{suffix}.dat'
            else:
                new_base_name = os.path.basename(file_path).split('_')[0] + '.dat'
            
            # 处理文件并保存
            process_single_file(file_path, output_dirs, new_base_name)

    print(f'处理完成！')
    print(f'原始文件数：{len(file_paths)}')
    print(f'扩大倍数：{config["expand_factor"]}')
    print(f'最终文件数：{len(file_paths) * config["expand_factor"]}')
    print('生成的数据集：')
    print('1. 理想图：')
    print('   - spectH_XCAT_ideal_1x: 原始理想图')
    print('   - spectL_XCAT_ideal_4x: 0.25倍理想图')
    print('   - spectL_XCAT_ideal_8x: 0.125倍理想图')
    print('2. 泊松噪声图：')
    print('   - spectH_XCAT_poisson_1x: 原始理想图 + 泊松噪声')
    print('   - spectL_XCAT_poisson_4x: 0.25倍理想图 + 泊松噪声')
    print('   - spectL_XCAT_poisson_8x: 0.125倍理想图 + 泊松噪声')
    print('3. 二项重采样图：')
    print('   - spectL_XCAT_binomial_4x: 1x poisson的0.25倍采样')
    print('   - spectL_XCAT_binomial_8x: 1x poisson的0.125倍采样')
    print('4. BM3D处理图：')
    print('   - spectH_XCAT_bm3d_1x: 1x poisson (前后位拼接) + Anscombe + BM3D(sigma=1) + InvAnscombe')

if __name__ == '__main__':
    main() 