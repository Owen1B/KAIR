import os
import numpy as np
from tqdm import tqdm
import sys
'''
# --------------------------------------------
# SPECT泊松图预处理脚本
# --------------------------------------------
# 该脚本用于处理SPECT泊松图数据，生成二项重采样图像
# 主要功能：
# 1. 读取泊松图数据
# 2. 进行二项重采样：
#    - 4x binomial (0.25倍采样)
#    - 8x binomial (0.125倍采样)
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

def process_single_file(file_path, output_dirs):
    """
    处理单个SPECT泊松图文件

    参数:
        file_path: 输入原始文件路径
        output_dirs: 输出目录字典，包含所有输出路径
    """
    # 读取.dat文件
    data = np.fromfile(file_path, dtype=np.float32)
    data = data.reshape(2, 1024, 256)
    anterior = data[0]
    posterior = data[1]

    # 对泊松图进行二项重采样
    # 4x binomial (0.25倍采样)
    binomial_4x = np.stack([
        np.random.binomial(anterior.astype(np.int32), 0.25).astype(np.float32) * 4,
        np.random.binomial(posterior.astype(np.int32), 0.25).astype(np.float32) * 4
    ], axis=0)

    # 8x binomial (0.125倍采样)
    binomial_8x = np.stack([
        np.random.binomial(anterior.astype(np.int32), 0.125).astype(np.float32) * 8,
        np.random.binomial(posterior.astype(np.int32), 0.125).astype(np.float32) * 8
    ], axis=0)

    # 保存处理后的数据
    base_name = os.path.basename(file_path)
    
    # 保存二项重采样图
    binomial_4x.astype(np.float32).tofile(os.path.join(output_dirs['binomial_4x'], base_name))
    binomial_8x.astype(np.float32).tofile(os.path.join(output_dirs['binomial_8x'], base_name))

def main():
    # =========================================
    # 配置参数
    # =========================================
    config = {
        'input_dir': 'SPECTdatasets/raw_data',  # 输入数据目录（泊松图）
        'output_base_dir': 'SPECTdatasets',  # 输出基础目录
    }

    # 创建所有输出目录
    output_dirs = {
        'binomial_4x': os.path.join(config['output_base_dir'], 'spectL_XCAT_binomial_4x'),
        'binomial_8x': os.path.join(config['output_base_dir'], 'spectL_XCAT_binomial_8x'),
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
        process_single_file(file_path, output_dirs)

    print(f'处理完成！')
    print('生成的数据集：')
    print('1. 二项重采样图：')
    print('   - spectL_XCAT_binomial_4x: 1x poisson的0.25倍采样')
    print('   - spectL_XCAT_binomial_8x: 1x poisson的0.125倍采样')

if __name__ == '__main__':
    main() 