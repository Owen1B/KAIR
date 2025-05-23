import os
import numpy as np
from tqdm import tqdm
import sys
import scipy.ndimage
'''
# --------------------------------------------
# SPECT数据预处理脚本
# --------------------------------------------
# 该脚本用于处理SPECT（单光子发射计算机断层扫描）数据
# 主要功能：
# 1. 支持多种数据处理方式：
#    - 理想图像处理：
#      * 可选添加高斯模糊（指定FWHM）
#      * 可选添加泊松噪声
#      * 可选降采样（除以speed系数）后添加泊松噪声
#    - 实际高计数图像处理：
#      * 可选二项重采样
# 2. 生成处理后的数据用于训练
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

def process_single_file(file_path, output_dir, process_config):
    """
    处理单个SPECT数据文件

    参数:
        file_path: 输入原始文件路径
        output_dir: 输出目录
        process_config: 处理配置字典，包含以下选项：
            - image_type: 'ideal' 或 'real'，表示图像类型
            - apply_gaussian: 是否应用高斯模糊
            - fwhm: 高斯模糊的半高宽度（如果apply_gaussian为True）
            - apply_poisson: 是否添加泊松噪声
            - apply_speed: 是否应用降采样
            - speed: 降采样系数（如果apply_speed为True）
            - apply_binomial: 是否应用二项重采样（仅用于real类型）
    """
    # 读取.dat文件
    data = np.fromfile(file_path, dtype=np.float32)
    data = data.reshape(2, 1024, 256)
    anterior = data[0]
    posterior = data[1]

    # 处理前位和后位投影
    if process_config['image_type'] == 'ideal':
        # 理想图像处理
        if process_config['apply_gaussian']:
            sigma = process_config['fwhm'] / 2.355
            anterior = scipy.ndimage.gaussian_filter(anterior, sigma)
            posterior = scipy.ndimage.gaussian_filter(posterior, sigma)

        if process_config['apply_speed']:
            anterior = anterior / process_config['speed']
            posterior = posterior / process_config['speed']

        if process_config['apply_poisson']:
            anterior = np.random.poisson(np.maximum(0, anterior)).astype(np.float32)*process_config['speed']
            posterior = np.random.poisson(np.maximum(0, posterior)).astype(np.float32)*process_config['speed']

    elif process_config['image_type'] == 'real':
        # 实际高计数图像处理
        if process_config['apply_binomial']:
            anterior = np.random.binomial(
                n=anterior.astype(np.int32),
                p=1.0/process_config['speed']
            ) * process_config['speed']
            posterior = np.random.binomial(
                n=posterior.astype(np.int32),
                p=1.0/process_config['speed']
            ) * process_config['speed']
    # 重新组合数据
    output_data = np.stack([anterior, posterior], axis=0)

    # 保存处理后的数据
    output_path = os.path.join(output_dir, os.path.basename(file_path))
    output_data.astype(np.float32).tofile(output_path)

def main():
    # =========================================
    # 配置参数
    # =========================================
    # 基本配置
    config = {
        'input_dir': 'trainsets/spectH_XCAT_fwhm7_poisson',  # 输入数据目录
        'output_dir': 'trainsets/spectL_XCAT_fwhm7_8x',  # 输出数据目录
        
        # 图像类型和处理方式
        'image_type': 'real',  # 'ideal' 或 'real'
        
        # 理想图像处理选项
        'apply_gaussian': True,  # 是否应用高斯模糊
        'fwhm': 7.0,  # 高斯模糊的半高宽度
        'apply_poisson': True,  # 是否添加泊松噪声
        'apply_speed': False,  # 是否应用降采样
        'speed': 8,  # 降采样系数
        
        # 实际高计数图像处理选项
        'apply_binomial': True,  # 是否应用二项重采样
    }

    # 验证参数组合
    if config['image_type'] == 'ideal':
        if config['apply_binomial']:
            print("警告：二项重采样仅用于real类型图像，将被忽略")
    elif config['image_type'] == 'real':
        if config['apply_gaussian'] or config['apply_poisson'] or config['apply_speed']:
            print("警告：高斯模糊、泊松噪声和降采样仅用于ideal类型图像，将被忽略")

    # 创建处理配置
    process_config = {
        'image_type': config['image_type'],
        'apply_gaussian': config['apply_gaussian'],
        'fwhm': config['fwhm'],
        'apply_poisson': config['apply_poisson'],
        'apply_speed': config['apply_speed'],
        'speed': config['speed'],
        'apply_binomial': config['apply_binomial']
    }

    # 创建输出目录
    os.makedirs(config['output_dir'], exist_ok=True)

    # 获取所有.dat文件
    file_paths = util.get_image_paths(config['input_dir'])
    if not file_paths:
        print(f"错误：在 '{config['input_dir']}' 中未找到任何 .dat 文件。请检查路径。")
        return

    # 处理每个文件
    for file_path in tqdm(file_paths, desc='处理数据'):
        process_single_file(file_path, config['output_dir'], process_config)

    print(f'处理完成！')
    print(f'处理后的数据已保存到: {config["output_dir"]}')

if __name__ == '__main__':
    main() 