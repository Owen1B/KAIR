import os
import numpy as np
from tqdm import tqdm
import sys

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

import utils.utils_image as util

def process_single_file(file_path, output_dir, speed=4):
    """
    处理单个SPECT数据文件
    
    参数:
        file_path: 输入文件路径
        output_dir: 输出目录
        speed: 快速扫描倍数
    """
    # 读取.dat文件
    data = np.fromfile(file_path, dtype=np.float32)
    
    # 重塑数据为正确的形状 (2, 1024, 256)
    data = data.reshape(2, 1024, 256)
    
    # 前位和后位
    anterior = data[0]
    posterior = data[1]
    
    # 使用二项重采样模拟低剂量SPECT
    anterior_low = np.zeros_like(anterior)
    posterior_low = np.zeros_like(posterior)
    
    # 将每个像素值视为泊松分布的均值，然后进行二项重采样
    anterior_counts = anterior.astype(np.int32)
    posterior_counts = posterior.astype(np.int32)
    
    # 使用二项分布模拟低剂量采集，p=1/speed表示保留的计数比例
    anterior_low = np.random.binomial(n=anterior_counts, p=1.0/speed) * speed
    posterior_low = np.random.binomial(n=posterior_counts, p=1.0/speed) * speed
    
    # 将处理后的数据重新组合成原始格式 (2, 1024, 256)
    output_data = np.stack([anterior_low, posterior_low], axis=0)
    
    # 保存退化后的数据，保持原始格式
    output_path = os.path.join(output_dir, os.path.basename(file_path))
    output_data.astype(np.float32).tofile(output_path)

def main():
    speed = 4  # 快速扫描倍数
    # 设置输入输出目录
    input_dir = 'trainsets/spectH'  # 高质量数据目录
    output_dir = f'trainsets/spectL_{speed}x'  # 低质量数据目录

    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有.dat文件
    file_paths = util.get_image_paths(input_dir)
    
    # 处理每个文件
    for file_path in tqdm(file_paths, desc='处理数据'):
        process_single_file(file_path, output_dir, speed)
    
    print(f'处理完成！数据已保存到: {output_dir}')

if __name__ == '__main__':
    main() 