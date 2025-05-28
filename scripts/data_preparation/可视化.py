import os
import numpy as np
import matplotlib.pyplot as plt
import argparse

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# Anscombe变换函数不再需要，可以删除或注释掉
# def anscombe_transform(x):
#     """Anscombe变换"""
#     return 2 * np.sqrt(x + 3/8)

# def inverse_anscombe_transform(x):
#     """Anscombe反变换"""
#     return (x/2)**2 - 3/8

def visualize_dat_file(file_path, save_dir, vmax=100):
    """
    可视化单个dat文件的前位和后位投影
    
    参数:
        file_path: dat文件路径
        save_dir: 保存图像的目录
        vmax: 显示的最大值
    """
    if not os.path.exists(file_path):
        print(f"错误：文件不存在: {file_path}")
        return
    if not file_path.endswith('.dat'):
        print(f"错误：文件 {file_path} 不是 .dat 文件。")
        return

    subdir = os.path.basename(os.path.dirname(file_path))
    base_filename = os.path.basename(file_path)

    data = np.fromfile(file_path, dtype=np.float32)
    data = data.reshape(2, 1024, 256)
    
    anterior_counts = np.sum(data[0])
    posterior_counts = np.sum(data[1])
    
    # 创建图像，布局改为1x2
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 10))
    
    # 显示原始前位投影
    im1 = ax1.imshow(data[0], cmap='gray', vmax=vmax)
    plt.colorbar(im1, ax=ax1)
    ax1.set_title(f'原始前位投影\n总计数: {anterior_counts:.0f}')
    
    # 显示原始后位投影
    im2 = ax2.imshow(data[1], cmap='gray', vmax=vmax)
    plt.colorbar(im2, ax=ax2)
    ax2.set_title(f'原始后位投影\n总计数: {posterior_counts:.0f}')
    
    plt.suptitle(f'数据集: {subdir}\n文件: {base_filename}', fontsize=16)

    
    safe_subdir_filename = f"{subdir.replace(os.sep, '_')}_{base_filename.replace('.dat', '')}.png"
    save_path = os.path.join(save_dir, safe_subdir_filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"图像已保存到: {save_path}")

def main():
    image_path = "SPECTdatasets/spectH_XCAT_ideal_1x/B001.dat"
    save_dir = "visualization_results_cli"  

    os.makedirs(save_dir, exist_ok=True)

    visualize_dat_file(image_path, save_dir, vmax=80)

if __name__ == '__main__':
    main()
