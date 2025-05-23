import os
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def visualize_dat_file(file_path, save_dir, subdir):
    """
    可视化单个dat文件的前位和后位投影
    
    参数:
        file_path: dat文件路径
        save_dir: 保存图像的目录
        subdir: 文件夹名称
    """
    # 读取数据
    data = np.fromfile(file_path, dtype=np.float32)
    data = data.reshape(2, 1024, 256)
    
    # 计算总计数
    anterior_counts = np.sum(data[0])
    posterior_counts = np.sum(data[1])
    
    # 创建图像
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # 显示前位投影
    im1 = ax1.imshow(data[0], cmap='gray')
    plt.colorbar(im1, ax=ax1)
    ax1.set_title(f'前位投影\n总计数: {anterior_counts:.0f}')
    
    # 显示后位投影
    im2 = ax2.imshow(data[1], cmap='gray')
    plt.colorbar(im2, ax=ax2)
    ax2.set_title(f'后位投影\n总计数: {posterior_counts:.0f}')
    
    # 设置总标题
    plt.suptitle(f'数据集: {subdir}', fontsize=16)
    plt.tight_layout()
    
    # 保存图像
    save_path = os.path.join(save_dir, f'{subdir}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # 设置基础目录
    base_dir = 'SPECTdatasets'
    save_dir = 'visualization_results'
    os.makedirs(save_dir, exist_ok=True)
    
    # 获取所有子目录
    subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    subdirs.sort()  # 按字母顺序排序
    
    for subdir in subdirs:
        print(f"\n处理目录: {subdir}")
        input_dir = os.path.join(base_dir, subdir)
        
        # 获取第一个dat文件
        dat_files = [f for f in os.listdir(input_dir) if f.endswith('.dat')]
        if not dat_files:
            print(f"警告：在 '{input_dir}' 中未找到任何 .dat 文件")
            continue
            
        first_file = dat_files[0]
        print(f"正在显示文件: {first_file}")
        
        # 可视化文件
        file_path = os.path.join(input_dir, first_file)
        visualize_dat_file(file_path, save_dir, subdir)
        
    print(f"\n所有图像已保存到: {save_dir}")
    print("生成的文件：")
    for subdir in subdirs:
        print(f"- {subdir}.png")

if __name__ == '__main__':
    main()
