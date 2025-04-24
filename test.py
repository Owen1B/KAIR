import torch
from models.network_rrdbnet import RRDBNet

def test_rrdbnet_sizes():
    # 创建模型实例
    model = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, sf=1)
    model.eval()
    
    # 测试不同的输入尺寸
    test_sizes = [
        (1, 3, 64, 64),    # 最小尺寸
        (1, 3, 128, 128),  # 中等尺寸
        (1, 3, 256, 256),  # 较大尺寸
        (1, 3, 512, 512),  # 大尺寸
    ]
    
    print("RRDBNet 输入输出尺寸测试:")
    print("-" * 50)
    print(f"上采样倍数 (sf): {model.sf}")
    
    for size in test_sizes:
        # 创建随机输入
        x = torch.randn(size)
        
        # 前向传播
        with torch.no_grad():
            y = model(x)
        
        # 打印结果
        print(f"\n输入尺寸: {size}")
        print(f"输出尺寸: {y.shape}")
        print(f"实际放大倍数: {y.shape[2]/size[2]:.1f}x{y.shape[3]/size[3]:.1f}")

if __name__ == "__main__":
    test_rrdbnet_sizes()