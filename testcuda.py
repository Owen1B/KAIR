import torch

print('CUDA是否可用:', torch.cuda.is_available())

if torch.cuda.is_available():
    print('GPU数量:', torch.cuda.device_count())
    
    # 测试所有GPU的可用性
    available_gpus = []
    for i in range(6):  # 测试6个GPU
        try:
            torch.cuda.set_device(i)
            # 测试GPU是否可用
            test_tensor = torch.tensor([1.0]).cuda(i)
            del test_tensor
            torch.cuda.empty_cache()
            available_gpus.append(i)
            print(f'GPU {i} 可用: {torch.cuda.get_device_name(i)}')
        except RuntimeError as e:
            print(f'GPU {i} 不可用: {e}')
            continue
    
    if available_gpus:
        print(f'\n找到 {len(available_gpus)} 个可用GPU: {available_gpus}')
        # 测试每个可用GPU
        for gpu_id in available_gpus:
            device = torch.device(f'cuda:{gpu_id}')
            print(f'\n测试 GPU {gpu_id}:')
            x = torch.rand(3, 3).to(device)
            print('张量已移动到GPU:', x)
            print('张量所在设备:', x.device)
    else:
        print('所有GPU都被占用，使用CPU')
        device = torch.device('cpu')
        x = torch.rand(3, 3).to(device)
        print('张量在CPU上:', x)
else:
    print('没有可用的GPU，请检查CUDA安装')