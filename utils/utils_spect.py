import numpy as np

def normalize_spect(img, method='log'):
    """SPECT图像归一化
    Args:
        img (np.ndarray): 输入图像
        method (str): 归一化方法，可选 'log' 或 'anscombe'
    Returns:
        np.ndarray: 归一化后的图像
    """
    if method == 'log':
        return np.log(img + 1.0) / 5  # 加1避免log(0)
    elif method == 'anscombe':
        return np.sqrt(img + 3/8) / 10
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def denormalize_spect(img, method='log'):
    """SPECT图像反归一化
    Args:
        img (np.ndarray): 归一化后的图像
        method (str): 归一化方法，可选 'log' 或 'anscombe'
    Returns:
        np.ndarray: 原始尺度的图像
    """
    if method == 'log':
        return np.exp(img * 5) - 1.0
    elif method == 'anscombe':
        return (10 * img) ** 2 - 3/8
    else:
        raise ValueError(f"Unknown normalization method: {method}")
