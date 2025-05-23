import numpy as np

def normalize_spect(img, method='log', max_pixel=150):
    """SPECT图像归一化
    Args:
        img (np.ndarray): 输入图像
        method (str): 归一化方法，可选 'log'、'anscombe' 或 'linear'
    Returns:
        np.ndarray: 归一化后的图像
    """
    if method == 'log':
        return np.log(img + 1.0) / np.log(max_pixel + 1.0)  # 加1避免log(0)
    elif method == 'anscombe':
        return np.sqrt(img + 3/8) / np.sqrt(max_pixel + 3/8)
    elif method == 'linear':
        return img / max_pixel
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def denormalize_spect(img, method='log', max_pixel=150):
    """SPECT图像反归一化
    Args:
        img (np.ndarray): 归一化后的图像
        method (str): 归一化方法，可选 'log'、'anscombe' 或 'linear'
    Returns:
        np.ndarray: 原始尺度的图像
    """
    if method == 'log':
        return np.exp(img * np.log(max_pixel + 1.0)) - 1.0
    elif method == 'anscombe':
        return (np.sqrt(max_pixel + 3/8) * img) ** 2 - 3/8
    elif method == 'linear':
        return img * max_pixel
    else:
        raise ValueError(f"Unknown normalization method: {method}")
