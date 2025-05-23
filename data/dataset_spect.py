import os.path
import sys
import random
import numpy as np
import torch
import torch.utils.data as data
import utils.utils_image as util
from utils import utils_spect


class DatasetSPECT(data.Dataset):
    """
    # -----------------------------------------
    # 获取SPECT数据的L/H图像对
    # 从实际的L和H数据中读取
    # -----------------------------------------
    """

    def __init__(self, opt):
        super(DatasetSPECT, self).__init__()
        self.opt = opt
        # 设置输入通道数，默认为2（前位和后位）
        self.n_channels = opt['n_channels'] if opt['n_channels'] else 2
        # 设置图像块大小，默认为64x64
        self.patch_size = opt.get('H_size', 64)
        # 设置归一化类型，默认为'log'
        self.normalization_type = opt['normalization']['type'] if opt['normalization'] and isinstance(opt['normalization'], dict) and 'type' in opt['normalization'] else 'log'
        # 设置归一化最大像素值，默认为150
        self.normalization_max_pixel = opt['normalization']['max_pixel'] if opt['normalization'] and isinstance(opt['normalization'], dict) and 'max_pixel' in opt['normalization'] else 150
        
        # 设置文件索引范围
        self.start_index = opt.get('start_index', 0)  # 默认从0开始
        self.end_index = opt.get('end_index', None)   # 默认到最后一个
        
        # ------------------------------------
        # 获取L/H数据路径
        # ------------------------------------
        self.paths_H = util.get_image_paths(opt['dataroot_H'])
        self.paths_L = util.get_image_paths(opt['dataroot_L'])

        assert self.paths_H, 'Error: H path is empty.'
        assert self.paths_L, 'Error: L path is empty.'
        if self.paths_L and self.paths_H:
            assert len(self.paths_L) == len(self.paths_H), 'L/H mismatch - {}, {}.'.format(len(self.paths_L), len(self.paths_H))
            
        # 根据索引范围截取文件列表
        if self.end_index is None:
            self.end_index = len(self.paths_H)
        else:
            self.end_index = min(self.end_index, len(self.paths_H))
            
        self.paths_H = self.paths_H[self.start_index:self.end_index]
        self.paths_L = self.paths_L[self.start_index:self.end_index]
        
        print(f'Using files from index {self.start_index} to {self.end_index-1} (total: {len(self.paths_H)} files)')

    def __getitem__(self, index):
        # ------------------------------------
        # 获取L/H图像
        # ------------------------------------
        H_path = self.paths_H[index]
        L_path = self.paths_L[index]
        
        # 读取.dat文件
        H_data = np.fromfile(H_path, dtype=np.float32)
        L_data = np.fromfile(L_path, dtype=np.float32)
        
        # 重塑数据
        H_data = H_data.reshape(2, 1024, 256)
        L_data = L_data.reshape(2, 1024, 256)
        
        # 前位和后位处理
        H_anterior = H_data[0]
        H_posterior = H_data[1]
        H_posterior_flipped = np.fliplr(H_posterior)
        img_H = np.stack([H_anterior, H_posterior_flipped], axis=2)  # (1024, 256, 2)
        
        L_anterior = L_data[0]
        L_posterior = L_data[1]
        L_posterior_flipped = np.fliplr(L_posterior)
        img_L = np.stack([L_anterior, L_posterior_flipped], axis=2)  # (1024, 256, 2)

        if self.opt['phase'] == 'train':
            """
            # --------------------------------
            # 获取L/H图像块对
            # --------------------------------
            """
            H, W, _ = img_H.shape

            # --------------------------------
            # 随机裁剪图像块
            # --------------------------------
            rnd_h = random.randint(0, max(0, H - self.patch_size))
            rnd_w = random.randint(0, max(0, W - self.patch_size))
            img_H = img_H[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
            img_L = img_L[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]

            # --------------------------------
            # 数据增强 - 翻转，旋转
            # --------------------------------
            mode = random.randint(0, 7)
            img_H = util.augment_img(img_H, mode=mode)
            img_L = util.augment_img(img_L, mode=mode)
            
            # --------------------------------
            # 使用工具函数进行归一化
            # --------------------------------
            img_L = utils_spect.normalize_spect(img_L, self.normalization_type, self.normalization_max_pixel)
            img_H = utils_spect.normalize_spect(img_H, self.normalization_type, self.normalization_max_pixel)
            
            # --------------------------------
            # HWC to CHW, numpy to tensor
            # --------------------------------
            img_H = torch.from_numpy(img_H.transpose((2, 0, 1))).float()
            img_L = torch.from_numpy(img_L.transpose((2, 0, 1))).float()
            
            return {'L': img_L, 'H': img_H}
        else:
            # --------------------------------
            # 使用工具函数进行归一化
            # --------------------------------
            img_L = utils_spect.normalize_spect(img_L, self.normalization_type, self.normalization_max_pixel)
            img_H = utils_spect.normalize_spect(img_H, self.normalization_type, self.normalization_max_pixel)
            
            # --------------------------------
            # HWC to CHW, numpy to tensor
            # --------------------------------
            img_H = torch.from_numpy(img_H.transpose((2, 0, 1))).float()
            img_L = torch.from_numpy(img_L.transpose((2, 0, 1))).float()
            
            return {'L': img_L, 'H': img_H, 'L_path': L_path, 'H_path': H_path}

    def __len__(self):
        return len(self.paths_H)
