import os.path
import sys
import random
import numpy as np
import torch
import torch.utils.data as data
import utils.utils_image as util
from utils import utils_spect


class DatasetSPECTIdeal(data.Dataset):
    """
    # -----------------------------------------
    # 获取SPECT数据的L/H图像对
    # H: 从 dataroot_H 读取 (已高斯模糊, 无泊松噪声源文件)。可选择性地在输出H前添加泊松噪声。
    # L:
    #   训练阶段: 由 H 图像的原始投影（无额外噪声）降质得到
    #           降质方式可选:
    #           1. 除以speed_factor + 添加泊松噪声 (默认)
    #           2. 二项重采样 (当use_binomial_resampling=True时)
    #   测试/验证阶段: 直接从 dataroot_L 读取
    # 训练时从每个图像动态提取一个随机patch，测试/验证集使用整张图像
    # 使用 dataroot_H (和测试/验证时的 dataroot_L) 中的所有图像
    # -----------------------------------------
    """

    def __init__(self, opt):
        super(DatasetSPECTIdeal, self).__init__()
        self.opt = opt
        # 设置输入通道数，默认为2（前位和后位）
        self.n_channels = opt['n_channels'] if opt['n_channels'] else 2
        # 设置图像块大小，默认为64x64
        self.patch_size = opt.get('H_size', 64)
        # 设置归一化类型，默认为'log'
        self.normalization_type = opt['normalization']['type'] if opt['normalization'] and isinstance(opt['normalization'], dict) and 'type' in opt['normalization'] else 'log'
        # 设置归一化最大像素值，默认为150
        self.normalization_max_pixel = opt['normalization']['max_pixel'] if opt['normalization'] and isinstance(opt['normalization'], dict) and 'max_pixel' in opt['normalization'] else 150
        
        # 设置降质因子，用于训练阶段生成低计数图像
        self.speed_factor = opt.get('speed', 8)
        # 新增选项：H是否为理想图
        self.is_ideal_H = opt.get('is_ideal_H', True)
        # 新增选项：是否使用二项重采样生成L图像
        self.use_binomial_resampling = opt.get('use_binomial_resampling', False)

        # 获取所有高计数图像路径
        self.paths_H_all = util.get_image_paths(opt['dataroot_H'])
        assert self.paths_H_all, '错误: dataroot_H为空'
        self.current_paths_H = list(self.paths_H_all)

        self.has_L = False
        if self.opt['phase'] == 'train':
            # 训练阶段：如果有L路径也加载
            if 'dataroot_L' in opt and opt['dataroot_L']:
                self.paths_L_all = util.get_image_paths(opt['dataroot_L'])
                if self.paths_L_all:
                    self.current_paths_L = list(self.paths_L_all)
                    assert len(self.current_paths_H) == len(self.current_paths_L), \
                        f'训练阶段H和L数据集的图像数量不同 - {len(self.current_paths_H)} vs {len(self.current_paths_L)}'
                    self.has_L = True
        elif self.opt['phase'] in ['test', 'val']:
            assert 'dataroot_L' in opt, f'错误: {self.opt["phase"]}阶段需要dataroot_L'
            self.paths_L_all = util.get_image_paths(opt['dataroot_L'])
            assert self.paths_L_all, f'错误: {self.opt["phase"]}阶段的dataroot_L为空'
            self.current_paths_L = list(self.paths_L_all)
            assert len(self.current_paths_H) == len(self.current_paths_L), \
                f'{self.opt["phase"]}阶段H和L数据集的图像数量不同 - {len(self.current_paths_H)} vs {len(self.current_paths_L)}'
            self.has_L = True
        else:
            raise ValueError(f"不支持的阶段: {self.opt['phase']}")
        
        self.total_items = len(self.current_paths_H)

    def _load_projections_and_stack(self, image_path):
        """加载.dat文件，重塑形状，并堆叠前位/后位投影"""
        # 读取原始数据
        data_raw = np.fromfile(image_path, dtype=np.float32)
        # 重塑为2x1024x256的形状（2个投影，每个1024x256）
        data_reshaped = data_raw.reshape(2, 1024, 256)
        # 获取前位投影
        anterior_proj = data_reshaped[0]
        # 获取后位投影并水平翻转
        posterior_proj_flipped = np.fliplr(data_reshaped[1].copy())
        # 将前位和后位投影堆叠在一起，确保维度顺序一致
        img_stacked = np.stack([anterior_proj, posterior_proj_flipped], axis=2)  # 结果形状为 (1024, 256, 2)
        return img_stacked

    def __getitem__(self, index):
        if self.opt['phase'] == 'train':
            if self.has_L:
                # 训练阶段直接读取L和H
                H_path = self.current_paths_H[index]
                L_path = self.current_paths_L[index]
                img_H = self._load_projections_and_stack(H_path)
                img_L = self._load_projections_and_stack(L_path)
                img_L = img_L * self.speed_factor
                img_L = utils_spect.normalize_spect(img_L, self.normalization_type, self.normalization_max_pixel)
                img_H = utils_spect.normalize_spect(img_H, self.normalization_type, self.normalization_max_pixel)
                mode = random.randint(0, 7)
                img_L = util.augment_img(img_L, mode=mode).copy()
                img_H = util.augment_img(img_H, mode=mode).copy()
                img_L = torch.from_numpy(img_L.transpose((2, 0, 1))).float()
                img_H = torch.from_numpy(img_H.transpose((2, 0, 1))).float()
                return {'L': img_L, 'H': img_H}
            # 否则走降质生成L的流程
            actual_image_path_H = self.current_paths_H[index]
            # 读取原始数据
            h_data_raw = np.fromfile(actual_image_path_H, dtype=np.float32)
            h_data_reshaped = h_data_raw.reshape(2, 1024, 256)
            h_anterior_raw_full = h_data_reshaped[0]
            h_posterior_raw_full = h_data_reshaped[1]
            
            # 检查图像大小是否满足patch要求
            H_proj_shape, W_proj_shape = h_anterior_raw_full.shape
            if H_proj_shape < self.patch_size or W_proj_shape < self.patch_size:
                 raise ValueError(f"图像投影 {actual_image_path_H} (形状 {H_proj_shape}x{W_proj_shape}) 小于patch大小 {self.patch_size}")
            
            # 随机选择patch的起始位置
            rnd_h = random.randint(0, max(0, H_proj_shape - self.patch_size))
            rnd_w = random.randint(0, max(0, W_proj_shape - self.patch_size))
            
            # 提取patch
            h_anterior_patch_raw = h_anterior_raw_full[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size]
            h_posterior_patch_raw = h_posterior_raw_full[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size]
            
            # 生成H和L图像
            if self.is_ideal_H:
                # 如果是理想图：
                # 1. 先对原始H添加泊松噪声得到现在H
                h_anterior_noisy = np.random.poisson(np.maximum(0, h_anterior_patch_raw)).astype(np.float32)
                h_posterior_noisy = np.random.poisson(np.maximum(0, h_posterior_patch_raw)).astype(np.float32)
                
                # 2. 根据配置生成L图像
                if self.use_binomial_resampling:
                    # 使用二项重采样生成L图像（基于带噪声的H）
                    n = h_anterior_noisy.astype(np.int32)
                    p = 1.0 / self.speed_factor
                    l_anterior_noisy = np.random.binomial(n, p).astype(np.float32)
                    l_posterior_noisy = np.random.binomial(h_posterior_noisy.astype(np.int32), p).astype(np.float32)
                else:
                    # 使用原始H/倍率+泊松噪声
                    l_anterior_scaled = h_anterior_patch_raw / self.speed_factor
                    l_posterior_scaled = h_posterior_patch_raw / self.speed_factor
                    l_anterior_noisy = np.random.poisson(np.maximum(0, l_anterior_scaled)).astype(np.float32)
                    l_posterior_noisy = np.random.poisson(np.maximum(0, l_posterior_scaled)).astype(np.float32)
                
                # 3. 使用带噪声的H作为最终的H图像
                patch_H = np.stack([h_anterior_noisy, np.fliplr(h_posterior_noisy.copy())], axis=2)
            else:
                # 如果是真实图：
                # 1. 直接使用原始H作为最终的H图像
                patch_H = np.stack([h_anterior_patch_raw, np.fliplr(h_posterior_patch_raw.copy())], axis=2)
                
                # 2. 对原始H进行二项采样得到L
                n = h_anterior_patch_raw.astype(np.int32)
                p = 1.0 / self.speed_factor
                l_anterior_noisy = np.random.binomial(n, p).astype(np.float32)
                l_posterior_noisy = np.random.binomial(h_posterior_patch_raw.astype(np.int32), p).astype(np.float32)
            
            # 堆叠L图像
            patch_L = np.stack([l_anterior_noisy, np.fliplr(l_posterior_noisy.copy())], axis=2)
            
            # 将L恢复到原始H计数水平
            patch_L = patch_L * self.speed_factor
            
            # 归一化处理
            patch_L = utils_spect.normalize_spect(patch_L, self.normalization_type, self.normalization_max_pixel)
            patch_H = utils_spect.normalize_spect(patch_H, self.normalization_type, self.normalization_max_pixel)
            
            # 数据增强
            mode = random.randint(0, 7)
            patch_L = util.augment_img(patch_L, mode=mode).copy()
            patch_H = util.augment_img(patch_H, mode=mode).copy()
            
            # 转换为PyTorch张量
            patch_L = torch.from_numpy(patch_L.transpose((2, 0, 1))).float()
            patch_H = torch.from_numpy(patch_H.transpose((2, 0, 1))).float()
            
            return {'L': patch_L, 'H': patch_H}
        
        elif self.opt['phase'] in ['test', 'val']:
            # 测试/验证阶段：直接加载H和L图像
            H_path = self.current_paths_H[index]
            L_path = self.current_paths_L[index]

            # 加载并堆叠投影
            img_H = self._load_projections_and_stack(H_path)
            img_L = self._load_projections_and_stack(L_path)
            
            # 将L恢复到原始H计数水平
            img_L = img_L * self.speed_factor
            
            # 归一化处理
            img_L = utils_spect.normalize_spect(img_L, self.normalization_type, self.normalization_max_pixel)
            img_H = utils_spect.normalize_spect(img_H, self.normalization_type, self.normalization_max_pixel)
            
            # 转换为PyTorch张量
            img_L = torch.from_numpy(img_L.transpose((2, 0, 1))).float()
            img_H = torch.from_numpy(img_H.transpose((2, 0, 1))).float()
            
            return {'L': img_L, 'H': img_H, 'L_path': L_path, 'H_path': H_path}

    def __len__(self):
        return self.total_items
