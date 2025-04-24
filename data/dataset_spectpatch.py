import os.path
import random
import numpy as np
import torch.utils.data as data
import utils.utils_image as util
import utils.utils_spect as utils_spect
import torch


class DatasetSPECTPatch(data.Dataset):
    '''
    # -----------------------------------------
    # 获取SPECT数据的L/H图像对
    # 需要同时提供L和H数据
    # 训练集使用patch，测试集使用整张图像
    # -----------------------------------------
    '''

    def __init__(self, opt):
        super(DatasetSPECTPatch, self).__init__()
        self.opt = opt
        self.n_channels = opt['n_channels'] if opt['n_channels'] else 2  # SPECT数据为2通道
        self.patch_size = opt['H_size'] if opt['H_size'] else 64
        self.normalization = opt['normalization'] if opt['normalization'] else 'log'  # 归一化方式

        self.num_patches_per_image = opt['num_patches_per_image'] if opt['num_patches_per_image'] else 40
        self.num_sampled = opt['num_sampled'] if opt['num_sampled'] else 3000

        # -------------------
        # 获取L/H数据路径
        # -------------------
        self.paths_H = util.get_image_paths(opt['dataroot_H'])
        self.paths_L = util.get_image_paths(opt['dataroot_L'])

        assert self.paths_H, 'Error: H path is empty.'
        assert self.paths_L, 'Error: L path is empty.'
        if self.paths_L and self.paths_H:
            assert len(self.paths_L) == len(self.paths_H), 'H and L datasets have different number of images - {}, {}.'.format(len(self.paths_L), len(self.paths_H))

        # ------------------------------------
        # 采样的图像数量
        # ------------------------------------
        self.num_sampled = min(self.num_sampled, len(self.paths_H))

        # ------------------------------------
        # 预分配内存
        # ------------------------------------
        if self.opt['phase'] == 'train':
            self.total_patches = self.num_sampled * self.num_patches_per_image
            self.H_data = np.zeros([self.total_patches, self.patch_size, self.patch_size, self.n_channels], dtype=np.float32)
            self.L_data = np.zeros([self.total_patches, self.patch_size, self.patch_size, self.n_channels], dtype=np.float32)

            # ------------------------------------
            # 更新数据
            # ------------------------------------
            self.update_data()
        else:
            self.total_patches = len(self.paths_H)

    def update_data(self):
        """
        # ------------------------------------
        # 更新所有L/H patches
        # ------------------------------------
        """
        self.index_sampled = random.sample(range(0, len(self.paths_H), 1), self.num_sampled)
        n_count = 0

        for i in range(len(self.index_sampled)):
            L_patches, H_patches = self.get_patches(self.index_sampled[i])
            for (L_patch, H_patch) in zip(L_patches, H_patches):
                self.L_data[n_count,:,:,:] = L_patch
                self.H_data[n_count,:,:,:] = H_patch
                n_count += 1

        print('训练数据已更新！总patch数量: %d' % (len(self.H_data)))

    def get_patches(self, index):
        """
        # ------------------------------------
        # 从L/H图像中获取patches
        # ------------------------------------
        """
        L_path = self.paths_L[index]
        H_path = self.paths_H[index]
        
        # 读取.dat文件
        L_data = np.fromfile(L_path, dtype=np.float32)
        H_data = np.fromfile(H_path, dtype=np.float32)
        
        # 重塑数据为正确的形状 (2, 1024, 256)
        L_data = L_data.reshape(2, 1024, 256)
        H_data = H_data.reshape(2, 1024, 256)
        
        # 前位和后位
        L_anterior = L_data[0]
        L_posterior = L_data[1]
        H_anterior = H_data[0]
        H_posterior = H_data[1]
        
        # 将后位左右翻转，并创建新的数组
        L_posterior_flipped = np.fliplr(L_posterior).copy()
        H_posterior_flipped = np.fliplr(H_posterior).copy()
        
        # 将前位和翻转后的后位组合成图像
        img_L = np.stack([L_anterior, L_posterior_flipped], axis=2)
        img_H = np.stack([H_anterior, H_posterior_flipped], axis=2)

        H, W = img_H.shape[:2]
        L_patches, H_patches = [], []

        num = self.num_patches_per_image
        for _ in range(num):
            rnd_h = random.randint(0, max(0, H - self.patch_size))
            rnd_w = random.randint(0, max(0, W - self.patch_size))
            L_patch = img_L[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
            H_patch = img_H[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
            
            # 使用工具函数进行归一化
            L_patch = utils_spect.normalize_spect(L_patch, self.normalization)
            H_patch = utils_spect.normalize_spect(H_patch, self.normalization)
                
            L_patches.append(L_patch)
            H_patches.append(H_patch)

        return L_patches, H_patches

    def __getitem__(self, index):
        if self.opt['phase'] == 'train':
            # 训练阶段使用预先提取的patches
            patch_L, patch_H = self.L_data[index], self.H_data[index]

            # --------------------------------
            # 数据增强 - 翻转和/或旋转
            # --------------------------------
            mode = random.randint(0, 7)
            patch_L = util.augment_img(patch_L, mode=mode).copy()
            patch_H = util.augment_img(patch_H, mode=mode).copy()

            # HWC to CHW, numpy to tensor
            patch_L = torch.from_numpy(patch_L.transpose((2, 0, 1))).float()
            patch_H = torch.from_numpy(patch_H.transpose((2, 0, 1))).float()
            
            return {'L': patch_L, 'H': patch_H}
        else:
            # 测试阶段使用整张图像
            L_path = self.paths_L[index]
            H_path = self.paths_H[index]
            
            # 读取.dat文件
            L_data = np.fromfile(L_path, dtype=np.float32)
            H_data = np.fromfile(H_path, dtype=np.float32)
            
            # 重塑数据为正确的形状 (2, 1024, 256)
            L_data = L_data.reshape(2, 1024, 256)
            H_data = H_data.reshape(2, 1024, 256)
            
            # 前位和后位
            L_anterior = L_data[0]
            L_posterior = L_data[1]
            H_anterior = H_data[0]
            H_posterior = H_data[1]
            
            # 将后位左右翻转，并创建新的数组
            L_posterior_flipped = np.fliplr(L_posterior).copy()
            H_posterior_flipped = np.fliplr(H_posterior).copy()
            
            # 将前位和翻转后的后位组合成图像
            img_L = np.stack([L_anterior, L_posterior_flipped], axis=2)
            img_H = np.stack([H_anterior, H_posterior_flipped], axis=2)
            
            # 使用工具函数进行归一化
            img_L = utils_spect.normalize_spect(img_L, self.normalization)
            img_H = utils_spect.normalize_spect(img_H, self.normalization)
            
            # HWC to CHW, numpy to tensor
            img_L = torch.from_numpy(img_L.transpose((2, 0, 1))).float()
            img_H = torch.from_numpy(img_H.transpose((2, 0, 1))).float()
            
            return {'L': img_L, 'H': img_H, 'L_path': L_path, 'H_path': H_path}

    def __len__(self):
        return self.total_patches
