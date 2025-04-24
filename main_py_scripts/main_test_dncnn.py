import os.path
import logging
import argparse

import numpy as np
from datetime import datetime
from collections import OrderedDict
# from scipy.io import loadmat

import torch

from utils import utils_logger
from utils import utils_model
from utils import utils_image as util


'''
Spyder (Python 3.6)
PyTorch 1.1.0
Windows 10 或 Linux

Kai Zhang (cskaizhang@gmail.com)
github: https://github.com/cszn/KAIR
        https://github.com/cszn/DnCNN

@article{zhang2017beyond,
  title={Beyond a gaussian denoiser: Residual learning of deep cnn for image denoising},
  author={Zhang, Kai and Zuo, Wangmeng and Chen, Yunjin and Meng, Deyu and Zhang, Lei},
  journal={IEEE Transactions on Image Processing},
  volume={26},
  number={7},
  pages={3142--3155},
  year={2017},
  publisher={IEEE}
}

% 如果您有任何问题，请随时与我联系。
% Kai Zhang (电子邮件: cskaizhang@gmail.com; github: https://github.com/cszn)

由 Kai Zhang 创建 (2019年12月12日)
'''

"""
# --------------------------------------------
|--model_zoo          # 模型库
   |--dncnn_15        # 模型名称
   |--dncnn_25
   |--dncnn_50
   |--dncnn_gray_blind
   |--dncnn_color_blind
   |--dncnn3
|--testset            # 测试集
   |--set12           # 测试集名称
   |--bsd68
   |--cbsd68
|--results            # 结果
   |--set12_dncnn_15  # 结果名称 = 测试集名称 + '_' + 模型名称
   |--set12_dncnn_25
   |--bsd68_dncnn_15
# --------------------------------------------
"""


def main():

    # ----------------------------------------
    # 准备工作
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='dncnn_15', help='dncnn_15, dncnn_25, dncnn_50, dncnn_gray_blind, dncnn_color_blind, dncnn3')
    parser.add_argument('--testset_name', type=str, default='set12', help='测试集, bsd68 | set12')
    parser.add_argument('--noise_level_img', type=int, default=15, help='噪声级别: 15, 25, 50')
    parser.add_argument('--x8', type=bool, default=False, help='使用x8提升性能')
    parser.add_argument('--show_img', type=bool, default=False, help='显示图像')
    parser.add_argument('--model_pool', type=str, default='model_zoo', help='模型库路径')
    parser.add_argument('--testsets', type=str, default='testsets', help='测试文件夹路径')
    parser.add_argument('--results', type=str, default='results', help='结果路径')
    parser.add_argument('--need_degradation', type=bool, default=True, help='是否添加噪声')
    parser.add_argument('--task_current', type=str, default='dn', help='dn表示去噪, 固定!')
    parser.add_argument('--sf', type=int, default=1, help='去噪时未使用')
    args = parser.parse_args()

    if 'color' in args.model_name:
        n_channels = 3        # 固定, 1表示灰度图像, 3表示彩色图像
    else:
        n_channels = 1        # 固定用于灰度图像
    if args.model_name in ['dncnn_gray_blind', 'dncnn_color_blind', 'dncnn3']:
        nb = 20               # 固定
    else:
        nb = 17               # 固定

    result_name = args.testset_name + '_' + args.model_name     # 固定
    border = args.sf if args.task_current == 'sr' else 0        # 裁剪边界以计算PSNR和SSIM
    model_path = os.path.join(args.model_pool, args.model_name+'.pth')

    # ----------------------------------------
    # L_path, E_path, H_path
    # ----------------------------------------

    L_path = os.path.join(args.testsets, args.testset_name) # L_path, 用于低质量图像
    H_path = L_path                               # H_path, 用于高质量图像
    E_path = os.path.join(args.results, result_name)   # E_path, 用于估计图像
    util.mkdir(E_path)

    if H_path == L_path:
        args.need_degradation = True
    logger_name = result_name
    utils_logger.logger_info(logger_name, log_path=os.path.join(E_path, logger_name+'.log'))
    logger = logging.getLogger(logger_name)

    need_H = True if H_path is not None else False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ----------------------------------------
    # 加载模型
    # ----------------------------------------

    from models.network_dncnn import DnCNN as net
    model = net(in_nc=n_channels, out_nc=n_channels, nc=64, nb=nb, act_mode='R')
    # model = net(in_nc=n_channels, out_nc=n_channels, nc=64, nb=nb, act_mode='BR')  # 如果BN未通过utils_bnorm.merge_bn(model)合并，请使用此选项
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)
    logger.info('模型路径: {:s}'.format(model_path))
    number_parameters = sum(map(lambda x: x.numel(), model.parameters()))
    logger.info('参数数量: {}'.format(number_parameters))

    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []

    logger.info('模型名称:{}, 图像噪声级别:{}'.format(args.model_name, args.noise_level_img))
    logger.info(L_path)
    L_paths = util.get_image_paths(L_path)
    H_paths = util.get_image_paths(H_path) if need_H else None

    for idx, img in enumerate(L_paths):

        # ------------------------------------
        # (1) img_L
        # ------------------------------------

        img_name, ext = os.path.splitext(os.path.basename(img))
        # logger.info('{:->4d}--> {:>10s}'.format(idx+1, img_name+ext))
        img_L = util.imread_uint(img, n_channels=n_channels)
        img_L = util.uint2single(img_L)

        if args.need_degradation:  # 降质过程
            np.random.seed(seed=0)  # 为了可重现性
            img_L += np.random.normal(0, args.noise_level_img/255., img_L.shape)

        util.imshow(util.single2uint(img_L), title='噪声图像，噪声级别 {}'.format(args.noise_level_img)) if args.show_img else None

        img_L = util.single2tensor4(img_L)
        img_L = img_L.to(device)

        # ------------------------------------
        # (2) img_E
        # ------------------------------------

        if not args.x8:
            img_E = model(img_L)
        else:
            img_E = utils_model.test_mode(model, img_L, mode=3)

        img_E = util.tensor2uint(img_E)

        if need_H:

            # --------------------------------
            # (3) img_H
            # --------------------------------

            img_H = util.imread_uint(H_paths[idx], n_channels=n_channels)
            img_H = img_H.squeeze()

            # --------------------------------
            # PSNR和SSIM
            # --------------------------------

            psnr = util.calculate_psnr(img_E, img_H, border=border)
            ssim = util.calculate_ssim(img_E, img_H, border=border)
            test_results['psnr'].append(psnr)
            test_results['ssim'].append(ssim)
            logger.info('{:s} - PSNR: {:.2f} dB; SSIM: {:.4f}.'.format(img_name+ext, psnr, ssim))
            util.imshow(np.concatenate([img_E, img_H], axis=1), title='恢复图像 / 原始图像') if args.show_img else None

        # ------------------------------------
        # 保存结果
        # ------------------------------------

        util.imsave(img_E, os.path.join(E_path, img_name+ext))

    if need_H:
        ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
        ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
        logger.info('平均 PSNR/SSIM(RGB) - {} - PSNR: {:.2f} dB; SSIM: {:.4f}'.format(result_name, ave_psnr, ave_ssim))

if __name__ == '__main__':

    main()
