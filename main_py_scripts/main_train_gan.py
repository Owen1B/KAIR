import os.path
import math
import argparse
import random
import numpy as np
import logging
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch

from utils import utils_logger
from utils import utils_image as util
from utils import utils_option as option
from utils.utils_dist import get_dist_info, init_dist

from data.select_dataset import define_Dataset
from models.select_model import define_Model


'''
# --------------------------------------------
# GAN模型训练代码
# 作者: Kai Zhang (cskaizhang@gmail.com)
# GitHub: https://github.com/cszn/KAIR
# --------------------------------------------
'''


def main(json_path='options/train_msrresnet_psnr.json'):
    '''
    # ----------------------------------------
    # 步骤1: 准备训练参数配置
    # ----------------------------------------
    '''

    # 创建命令行参数解析器
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default=json_path, help='JSON配置文件路径')
    parser.add_argument('--launcher', default='pytorch', help='任务启动器')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--dist', default=False)

    # 解析并加载配置文件
    opt = option.parse(parser.parse_args().opt, is_train=True)
    opt['dist'] = parser.parse_args().dist

    # ----------------------------------------
    # 分布式训练设置
    # ----------------------------------------
    if opt['dist']:
        init_dist('pytorch')
    opt['rank'], opt['world_size'] = get_dist_info()

    # 创建必要的目录
    if opt['rank'] == 0:
        util.mkdirs((path for key, path in opt['path'].items() if 'pretrained' not in key))

    # ----------------------------------------
    # 更新训练配置
    # ----------------------------------------
    # 查找并加载最新的检查点
    init_iter_G, init_path_G = option.find_last_checkpoint(opt['path']['models'], net_type='G')
    init_iter_D, init_path_D = option.find_last_checkpoint(opt['path']['models'], net_type='D')
    init_iter_E, init_path_E = option.find_last_checkpoint(opt['path']['models'], net_type='E')
    opt['path']['pretrained_netG'] = init_path_G
    opt['path']['pretrained_netD'] = init_path_D
    opt['path']['pretrained_netE'] = init_path_E
    init_iter_optimizerG, init_path_optimizerG = option.find_last_checkpoint(opt['path']['models'], net_type='optimizerG')
    init_iter_optimizerD, init_path_optimizerD = option.find_last_checkpoint(opt['path']['models'], net_type='optimizerD')
    opt['path']['pretrained_optimizerG'] = init_path_optimizerG
    opt['path']['pretrained_optimizerD'] = init_path_optimizerD
    current_step = max(init_iter_G, init_iter_D, init_iter_E, init_iter_optimizerG, init_iter_optimizerD)

    border = opt['scale']

    # 保存配置到文件
    if opt['rank'] == 0:
        option.save(opt)

    # 处理缺失的配置项
    opt = option.dict_to_nonedict(opt)

    # ----------------------------------------
    # 配置日志记录器
    # ----------------------------------------
    if opt['rank'] == 0:
        logger_name = 'train'
        utils_logger.logger_info(logger_name, os.path.join(opt['path']['log'], logger_name+'.log'))
        logger = logging.getLogger(logger_name)
        logger.info(option.dict2str(opt))

    # ----------------------------------------
    # 设置随机种子
    # ----------------------------------------
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    print('随机种子: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    '''
    # ----------------------------------------
    # 步骤2: 创建数据加载器
    # ----------------------------------------
    '''

    # 创建训练和测试数据集及数据加载器
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = define_Dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['dataloader_batch_size']))
            if opt['rank'] == 0:
                logger.info('训练图像数量: {:,d}, 迭代次数: {:,d}'.format(len(train_set), train_size))
            if opt['dist']:
                # 分布式训练的数据加载器设置
                train_sampler = DistributedSampler(train_set, shuffle=dataset_opt['dataloader_shuffle'], drop_last=True, seed=seed)
                train_loader = DataLoader(train_set,
                                          batch_size=dataset_opt['dataloader_batch_size']//opt['num_gpu'],
                                          shuffle=False,
                                          num_workers=dataset_opt['dataloader_num_workers']//opt['num_gpu'],
                                          drop_last=True,
                                          pin_memory=True,
                                          sampler=train_sampler)
            else:
                # 单机训练的数据加载器设置
                train_loader = DataLoader(train_set,
                                          batch_size=dataset_opt['dataloader_batch_size'],
                                          shuffle=dataset_opt['dataloader_shuffle'],
                                          num_workers=dataset_opt['dataloader_num_workers'],
                                          drop_last=True,
                                          pin_memory=True)

        elif phase == 'test':
            # 测试数据集设置
            test_set = define_Dataset(dataset_opt)
            test_loader = DataLoader(test_set, batch_size=1,
                                     shuffle=False, num_workers=1,
                                     drop_last=False, pin_memory=True)
        else:
            raise NotImplementedError("阶段 [%s] 未定义" % phase)

    '''
    # ----------------------------------------
    # 步骤3: 初始化模型
    # ----------------------------------------
    '''

    # 创建并初始化模型
    model = define_Model(opt)
    model.init_train()
    if opt['rank'] == 0:
        logger.info(model.info_network())
        logger.info(model.info_params())

    '''
    # ----------------------------------------
    # 步骤4: 主训练循环
    # ----------------------------------------
    '''

    for epoch in range(1000000):  # 持续训练
        if opt['dist']:
            train_sampler.set_epoch(epoch + seed)

        for i, train_data in enumerate(train_loader):
            current_step += 1

            # 1) 更新学习率
            model.update_learning_rate(current_step)

            # 2) 输入训练数据
            model.feed_data(train_data)

            # 3) 优化模型参数
            model.optimize_parameters(current_step)

            # 4) 记录训练信息
            if current_step % opt['train']['checkpoint_print'] == 0 and opt['rank'] == 0:
                logs = model.current_log()  # 获取当前损失等信息
                message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(epoch, current_step, model.current_learning_rate())
                for k, v in logs.items():
                    message += '{:s}: {:.3e} '.format(k, v)
                logger.info(message)

            # 5) 保存模型
            if current_step % opt['train']['checkpoint_save'] == 0 and opt['rank'] == 0:
                logger.info('保存模型中...')
                model.save(current_step)

            # 6) 测试模型性能
            if current_step % opt['train']['checkpoint_test'] == 0 and opt['rank'] == 0:
                avg_psnr = 0.0
                idx = 0

                for test_data in test_loader:
                    idx += 1
                    image_name_ext = os.path.basename(test_data['L_path'][0])
                    img_name, ext = os.path.splitext(image_name_ext)

                    img_dir = os.path.join(opt['path']['images'], img_name)
                    util.mkdir(img_dir)

                    model.feed_data(test_data)
                    model.test()

                    visuals = model.current_visuals()
                    E_img = util.tensor2uint(visuals['E'])  # 估计图像
                    H_img = util.tensor2uint(visuals['H'])  # 真实图像

                    # 保存估计图像
                    save_img_path = os.path.join(img_dir, '{:s}_{:d}.png'.format(img_name, current_step))
                    util.imsave(E_img, save_img_path)

                    # 计算PSNR
                    current_psnr = util.calculate_psnr(E_img, H_img, border=border)
                    logger.info('{:->4d}--> {:>10s} | {:<4.2f}dB'.format(idx, image_name_ext, current_psnr))
                    avg_psnr += current_psnr

                avg_psnr = avg_psnr / idx
                logger.info('<epoch:{:3d}, iter:{:8,d}, 平均PSNR: {:<.2f}dB\n'.format(epoch, current_step, avg_psnr))

if __name__ == '__main__':
    main()
