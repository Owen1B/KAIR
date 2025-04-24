import os.path
import math
import argparse
import time
import random
import numpy as np
from collections import OrderedDict
import logging
import torch
from torch.utils.data import DataLoader


from utils import utils_logger
from utils import utils_image as util
from utils import utils_option as option

from data.select_dataset import define_Dataset
from models.select_model import define_Model


'''
# --------------------------------------------
# DnCNN训练代码
# --------------------------------------------
# 作者: Kai Zhang (cskaizhang@gmail.com)
# GitHub: https://github.com/cszn/KAIR
#         https://github.com/cszn/DnCNN
#
# 参考文献:
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
# --------------------------------------------
# 基于 https://github.com/xinntao/BasicSR
# --------------------------------------------
'''


def main(json_path='options/train_dncnn.json'):

    '''
    # ----------------------------------------
    # 步骤1: 准备训练参数配置
    # ----------------------------------------
    '''

    # 创建命令行参数解析器
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, default=json_path, help='配置JSON文件路径')

    # 解析配置文件
    opt = option.parse(parser.parse_args().opt, is_train=True)
    # 创建必要的目录（排除预训练模型路径）
    util.mkdirs((path for key, path in opt['path'].items() if 'pretrained' not in key))

    # ----------------------------------------
    # 更新配置参数
    # ----------------------------------------
    # -->-->-->-->-->-->-->-->-->-->-->-->-->-
    # 查找最后一个检查点，用于继续训练
    init_iter, init_path_G = option.find_last_checkpoint(opt['path']['models'], net_type='G')
    opt['path']['pretrained_netG'] = init_path_G
    current_step = init_iter

    border = 0  # 计算PSNR时的边界裁剪大小
    # --<--<--<--<--<--<--<--<--<--<--<--<--<-

    # ----------------------------------------
    # 保存配置到'../option.json'文件
    # ----------------------------------------
    option.save(opt)

    # ----------------------------------------
    # 对缺失的键返回None
    # ----------------------------------------
    opt = option.dict_to_nonedict(opt)

    # ----------------------------------------
    # 配置日志记录器
    # ----------------------------------------
    logger_name = 'train'
    utils_logger.logger_info(logger_name, os.path.join(opt['path']['log'], logger_name+'.log'))
    logger = logging.getLogger(logger_name)
    logger.info(option.dict2str(opt))  # 将配置信息写入日志

    # ----------------------------------------
    # 设置随机种子，确保实验可重复性
    # ----------------------------------------
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    logger.info('随机种子: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 为所有GPU设置种子

    '''
    # ----------------------------------------
    # 步骤2: 创建数据加载器
    # ----------------------------------------
    '''

    # ----------------------------------------
    # 1) 创建数据集
    # 2) 为训练和测试创建数据加载器
    # ----------------------------------------
    dataset_type = opt['datasets']['train']['dataset_type']
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            # 创建训练集
            train_set = define_Dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['dataloader_batch_size']))
            logger.info('训练图像数量: {:,d}, 迭代次数: {:,d}'.format(len(train_set), train_size))
            # 创建训练数据加载器
            train_loader = DataLoader(train_set,
                                      batch_size=dataset_opt['dataloader_batch_size'],
                                      shuffle=dataset_opt['dataloader_shuffle'],  # 是否打乱数据
                                      num_workers=dataset_opt['dataloader_num_workers'],  # 数据加载线程数
                                      drop_last=True,  # 丢弃最后不完整的批次
                                      pin_memory=True)  # 将数据加载到CUDA固定内存，加速GPU传输
        elif phase == 'test':
            # 创建测试集和测试数据加载器
            test_set = define_Dataset(dataset_opt)
            test_loader = DataLoader(test_set, batch_size=1,
                                     shuffle=False, num_workers=1,
                                     drop_last=False, pin_memory=True)
        else:
            raise NotImplementedError("阶段 [%s] 未被识别." % phase)

    '''
    # ----------------------------------------
    # 步骤3: 初始化模型
    # ----------------------------------------
    '''

    # 根据配置定义模型
    model = define_Model(opt)

    # 如果启用了BN层合并且当前步骤大于合并起点，执行BN层合并
    if opt['merge_bn'] and current_step > opt['merge_bn_startpoint']:
        logger.info('^_^ -----合并BN层----- ^_^')
        model.merge_bnorm_test()

    # 打印网络结构信息
    logger.info(model.info_network())
    # 初始化训练设置（如优化器等）
    model.init_train()
    # 打印模型参数信息
    logger.info(model.info_params())

    '''
    # ----------------------------------------
    # 步骤4: 主训练循环
    # ----------------------------------------
    '''

    for epoch in range(1000000):  # 持续运行（实际会由其他条件终止）
        for i, train_data in enumerate(train_loader):

            current_step += 1

            # 对于'dnpatch'数据集类型，每20000步更新一次数据
            if dataset_type == 'dnpatch' and current_step % 20000 == 0:  # 适用于'train400'数据集
                train_loader.dataset.update_data()

            # -------------------------------
            # 1) 更新学习率（学习率调度）
            # -------------------------------
            model.update_learning_rate(current_step)

            # -------------------------------
            # 2) 输入数据到模型
            # -------------------------------
            model.feed_data(train_data)

            # -------------------------------
            # 3) 优化模型参数
            # -------------------------------
            model.optimize_parameters(current_step)

            # -------------------------------
            # 合并BN层（如果需要）
            # -------------------------------
            if opt['merge_bn'] and opt['merge_bn_startpoint'] == current_step:
                logger.info('^_^ -----合并BN层----- ^_^')
                model.merge_bnorm_train()
                model.print_network()

            # -------------------------------
            # 4) 输出训练信息
            # -------------------------------
            if current_step % opt['train']['checkpoint_print'] == 0:
                logs = model.current_log()  # 获取当前损失等信息
                message = '<轮次:{:3d}, 迭代:{:8,d}, 学习率:{:.3e}> '.format(epoch, current_step, model.current_learning_rate())
                for k, v in logs.items():  # 将日志信息合并到消息中
                    message += '{:s}: {:.3e} '.format(k, v)
                logger.info(message)

            # -------------------------------
            # 5) 保存模型
            # -------------------------------
            if current_step % opt['train']['checkpoint_save'] == 0:
                logger.info('保存模型中...')
                model.save(current_step)

            # -------------------------------
            # 6) 测试模型性能
            # -------------------------------
            if current_step % opt['train']['checkpoint_test'] == 0:

                avg_psnr = 0.0  # 平均峰值信噪比
                idx = 0

                # 遍历测试数据集
                for test_data in test_loader:
                    idx += 1
                    image_name_ext = os.path.basename(test_data['L_path'][0])
                    img_name, ext = os.path.splitext(image_name_ext)

                    # 创建保存图像的目录
                    img_dir = os.path.join(opt['path']['images'], img_name)
                    util.mkdir(img_dir)

                    # 输入测试数据并执行测试
                    model.feed_data(test_data)
                    model.test()

                    # 获取模型输出的可视化结果
                    visuals = model.current_visuals()
                    E_img = util.tensor2uint(visuals['E'])  # 估计的图像
                    H_img = util.tensor2uint(visuals['H'])  # 高质量参考图像

                    # -----------------------
                    # 保存估计的图像E
                    # -----------------------
                    save_img_path = os.path.join(img_dir, '{:s}_{:d}.png'.format(img_name, current_step))
                    util.imsave(E_img, save_img_path)

                    # -----------------------
                    # 计算PSNR（峰值信噪比）
                    # -----------------------
                    current_psnr = util.calculate_psnr(E_img, H_img, border=border)

                    # 输出当前测试图像的PSNR
                    logger.info('{:->4d}--> {:>10s} | {:<4.2f}dB'.format(idx, image_name_ext, current_psnr))

                    avg_psnr += current_psnr

                # 计算平均PSNR
                avg_psnr = avg_psnr / idx

                # 输出测试日志
                logger.info('<轮次:{:3d}, 迭代:{:8,d}, 平均PSNR: {:<.2f}dB\n'.format(epoch, current_step, avg_psnr))

    # 训练结束，保存最终模型
    logger.info('保存最终模型...')
    model.save('latest')
    logger.info('训练结束.')


if __name__ == '__main__':
    main()
