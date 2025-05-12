import os.path
import math
import argparse
import random
import numpy as np
import logging
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch
import wandb
import json
from tqdm import tqdm

from utils import utils_logger
from utils import utils_image as util
from utils import utils_option as option
from utils.utils_dist import get_dist_info, init_dist

from data.select_dataset import define_Dataset
from models.select_model import define_Model


'''
# --------------------------------------------
# 骨扫描去噪任务的训练代码
# --------------------------------------------
# Haowen Zheng(zhenghw24@mails.tsinghua.edu.cn)
# github: https://github.com/Owen1B
'''


def main(json_path='SPECToptions/train_drunet.json'):

    '''
    # ----------------------------------------
    # 步骤1: 准备配置参数
    # ----------------------------------------
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default=json_path, help='Path to option JSON file.')
    parser.add_argument('--launcher', default='pytorch', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--dist', default=False)

    opt = option.parse(parser.parse_args().opt, is_train=True)
    opt['dist'] = parser.parse_args().dist

    # ----------------------------------------
    # 分布式设置
    # ----------------------------------------
    if opt['dist']:
        init_dist('pytorch')
    opt['rank'], opt['world_size'] = get_dist_info()

    if opt['rank'] == 0:
        util.mkdirs((path for key, path in opt['path'].items() if 'pretrained' not in key))

    # ----------------------------------------
    # 更新配置参数
    # ----------------------------------------
    # -->-->-->-->-->-->-->-->-->-->-->-->-->-
    init_iter_G, init_path_G = option.find_last_checkpoint(opt['path']['models'], net_type='G')
    init_iter_E, init_path_E = option.find_last_checkpoint(opt['path']['models'], net_type='E')
    opt['path']['pretrained_netG'] = init_path_G
    opt['path']['pretrained_netE'] = init_path_E
    init_iter_optimizerG, init_path_optimizerG = option.find_last_checkpoint(opt['path']['models'], net_type='optimizerG')
    opt['path']['pretrained_optimizerG'] = init_path_optimizerG

    # Find and set paths for schedulers
    init_iter_schedulerG, init_path_schedulerG = option.find_last_checkpoint(opt['path']['models'], net_type='schedulerG')
    opt['path']['pretrained_schedulerG'] = init_path_schedulerG

    current_step = max(init_iter_G, init_iter_E, init_iter_optimizerG, init_iter_schedulerG)


    # --<--<--<--<--<--<--<--<--<--<--<--<--<-

    # ----------------------------------------
    # 保存配置到'../option.json'文件
    # ----------------------------------------
    if opt['rank'] == 0:
        option.save(opt)

    # ----------------------------------------
    # 对缺失的键返回None
    # ----------------------------------------
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

    # ----------------------------------------
    # 初始化wandb
    # ----------------------------------------
    if opt['rank'] == 0:
        run_id_path = os.path.join(opt['path']['log'], 'wandb_run_id.json')
        if os.path.exists(run_id_path):
            with open(run_id_path, 'r') as f:
                run_id = json.load(f)['run_id']
        else:
            run_id = None
        wandb.init(
            entity=opt['wandb']['entity'],
            project=opt['wandb']['project'],
            name=opt['task'],
            group=opt['wandb']['group'],
            config=opt,
            id=run_id,
            resume="allow" if run_id else None
        )
        with open(run_id_path, 'w') as f:
            json.dump({'run_id': wandb.run.id}, f)

    '''
    # ----------------------------------------
    # 步骤2: 创建数据加载器
    # ----------------------------------------
    '''

    # ----------------------------------------
    # 1) 创建数据集
    # 2) 为训练和测试创建数据加载器
    # ----------------------------------------
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = define_Dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['dataloader_batch_size']))
            if opt['rank'] == 0:
                logger.info('训练图像数量: {:,d}, 迭代次数: {:,d}'.format(len(train_set), train_size))
            if opt['dist']:
                train_sampler = DistributedSampler(train_set, shuffle=dataset_opt['dataloader_shuffle'], drop_last=True, seed=seed)
                train_loader = DataLoader(train_set,
                                          batch_size=dataset_opt['dataloader_batch_size']//opt['num_gpu'],
                                          shuffle=False,
                                          num_workers=dataset_opt['dataloader_num_workers']//opt['num_gpu'],
                                          drop_last=True,
                                          pin_memory=True,
                                          sampler=train_sampler)
            else:
                train_loader = DataLoader(train_set,
                                          batch_size=dataset_opt['dataloader_batch_size'],
                                          shuffle=dataset_opt['dataloader_shuffle'],
                                          num_workers=dataset_opt['dataloader_num_workers'],
                                          drop_last=True,
                                          pin_memory=True)

        elif phase == 'test':
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

    model = define_Model(opt)
    model.init_train()
    if opt['rank'] == 0:
        logger.info(model.info_network())
        logger.info(model.info_params())
    

    '''
    # ----------------------------------------
    # 步骤4: 主要训练过程
    # ----------------------------------------
    '''
    best_psnr = 0
    best_ssim = 0
    
    # 初始化 tqdm 进度条
    if opt['rank'] == 0:
        pbar = tqdm(total=opt['train']['max_iter'], initial=current_step, unit="iter", dynamic_ncols=True)

    for epoch in range(1000000):  # keep running, will be stopped by max_iter
        if opt['dist']:
            train_sampler.set_epoch(epoch + seed)
        
        if opt['rank'] == 0 and 'pbar' in locals(): # 更新进度条的描述信息
            pbar.set_description(f"Train Epoch {epoch}")

        for i, train_data in enumerate(train_loader):
            current_step += 1
            # -------------------------------
            # 1) 喂入patch对
            # -------------------------------
            model.feed_data(train_data)

            # -------------------------------
            # 2) 优化参数
            # -------------------------------
            model.optimize_parameters(current_step)
        
            # -------------------------------
            # 3) 更新学习率
            # -------------------------------
            model.update_learning_rate(current_step)

            # 更新进度条 (在完成一步迭代后)
            if opt['rank'] == 0 and 'pbar' in locals():
                pbar.update(1)

            # -------------------------------
            # 4) 打印训练信息
            # -------------------------------
            if current_step % opt['train']['checkpoint_print'] == 0 and opt['rank'] == 0:
                logs = model.current_log()
                message = '<轮次:{:3d}, 迭代:{:6,d}, 学习率:{:.3e}>'.format(
                    epoch, current_step, model.current_learning_rate())
                for k, v in logs.items():  
                    message += ', {:s}:{:.3e}'.format(k, v)
                logger.info(message)
                wandb_log = {
                    'iteration':current_step,
                    'train/learning_rate': model.current_learning_rate(),
                    **{f'train/{k}': v for k, v in logs.items()}
                }
                wandb.log(wandb_log)
        
            # -------------------------------
            # 5) 评估模型并保存最佳模型及图像
            # -------------------------------
            if current_step % opt['train']['checkpoint_test'] == 0 and opt['rank'] == 0:
                metrics_avg, visuals_list, image_names = model.evaluate_metrics(test_loader)
                message_global = '<轮次:{:3d}, 迭代:{:6,d}>, global测试结果: PSNR:{:<.2f}dB, SSIM:{:<.4f}, LPIPS:{:<.4f}'.format(
                    epoch,
                    current_step,
                    metrics_avg['psnr_global'],
                    metrics_avg['ssim_global'],
                    metrics_avg['lpips_global']
                )
                message_local = '<轮次:{:3d}, 迭代:{:6,d}>, local测试结果: PSNR:{:<.2f}dB, SSIM:{:<.4f}, LPIPS:{:<.4f}'.format(
                    epoch,
                    current_step,
                    metrics_avg['psnr_local'],
                    metrics_avg['ssim_local'],
                    metrics_avg['lpips_local']
                )
                logger.info(message_global)
                logger.info(message_local)
                metrics_dict = {
                    'test/psnr_global': metrics_avg['psnr_global'],
                    'test/ssim_global': metrics_avg['ssim_global'],
                    'test/lpips_global': metrics_avg['lpips_global'],
                    'test/psnr_local': metrics_avg['psnr_local'],
                    'test/ssim_local': metrics_avg['ssim_local'],
                    'test/lpips_local': metrics_avg['lpips_local']
                }
                wandb.log({'iteration': current_step, **metrics_dict})
                
                # 最佳模型的判断基于全局PSNR和全局SSIM (与之前行为保持一致)
                current_psnr_global = metrics_avg['psnr_global']
                current_ssim_global = metrics_avg['ssim_global']

                save_images = False
                
                if current_psnr_global > best_psnr:
                    best_psnr = current_psnr_global
                    model.save_best_network(model.netG, 'G', 'psnr_global', current_step)
                    logger.info('<轮次:{:3d}, 迭代:{:6,d}>, 已保存最佳PSNR_global模型: {:.2f}dB'.format(epoch, current_step, best_psnr))
                    save_images = True
                
                if current_ssim_global > best_ssim:
                    best_ssim = current_ssim_global
                    model.save_best_network(model.netG, 'G', 'ssim_global', current_step)
                    logger.info('<轮次:{:3d}, 迭代:{:6,d}>, 已保存最佳SSIM_global模型: {:.4f}'.format(epoch, current_step, best_ssim))
                    save_images = True
                
                # 最佳模型更新时保存一次图像
                if save_images:
                    for img_array, img_name in zip(visuals_list, image_names):
                        wandb.log({f"best_model_images/{img_name}": wandb.Image(img_array), 'iteration': current_step})

            # -------------------------------
            # 6) 定期保存模型
            # -------------------------------
            if current_step % opt['train']['checkpoint_save'] == 0 and opt['rank'] == 0:
                model.save(current_step)
                logger.info('<轮次:{:3d}, 迭代:{:6,d}>, 已定期保存模型'.format(epoch, current_step))

            # -------------------------------
            # 7) 检查是否达到最大迭代次数
            # -------------------------------
            if 'max_iter' in opt['train'] and current_step >= opt['train']['max_iter']:
                if opt['rank'] == 0:
                    logger.info('<迭代:{:6,d}>, 已达到最大迭代次数 {:,d}, 训练结束'.format(
                        current_step, opt['train']['max_iter']))
                    # 保存最终模型
                    model.save(current_step)
                break
        
        # 达到最大迭代次数后跳出外层epoch循环
        if 'max_iter' in opt['train'] and current_step >= opt['train']['max_iter']:
            if opt['rank'] == 0 and 'pbar' in locals():
                pbar.close()
            break

    # 在训练结束时关闭wandb 和 tqdm (如果循环正常结束且pbar存在)
    if opt['rank'] == 0:
        if 'pbar' in locals() and pbar.n < pbar.total: # Check if pbar was created and not already closed by break
            pbar.close()
        wandb.finish()

if __name__ == '__main__':
    main()
