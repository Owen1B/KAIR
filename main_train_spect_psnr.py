import os.path
import math
import argparse
import random
import numpy as np
import logging
import csv
import matplotlib.pyplot as plt
import shutil
import threading
import queue
import time
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


class AsyncWandBUploader:
    """异步WandB上传器，避免图片上传阻塞训练过程"""
    
    def __init__(self, max_queue_size=100):
        self.upload_queue = queue.Queue(maxsize=max_queue_size)
        self.worker_thread = None
        self.stop_event = threading.Event()
        self.start_worker()
    
    def start_worker(self):
        """启动上传工作线程"""
        self.worker_thread = threading.Thread(target=self._upload_worker, daemon=True)
        self.worker_thread.start()
    
    def _upload_worker(self):
        """上传工作线程主函数"""
        while not self.stop_event.is_set():
            try:
                # 从队列获取上传任务，超时1秒
                upload_data = self.upload_queue.get(timeout=1.0)
                if upload_data is None:  # 停止信号
                    break
                
                # 执行实际上传
                try:
                    # 检查是否需要处理图片数据
                    processed_data = {}
                    for key, value in upload_data.items():
                        if isinstance(value, tuple) and len(value) == 2:
                            # 这是原始图片数据 (img_array, img_name)
                            img_array, img_name = value
                            processed_data[key] = wandb.Image(img_array)
                        else:
                            processed_data[key] = value
                    
                    wandb.log(processed_data)
                except Exception as e:
                    print(f"WandB上传失败: {e}")
                finally:
                    self.upload_queue.task_done()
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"上传工作线程异常: {e}")
    
    def log_async(self, data_dict):
        """异步提交日志数据到上传队列"""
        try:
            self.upload_queue.put(data_dict, block=False)
        except queue.Full:
            print("警告: WandB上传队列已满，跳过本次图片上传")
    
    def log_async_raw_image(self, img_array, img_key, iteration):
        """异步提交原始图片数据，避免在主线程中创建wandb.Image对象"""
        try:
            # 将原始图片数据和键值作为元组传递，在工作线程中处理
            upload_data = {
                img_key: (img_array, img_key),
                'iteration': iteration
            }
            self.upload_queue.put(upload_data, block=False)
        except queue.Full:
            print("警告: WandB上传队列已满，跳过本次图片上传")
    
    def log_sync(self, data_dict):
        """同步上传（用于重要的度量数据）"""
        wandb.log(data_dict)
    
    def stop_and_wait(self, timeout=30):
        """停止上传器并等待所有任务完成"""
        # 等待队列中的任务完成
        self.upload_queue.join()
        
        # 发送停止信号
        self.stop_event.set()
        self.upload_queue.put(None)  # 唤醒工作线程
        
        # 等待工作线程结束
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=timeout)


def _plot_scatter(x_data, y_data, c_data, x_label, y_label, c_label, title, plot_path):
    """Helper function to create a scatter plot."""
    fig, ax = plt.subplots(figsize=(8, 8))
    scatter = ax.scatter(x_data, y_data, alpha=0.7, c=c_data, cmap='viridis')
    
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label(c_label)

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(plot_path)
    plt.close(fig)
    return plot_path

def log_and_plot_correlations(log_dir, current_step, test_lpips, test_ssim, test_psnr, val_lpips, async_uploader=None):
    """记录关键指标到CSV并绘制相关性散点图"""
    csv_path = os.path.join(log_dir, 'metrics_correlation.csv')

    # 1. 记录当前数据到CSV
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(['step', 'test_lpips_local', 'test_ssim_local', 'test_psnr_local', 'val_2_poisson_lpips_local'])
        writer.writerow([current_step, test_lpips, test_ssim, test_psnr, val_lpips])

    # 2. 读取所有历史数据
    steps, test_lpips_data, test_ssim_data, test_psnr_data, val_lpips_data = [], [], [], [], []
    with open(csv_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # 跳过表头
        for row in reader:
            steps.append(int(row[0]))
            test_lpips_data.append(float(row[1]))
            test_ssim_data.append(float(row[2]))
            test_psnr_data.append(float(row[3]))
            val_lpips_data.append(float(row[4]))
            
    # 3. 绘制并记录三个散点图
    plots = {
        'LPIPS_Correlation': {
            'x_data': test_lpips_data, 'y_data': val_lpips_data,
            'x_label': 'test/lpips_local (no noise)', 'y_label': 'val_2_poisson/lpips_local (with noise)',
            'path': os.path.join(log_dir, 'correlation_lpips.png')
        },
        'SSIM_vs_LPIPS_Correlation': {
            'x_data': test_ssim_data, 'y_data': val_lpips_data,
            'x_label': 'test/ssim_local (no noise)', 'y_label': 'val_2_poisson/lpips_local (with noise)',
            'path': os.path.join(log_dir, 'correlation_ssim_vs_lpips.png')
        },
        'PSNR_vs_LPIPS_Correlation': {
            'x_data': test_psnr_data, 'y_data': val_lpips_data,
            'x_label': 'test/psnr_local (no noise)', 'y_label': 'val_2_poisson/lpips_local (with noise)',
            'path': os.path.join(log_dir, 'correlation_psnr_vs_lpips.png')
        }
    }
    
    wandb_images = {}
    for title, data in plots.items():
        plot_path = _plot_scatter(
            x_data=data['x_data'], y_data=data['y_data'], c_data=steps,
            x_label=data['x_label'], y_label=data['y_label'], c_label='Iteration Step',
            title=title.replace('_', ' '), plot_path=data['path']
        )
        wandb_images[f'correlation_plots/{title}'] = wandb.Image(plot_path)

    # 使用异步上传器上传图片
    if async_uploader:
        # 分别处理每个图片，避免在主线程中创建wandb.Image对象
        for key, img_obj in wandb_images.items():
            # 获取原始图片路径并异步上传
            if hasattr(img_obj, '_path'):
                async_uploader.log_async({key: img_obj, 'iteration': current_step})
            else:
                async_uploader.log_async({key: img_obj, 'iteration': current_step})
    else:
        upload_data = {**wandb_images, 'iteration': current_step}
        wandb.log(upload_data)


def main(json_path='SPECToptions/train_drunet_psnr_raw.json'):

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
    parser.add_argument('--restart', action='store_true', help='Remove existing experiment directory and restart training')

    opt = option.parse(parser.parse_args().opt, is_train=True)
    opt['dist'] = parser.parse_args().dist
    restart = parser.parse_args().restart

    # ----------------------------------------
    # 处理重新训练选项
    # ----------------------------------------
    if restart:
        # 获取任务目录路径，应该删除的是整个任务目录而不是根目录
        if 'path' in opt and 'task' in opt['path']:
            # 任务目录就是实验的完整目录，包含models、images、options等子目录
            task_path = opt['path']['task']
            
            if os.path.exists(task_path):
                print(f'重新训练模式：删除现有任务目录 {task_path}')
                try:
                    shutil.rmtree(task_path)
                    print(f'成功删除目录: {task_path}')
                except Exception as e:
                    print(f'删除目录时出错: {e}')
                    print('继续训练...')
            else:
                print(f'任务目录 {task_path} 不存在，继续训练...')

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
        
        # 初始化异步上传器
        async_uploader = AsyncWandBUploader(max_queue_size=200)
        logger.info("已初始化异步WandB上传器，图片上传将在后台进行")

    '''
    # ----------------------------------------
    # 步骤2: 创建数据加载器
    # ----------------------------------------
    '''

    # ----------------------------------------
    # 1) 创建数据集
    # 2) 为训练、验证和测试创建数据加载器
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
        elif phase.startswith('val_'):
            # 为每个验证集创建数据加载器
            val_set = define_Dataset(dataset_opt)
            val_loader = DataLoader(val_set, batch_size=1,
                                   shuffle=False, num_workers=1,
                                   drop_last=False, pin_memory=True)
            # 将验证集加载器存储在字典中
            if 'val_loaders' not in locals():
                val_loaders = {}
            val_loaders[phase] = val_loader
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
                message = '<轮次:{:3d}, 迭代:{:6,d}>, 学习率:{:.3e}'.format(
                    epoch, current_step, model.current_learning_rate())
                for k, v in logs.items():  
                    message += ', {:s}:{:.3e}'.format(k, v)
                logger.info(message)
                wandb_log = {
                    'iteration':current_step,
                    'train/learning_rate': model.current_learning_rate(),
                    **{f'train/{k}': v for k, v in logs.items()}
                }
                # 训练度量数据使用同步上传，确保及时性
                async_uploader.log_sync(wandb_log)
        
            # -------------------------------
            # 5) 评估模型并保存最佳模型及图像
            # -------------------------------
            if current_step % opt['train']['checkpoint_test'] == 0 and opt['rank'] == 0:
                # 定义变量以存储用于CSV记录的指标
                test_lpips_local_for_csv = None
                test_ssim_local_for_csv = None
                test_psnr_local_for_csv = None
                val2_poisson_lpips_local_for_csv = None

                # 清理训练时GPU上的数据，为评估腾出空间
                if hasattr(model, 'L'):
                    del model.L
                if hasattr(model, 'H'):
                    del model.H
                if hasattr(model, 'E'): # model.E 是前向传播的结果
                    del model.E
                torch.cuda.empty_cache() # 请求PyTorch释放未使用的缓存显存

                # 统一日志格式
                msg_format_basic = '<轮次:{:3d}, 迭代:{:6,d}>, {:<20s}: PSNR:{:<.2f}dB, SSIM:{:<.4f}, LPIPS:{:<.4f}'
                msg_format_poissonll = msg_format_basic + ', PoissonLL:{:<.4f}'

                # 测试集评估
                metrics_avg, visuals_list, image_names = model.evaluate_metrics(test_loader, add_poisson_noise=False)
                test_lpips_local_for_csv = metrics_avg['lpips_local'] # 获取test/lpips_local
                test_ssim_local_for_csv = metrics_avg['ssim_local']
                test_psnr_local_for_csv = metrics_avg['psnr_local']
                message_global = msg_format_basic.format(
                    epoch,
                    current_step,
                    'test_global',
                    metrics_avg['psnr_global'],
                    metrics_avg['ssim_global'],
                    metrics_avg['lpips_global']
                )
                message_local = msg_format_basic.format(
                    epoch,
                    current_step,
                    'test_local',
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
                    'test/lpips_local': metrics_avg['lpips_local'],
                    'test/loss': metrics_avg['loss']
                }
                # 度量数据同步上传
                async_uploader.log_sync({'iteration': current_step, **metrics_dict})

                # 测试集评估 (带泊松噪声)
                metrics_avg_poisson, visuals_list_poisson, image_names_poisson = model.evaluate_metrics(test_loader, add_poisson_noise=True)
                message_global_poisson = msg_format_basic.format(
                    epoch,
                    current_step,
                    'test_poisson_global',
                    metrics_avg_poisson['psnr_global'],
                    metrics_avg_poisson['ssim_global'],
                    metrics_avg_poisson['lpips_global']
                )
                message_local_poisson = msg_format_basic.format(
                    epoch,
                    current_step,
                    'test_poisson_local',
                    metrics_avg_poisson['psnr_local'],
                    metrics_avg_poisson['ssim_local'],
                    metrics_avg_poisson['lpips_local']
                )
                logger.info(message_global_poisson)
                logger.info(message_local_poisson)
                metrics_dict_poisson = {
                    'test_poisson/psnr_global': metrics_avg_poisson['psnr_global'],
                    'test_poisson/ssim_global': metrics_avg_poisson['ssim_global'],
                    'test_poisson/lpips_global': metrics_avg_poisson['lpips_global'],
                    'test_poisson/psnr_local': metrics_avg_poisson['psnr_local'],
                    'test_poisson/ssim_local': metrics_avg_poisson['ssim_local'],
                    'test_poisson/lpips_local': metrics_avg_poisson['lpips_local'],
                    'test_poisson/loss': metrics_avg_poisson['loss']
                }
                # 度量数据同步上传
                async_uploader.log_sync({'iteration': current_step, **metrics_dict_poisson})


                # 验证集评估（如果存在）
                val_images_data = {}  # 存储验证集图片数据，避免重复评估
                if 'val_loaders' in locals():
                    for val_name, val_loader in val_loaders.items():
                        # 不加泊松噪声
                        val_metrics_avg, val_visuals_list, val_image_names = model.evaluate_metrics(val_loader, add_poisson_noise=False)
                        # 保存图片数据供后续使用，避免重复评估
                        val_images_data[f'{val_name}'] = (val_visuals_list, val_image_names)
                        val_message_global = msg_format_poissonll.format(
                            epoch,
                            current_step,
                            f'{val_name}_global',
                            val_metrics_avg['psnr_global'],
                            val_metrics_avg['ssim_global'],
                            val_metrics_avg['lpips_global'],
                            val_metrics_avg['poisson_ll']
                        )
                        val_message_local = msg_format_basic.format(
                            epoch,
                            current_step,
                            f'{val_name}_local',
                            val_metrics_avg['psnr_local'],
                            val_metrics_avg['ssim_local'],
                            val_metrics_avg['lpips_local']
                        )
                        logger.info(val_message_global)
                        logger.info(val_message_local)
                        val_metrics_dict = {
                            f'{val_name}/psnr_global': val_metrics_avg['psnr_global'],
                            f'{val_name}/ssim_global': val_metrics_avg['ssim_global'],
                            f'{val_name}/lpips_global': val_metrics_avg['lpips_global'],
                            f'{val_name}/psnr_local': val_metrics_avg['psnr_local'],
                            f'{val_name}/ssim_local': val_metrics_avg['ssim_local'],
                            f'{val_name}/lpips_local': val_metrics_avg['lpips_local'],
                            f'{val_name}/loss': val_metrics_avg['loss'],
                            f'{val_name}/poisson_ll': val_metrics_avg['poisson_ll']
                        }
                        # 度量数据同步上传
                        async_uploader.log_sync({'iteration': current_step, **val_metrics_dict})

                        # 加泊松噪声
                        val_dataset_opt = opt['datasets'][val_name]
                        lpips_repeats = val_dataset_opt.get('lpips_local_repeat_n', 1)
                        val_metrics_avg_poisson, val_visuals_list_poisson, val_image_names_poisson = model.evaluate_metrics(val_loader, add_poisson_noise=True, lpips_local_repeat_n=lpips_repeats)
                        # 保存泊松噪声图片数据供后续使用，避免重复评估
                        val_images_data[f'{val_name}_poisson'] = (val_visuals_list_poisson, val_image_names_poisson)
                        
                        if val_name == 'val_2':
                            val2_poisson_lpips_local_for_csv = val_metrics_avg_poisson['lpips_local'] # 获取val_2_poisson/lpips_local

                        val_message_global_poisson = msg_format_basic.format(
                            epoch,
                            current_step,
                            f'{val_name}_poisson_global',
                            val_metrics_avg_poisson['psnr_global'],
                            val_metrics_avg_poisson['ssim_global'],
                            val_metrics_avg_poisson['lpips_global']
                        )
                        val_message_local_poisson = msg_format_basic.format(
                            epoch,
                            current_step,
                            f'{val_name}_poisson_local',
                            val_metrics_avg_poisson['psnr_local'],
                            val_metrics_avg_poisson['ssim_local'],
                            val_metrics_avg_poisson['lpips_local']
                        )
                        logger.info(val_message_global_poisson)
                        logger.info(val_message_local_poisson)
                        val_metrics_dict_poisson = {
                            f'{val_name}_poisson/psnr_global': val_metrics_avg_poisson['psnr_global'],
                            f'{val_name}_poisson/ssim_global': val_metrics_avg_poisson['ssim_global'],
                            f'{val_name}_poisson/lpips_global': val_metrics_avg_poisson['lpips_global'],
                            f'{val_name}_poisson/psnr_local': val_metrics_avg_poisson['psnr_local'],
                            f'{val_name}_poisson/ssim_local': val_metrics_avg_poisson['ssim_local'],
                            f'{val_name}_poisson/lpips_local': val_metrics_avg_poisson['lpips_local'],
                            f'{val_name}_poisson/loss': val_metrics_avg_poisson['loss']
                        }
                        # 度量数据同步上传
                        async_uploader.log_sync({'iteration': current_step, **val_metrics_dict_poisson})
                # 在所有评估完成后，检查是否需要记录CSV和绘图
                if test_lpips_local_for_csv is not None and val2_poisson_lpips_local_for_csv is not None:
                    log_and_plot_correlations(
                        log_dir=opt['path']['log'], 
                        current_step=current_step,
                        test_lpips=test_lpips_local_for_csv, 
                        test_ssim=test_ssim_local_for_csv,
                        test_psnr=test_psnr_local_for_csv,
                        val_lpips=val2_poisson_lpips_local_for_csv,
                        async_uploader=async_uploader
                    )
                
                # 最佳模型的判断基于全局PSNR和全局SSIM (与之前行为保持一致，不使用泊松噪声后的结果判断最佳模型)
                current_psnr_global = metrics_avg['psnr_global'] # 使用未加噪声的结果进行判断
                current_ssim_global = metrics_avg['ssim_global'] # 使用未加噪声的结果进行判断

                # 根据配置文件决定图像保存策略
                save_images_always = opt['train'].get('save_images', False)
                save_images = save_images_always  # 如果配置为True，则每次都保存
                
                if current_psnr_global > best_psnr:
                    best_psnr = current_psnr_global
                    model.save_best_network(model.netG, 'G', 'psnr_global', current_step)
                    logger.info('<轮次:{:3d}, 迭代:{:6,d}>, 已保存最佳PSNR_global模型: {:.2f}dB'.format(epoch, current_step, best_psnr))
                    if not save_images_always:  # 如果不是每次都保存，则在最佳模型时保存
                        save_images = True
                
                if current_ssim_global > best_ssim:
                    best_ssim = current_ssim_global
                    model.save_best_network(model.netG, 'G', 'ssim_global', current_step)
                    logger.info('<轮次:{:3d}, 迭代:{:6,d}>, 已保存最佳SSIM_global模型: {:.4f}'.format(epoch, current_step, best_ssim))
                    if not save_images_always:  # 如果不是每次都保存，则在最佳模型时保存
                        save_images = True

                # 根据配置决定保存图像的时机
                if save_images:
                    # 使用已有的测试集图片数据，避免重复评估
                    for img_array, img_name in zip(visuals_list, image_names):
                        async_uploader.log_async_raw_image(img_array, f"images_test/{img_name}", current_step)
                    # 同时保存带泊松噪声的测试集图像 (如果需要)
                    for img_array, img_name in zip(visuals_list_poisson, image_names_poisson):
                        async_uploader.log_async_raw_image(img_array, f"images_test_poisson/{img_name}", current_step)

                    # 保存验证集图像 - 使用之前评估时已经获得的图片数据，避免重复评估
                    if 'val_images_data' in locals():
                        for key, (visuals_list, image_names) in val_images_data.items():
                            for img_array, img_name in zip(visuals_list, image_names):
                                async_uploader.log_async_raw_image(img_array, f"images_{key}/{img_name}", current_step)

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
        
        # 等待所有异步上传任务完成
        if 'async_uploader' in locals():
            logger.info("正在等待异步上传任务完成...")
            async_uploader.stop_and_wait(timeout=60)
            logger.info("异步上传任务已完成")
        
        wandb.finish()

if __name__ == '__main__':
    main()
