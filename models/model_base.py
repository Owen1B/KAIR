import os
import torch
import torch.nn as nn
from utils.utils_bnorm import merge_bn, tidy_sequential
from torch.nn.parallel import DataParallel, DistributedDataParallel
import matplotlib.pyplot as plt
import numpy as np
import os
from utils import utils_image as util
from utils import utils_spect
from pytorch_fid import fid_score
class ModelBase():
    def __init__(self, opt):
        self.opt = opt                         # opt
        self.save_dir = opt['path']['models']  # save models
        self.device = torch.device('cuda' if opt['gpu_ids'] is not None else 'cpu')
        self.is_train = opt['is_train']        # training or not
        self.schedulers = []                   # schedulers

    """
    # ----------------------------------------
    # Preparation before training with data
    # Save model during training
    # ----------------------------------------
    """

    def init_train(self):
        pass

    def load(self):
        pass

    def save(self, label):
        pass

    def define_loss(self):
        pass

    def define_optimizer(self):
        pass

    def define_scheduler(self):
        pass

    """
    # ----------------------------------------
    # Optimization during training with data
    # Testing/evaluation
    # ----------------------------------------
    """

    def feed_data(self, data):
        pass

    def optimize_parameters(self):
        pass

    def current_visuals(self):
        pass

    def current_losses(self):
        pass

    def update_learning_rate(self, n):
        for scheduler in self.schedulers:
            scheduler.step()

    def current_learning_rate(self):
        return self.schedulers[0].get_last_lr()[0]

    def requires_grad(self, model, flag=True):
        for p in model.parameters():
            p.requires_grad = flag

    """
    # ----------------------------------------
    # Information of net
    # ----------------------------------------
    """

    def print_network(self):
        pass

    def info_network(self):
        pass

    def print_params(self):
        pass

    def info_params(self):
        pass

    def get_bare_model(self, network):
        """Get bare model, especially under wrapping with
        DistributedDataParallel or DataParallel.
        """
        if isinstance(network, (DataParallel, DistributedDataParallel)):
            network = network.module
        return network

    def model_to_device(self, network):
        """Model to device. It also warps models with DistributedDataParallel
        or DataParallel.
        Args:
            network (nn.Module)
        """
        network = network.to(self.device)
        if self.opt['dist']:
            find_unused_parameters = self.opt.get('find_unused_parameters', True)
            use_static_graph = self.opt.get('use_static_graph', False)
            network = DistributedDataParallel(network, device_ids=[torch.cuda.current_device()], find_unused_parameters=find_unused_parameters)
            if use_static_graph:
                print('Using static graph. Make sure that "unused parameters" will not change during training loop.')
                network._set_static_graph()
        else:
            network = DataParallel(network)
        return network

    # ----------------------------------------
    # network name and number of parameters
    # ----------------------------------------
    def describe_network(self, network):
        network = self.get_bare_model(network)
        msg = '\n'
        msg += 'Networks name: {}'.format(network.__class__.__name__) + '\n'
        msg += 'Params number: {}'.format(sum(map(lambda x: x.numel(), network.parameters()))) + '\n'
        msg += 'Net structure:\n{}'.format(str(network)) + '\n'
        return msg

    # ----------------------------------------
    # parameters description
    # ----------------------------------------
    def describe_params(self, network):
        network = self.get_bare_model(network)
        msg = '\n'
        msg += ' | {:^6s} | {:^6s} | {:^6s} | {:^6s} || {:<20s}'.format('mean', 'min', 'max', 'std', 'shape', 'param_name') + '\n'
        for name, param in network.state_dict().items():
            if not 'num_batches_tracked' in name:
                v = param.data.clone().float()
                msg += ' | {:>6.3f} | {:>6.3f} | {:>6.3f} | {:>6.3f} | {} || {:s}'.format(v.mean(), v.min(), v.max(), v.std(), v.shape, name) + '\n'
        return msg

    """
    # ----------------------------------------
    # Save prameters
    # Load prameters
    # ----------------------------------------
    """

    # ----------------------------------------
    # save the state_dict of the network
    # ----------------------------------------
    def save_network(self, save_dir, network, network_label, iter_label):
        save_filename = '{}_{}.pth'.format(iter_label, network_label)
        save_path = os.path.join(save_dir, save_filename)
        network = self.get_bare_model(network)
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)

    # ----------------------------------------
    # save the state_dict of the best network
    # ----------------------------------------
    def save_best_network(self, network, network_label, name_suffix='', current_step=None):
        best_model_folder_name = 'bestmodel'
        full_best_model_dir = os.path.join(self.save_dir, best_model_folder_name)
        
        # Ensure the directory exists
        os.makedirs(full_best_model_dir, exist_ok=True)
        
        # 如果提供了迭代次数，则在文件名中包含
        if current_step is not None:
            save_filename = f'best_{name_suffix}_{network_label}_{current_step}.pth'
        else:
            save_filename = f'best_{name_suffix}_{network_label}.pth'
        
        save_path = os.path.join(full_best_model_dir, save_filename)

        # 删除同类型的旧的最佳模型（仅删除相同 name_suffix 和 network_label 的旧模型）
        for old_file in os.listdir(full_best_model_dir):
            if old_file.startswith(f'best_{name_suffix}_{network_label}') and old_file != save_filename:
                old_file_path = os.path.join(full_best_model_dir, old_file)
                os.remove(old_file_path)
                    

        network = self.get_bare_model(network)
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)

    # ----------------------------------------
    # load the state_dict of the network
    # ----------------------------------------
    def load_network(self, load_path, network, strict=True, param_key='params'):
        network = self.get_bare_model(network)
        if strict:
            state_dict = torch.load(load_path)
            if param_key in state_dict.keys():
                state_dict = state_dict[param_key]
            network.load_state_dict(state_dict, strict=strict)
        else:
            state_dict_old = torch.load(load_path)
            if param_key in state_dict_old.keys():
                state_dict_old = state_dict_old[param_key]
            state_dict = network.state_dict()
            for ((key_old, param_old),(key, param)) in zip(state_dict_old.items(), state_dict.items()):
                state_dict[key] = param_old
            network.load_state_dict(state_dict, strict=True)
            del state_dict_old, state_dict

    # ----------------------------------------
    # save the state_dict of the optimizer
    # ----------------------------------------
    def save_optimizer(self, save_dir, optimizer, optimizer_label, iter_label):
        save_filename = '{}_{}.pth'.format(iter_label, optimizer_label)
        save_path = os.path.join(save_dir, save_filename)
        torch.save(optimizer.state_dict(), save_path)

    # ----------------------------------------
    # load the state_dict of the optimizer
    # ----------------------------------------
    def load_optimizer(self, load_path, optimizer):
        optimizer.load_state_dict(torch.load(load_path, map_location=lambda storage, loc: storage.cuda(torch.cuda.current_device())))

    # ----------------------------------------
    # save the state_dict of the scheduler
    # ----------------------------------------
    def save_scheduler(self, save_dir, scheduler, scheduler_label, iter_label):
        save_filename = '{}_{}.pth'.format(iter_label, scheduler_label)
        save_path = os.path.join(save_dir, save_filename)
        torch.save(scheduler.state_dict(), save_path)

    # ----------------------------------------
    # load the state_dict of the scheduler
    # ----------------------------------------
    def load_scheduler(self, load_path, scheduler):
        print(f"Loading scheduler state from {load_path}")
        scheduler.load_state_dict(torch.load(load_path, map_location=lambda storage, loc: storage.cuda(torch.cuda.current_device())))

    def update_E(self, decay=0.999):
        netG = self.get_bare_model(self.netG)
        netG_params = dict(netG.named_parameters())
        netE_params = dict(self.netE.named_parameters())
        for k in netG_params.keys():
            netE_params[k].data.mul_(decay).add_(netG_params[k].data, alpha=1-decay)

    """
    # ----------------------------------------
    # Merge Batch Normalization for training
    # Merge Batch Normalization for testing
    # ----------------------------------------
    """

    # ----------------------------------------
    # merge bn during training
    # ----------------------------------------
    def merge_bnorm_train(self):
        merge_bn(self.netG)
        tidy_sequential(self.netG)
        self.define_optimizer()
        self.define_scheduler()

    # ----------------------------------------
    # merge bn before testing
    # ----------------------------------------
    def merge_bnorm_test(self):
        merge_bn(self.netG)
        tidy_sequential(self.netG)

    # ----------------------------------------
    # evaluate metrics and generate visuals
    # ----------------------------------------
    def evaluate_metrics(self, test_loader):
        """评估模型性能,计算PSNR、SSIM、LPIPS指标,并保存可视化结果
        Args:
            test_loader: 测试数据加载器
        Returns:
            metrics_avg: 包含平均指标的字典
            {
                'psnr_global': float, 'ssim_global': float, 'lpips_global': float,
                'psnr_local': float, 'ssim_local': float, 'lpips_local': float
            }
            visuals_list: 所有测试样本的可视化图像列表
            image_names: 所有测试样本的文件名列表
        """
        self.netG.eval()
        
        # 初始化指标累加器
        metrics_sum_global = {'psnr': 0, 'ssim': 0, 'lpips': 0}
        metrics_sum_local = {'psnr': 0, 'ssim': 0, 'lpips': 0}
        loss_sum = 0
        count = 0
        
        # 存储所有图像数据和文件名
        all_imgs = {'L': [], 'E': [], 'H': []}
        image_names = []
        
        # 创建保存目录 (图像保存逻辑不变, 仍基于全局归一化的视觉效果)
        save_dir = os.path.join(self.opt['path']['images'])
        for key in ['L', 'E', 'H']:
            os.makedirs(os.path.join(save_dir, key), exist_ok=True)
        
        # 设置返回路径信息
        test_loader.dataset.return_paths = True
        
        # 遍历：收集所有图像数据
        with torch.no_grad():
            for test_data in test_loader:
                self.feed_data(test_data)
                self.test()
                visuals = self.current_visuals()
                
                # 计算loss
                loss = self.G_lossfn_weight * self.G_lossfn(self.E, self.H)
                loss_sum += loss.item()
                
                image_name_ext = os.path.basename(test_data['L_path'][0])
                image_name = os.path.splitext(image_name_ext)[0]
                image_names.append(image_name)
                
                for key in ['L', 'E', 'H']:
                    img = visuals[key].cpu().float().numpy().transpose((1, 2, 0))
                    img_norm = utils_spect.denormalize_spect(
                                img, 
                                self.opt['datasets']['test']['normalization']['type'],
                                self.opt['datasets']['test']['normalization']['max_pixel']
                            )
                    all_imgs[key].append(img_norm)
        
        # 计算全局最大最小值 (用于全局归一化指标和图像保存)
        all_H_values = np.concatenate([img.flatten() for img in all_imgs['H']])
        max_val_global = np.max(all_H_values) * 1  # 设置为H最大值的110%
        min_val_global = 0  # 固定最小值为0
        
        # 计算每张图片的指标并生成可视化
        visuals_list = []
        
        for idx, imgs_original_scale in enumerate(zip(*[all_imgs[k] for k in ['L', 'E', 'H']])):
            L_img_norm, E_img_norm, H_img_norm = imgs_original_scale
            count += 1
            image_name = image_names[idx]
            
            # --- 全局归一化处理 (用于图像保存和全局指标) ---
            imgs_255_global = {}
            for key, img in zip(['L', 'E', 'H'], [L_img_norm, E_img_norm, H_img_norm]):
                # 先clip到[0, max_val_global]
                img_clipped = np.clip(img, 0, max_val_global)
                # 归一化到[0,255]
                img_255 = (img_clipped / max_val_global * 255).astype(np.uint8)
                imgs_255_global[key] = img_255
                
                # 保存每个通道为RGB图
                for ch in range(img.shape[2]):
                    gray_img = img_255[:, :, ch]
                    rgb_img = np.stack([gray_img, gray_img, gray_img], axis=2)
                    rgb_path = os.path.join(save_dir, key, f"{image_name}_ch{ch}.png")
                    util.imsave(rgb_img, rgb_path)
            
            # 从保存的图像中读取并计算全局指标
            for ch in range(L_img_norm.shape[2]):
                H_rgb_global_saved = util.imread_uint(os.path.join(save_dir, 'H', f"{image_name}_ch{ch}.png"), n_channels=3)
                E_rgb_global_saved = util.imread_uint(os.path.join(save_dir, 'E', f"{image_name}_ch{ch}.png"), n_channels=3)
                psnr_global = util.calculate_psnr(E_rgb_global_saved, H_rgb_global_saved)
                ssim_global = util.calculate_ssim(E_rgb_global_saved, H_rgb_global_saved)
                lpips_global = util.calculate_lpips(E_rgb_global_saved, H_rgb_global_saved)
                metrics_sum_global['psnr'] += psnr_global
                metrics_sum_global['ssim'] += ssim_global
                metrics_sum_global['lpips'] += lpips_global

            # --- 局部/自适应归一化处理 (仅用于局部指标计算) ---
            max_val_local = np.max(H_img_norm) * 1  # 设置为当前H最大值的110%
            min_val_local = 0  # 固定最小值为0
            
            for ch in range(L_img_norm.shape[2]):
                # 先clip到[0, max_val_local]
                E_ch_clipped = np.clip(E_img_norm[:,:,ch], 0, max_val_local)
                H_ch_clipped = np.clip(H_img_norm[:,:,ch], 0, max_val_local)
                
                # 归一化到[0,255]
                E_ch_local = (E_ch_clipped / max_val_local * 255).astype(np.uint8)
                H_ch_local = (H_ch_clipped / max_val_local * 255).astype(np.uint8)
                
                # 扩展为3通道RGB图像以供度量函数使用
                E_rgb_local = np.stack([E_ch_local, E_ch_local, E_ch_local], axis=2)
                H_rgb_local = np.stack([H_ch_local, H_ch_local, H_ch_local], axis=2)
                psnr_local = util.calculate_psnr(E_rgb_local, H_rgb_local)
                ssim_local = util.calculate_ssim(E_rgb_local, H_rgb_local)
                lpips_local = util.calculate_lpips(E_rgb_local, H_rgb_local)
                metrics_sum_local['psnr'] += psnr_local
                metrics_sum_local['ssim'] += ssim_local
                metrics_sum_local['lpips'] += lpips_local

            # 为每个样本创建可视化图像 (基于全局归一化的图像)
            fig = plt.figure(figsize=(8, 20))
            gs = plt.GridSpec(2, 4, height_ratios=[1, 1], width_ratios=[1, 1, 1, 0.05])
            
            titles = {
                'L': 'Input (L)',
                'E': 'Estimated (E)',
                'H': 'Ground Truth (H)'
            }
            
            sample_imgs_for_vis = {'L': L_img_norm, 'E': E_img_norm, 'H': H_img_norm}
            # 可视化时使用局部的min/max
            vmax_vis = max_val_local
            vmin_vis = min_val_local

            # 添加大标题显示PSNR和SSIM
            plt.suptitle(f'PSNR: {psnr_local:.2f}dB, SSIM: {ssim_local:.4f}', fontsize=16)

            for row, view in enumerate(['Anterior', 'Posterior']):
                for col, (key, title) in enumerate(titles.items()):
                    ax = plt.subplot(gs[row, col])
                    # 使用原始图像配合局部的min_val_local, max_val_local进行显示
                    im = ax.imshow(sample_imgs_for_vis[key][:,:,row], cmap='gray', vmin=vmin_vis, vmax=vmax_vis)
                    ax.set_title(f'{title} - {view}')
                    ax.axis('off')
            
            cax = plt.subplot(gs[:, 3])
            plt.colorbar(im, cax=cax) # colorbar 应该对应imshow的im对象
            
            plt.tight_layout(rect=[0, 0, 1, 0.95])  # 为顶部标题留出空间
            
            fig.canvas.draw()
            img_array = np.array(fig.canvas.renderer.buffer_rgba())
            visuals_list.append(img_array)
            plt.close(fig)
        
        # FID计算逻辑保持不变 (如果启用)
        # # FID计算需要从文件夹路径计算
        # fid_value = fid_score.calculate_fid_given_paths([os.path.join(save_dir, 'H'), os.path.join(save_dir, 'E')], 
        #                                                 batch_size=50, device=self.device, dims=2048)
        # metrics_sum['fid'] = fid_value # 需要决定fid是全局还是局部，或者独立
        
        # 计算平均值
        total_channels = count * L_img_norm.shape[2] if count > 0 and L_img_norm.ndim == 3 else count # Handle cases where L_img_norm might not have channels dim if issues
        if total_channels == 0: # Avoid division by zero if no valid data processed
            metrics_avg = {
                'psnr_global': 0, 'ssim_global': 0, 'lpips_global': 0,
                'psnr_local': 0, 'ssim_local': 0, 'lpips_local': 0,
                'loss': 0
            }
        else:
            metrics_avg = {
                'psnr_global': metrics_sum_global['psnr'] / total_channels,
                'ssim_global': metrics_sum_global['ssim'] / total_channels,
                'lpips_global': metrics_sum_global['lpips'] / total_channels,
                'psnr_local': metrics_sum_local['psnr'] / total_channels,
                'ssim_local': metrics_sum_local['ssim'] / total_channels,
                'lpips_local': metrics_sum_local['lpips'] / total_channels,
                'loss': loss_sum / count,
                # 'fid': metrics_sum['fid'] # FID已经是一个整体值
            }
        
        self.netG.train()
        return metrics_avg, visuals_list, image_names
