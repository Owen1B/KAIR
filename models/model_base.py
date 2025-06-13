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
from tqdm import tqdm

class ModelBase():
    def __init__(self, opt):
        self.opt = opt                         # opt
        self.save_dir = opt['path']['models']  # save models
        self.device = torch.device('cuda' if opt['gpu_ids'] is not None else 'cpu')
        self.is_train = opt['is_train']        # training or not
        self.schedulers = []                   # schedulers

        # LPIPS VGG aAlexNet
        self.lpips_net_type = opt['train']['lpips_net']

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
    def evaluate_metrics(self, test_loader, add_poisson_noise=False, lpips_local_repeat_n=1):
        """评估模型性能,计算PSNR、SSIM、LPIPS指标,并保存可视化结果
        Args:
            test_loader: 测试数据加载器
            add_poisson_noise: 是否对模型输出E添加泊松噪声。
            lpips_local_repeat_n: 当add_poisson_noise=True时,为计算局部LPIPS指标重复采样的次数。
        Returns:
            metrics_avg: 包含平均指标的字典
            visuals_list: 所有测试样本的可视化图像列表
            image_names: 所有测试样本的文件名列表
        """
        self.netG.eval()
        
        # 收集所有图像数据
        all_imgs, image_names, loss_sum = self._collect_test_data(test_loader)
        
        # 计算全局归一化参数
        max_val_global = self._compute_global_max(all_imgs['H'])
        
        # 计算指标和生成可视化
        metrics_avg, visuals_list = self._compute_metrics_and_visuals(
            all_imgs, image_names, max_val_global, loss_sum, 
            add_poisson_noise, lpips_local_repeat_n
        )
        
        self.netG.train()
        return metrics_avg, visuals_list, image_names

    def _collect_test_data(self, test_loader):
        """收集测试数据"""
        all_imgs = {'L': [], 'E': [], 'H': []}
        image_names = []
        loss_sum = 0
        
        # 创建保存目录
        save_dir = os.path.join(self.opt['path']['images'])
        for key in ['L', 'E', 'H']:
            os.makedirs(os.path.join(save_dir, key), exist_ok=True)
        
        # 设置返回路径信息
        test_loader.dataset.return_paths = True
        
        # 遍历收集数据
        with torch.no_grad():
            for test_data in test_loader:
                self.feed_data(test_data)
                self.test()
                visuals = self.current_visuals()
                
                # 计算loss
                loss = self.G_lossfn_weight * self.G_lossfn(self.E, self.H)
                loss_sum += loss.item()
                
                # 提取文件名
                image_name_ext = os.path.basename(test_data['L_path'][0])
                image_name = os.path.splitext(image_name_ext)[0]
                image_names.append(image_name)
                
                # 反归一化并存储图像
                for key in ['L', 'E', 'H']:
                    img = visuals[key].cpu().float().numpy().transpose((1, 2, 0))
                    img_norm = utils_spect.denormalize_spect(
                        img, 
                        self.opt['datasets']['test']['normalization']['type'],
                        self.opt['datasets']['test']['normalization']['max_pixel']
                    )
                    all_imgs[key].append(img_norm)
        
        return all_imgs, image_names, loss_sum

    def _compute_global_max(self, H_imgs):
        """计算全局最大值用于归一化"""
        all_H_values = np.concatenate([img.flatten() for img in H_imgs])
        return np.max(all_H_values)

    def _compute_channel_metrics(self, E_img, H_img, max_val, ch):
        """计算单个通道的指标"""
        # Clip and normalize
        E_ch_clip = np.clip(E_img[:,:,ch], 0, max_val)
        H_ch_clip = np.clip(H_img[:,:,ch], 0, max_val)
        
        if max_val > 0:
            E_ch_255 = (E_ch_clip / max_val * 255).astype(np.uint8)
            H_ch_255 = (H_ch_clip / max_val * 255).astype(np.uint8)
        else:
            E_ch_255 = np.zeros_like(E_ch_clip, dtype=np.uint8)
            H_ch_255 = np.zeros_like(H_ch_clip, dtype=np.uint8)
        
        # Convert to RGB for metric calculation
        E_rgb = np.stack([E_ch_255] * 3, axis=2)
        H_rgb = np.stack([H_ch_255] * 3, axis=2)
        
        # Calculate metrics
        psnr = util.calculate_psnr(E_rgb, H_rgb)
        ssim = util.calculate_ssim(E_rgb, H_rgb)
        lpips = util.calculate_lpips(E_rgb, H_rgb,net=self.lpips_net_type)
        
        return psnr, ssim, lpips

    def _compute_metrics_for_image(self, E_img, H_img, max_val_global, max_val_local, 
                                   add_poisson_noise, lpips_local_repeat_n):
        """计算单张图像的所有指标"""
        num_channels = E_img.shape[2]
        
        # 初始化指标累加器
        metrics_global = {'psnr': 0, 'ssim': 0, 'lpips': 0}
        metrics_local = {'psnr': 0, 'ssim': 0, 'lpips': 0}
        
        # 计算全局和局部指标（PSNR, SSIM）
        for ch in range(num_channels):
            # 全局指标
            psnr_g, ssim_g, lpips_g = self._compute_channel_metrics(
                E_img, H_img, max_val_global, ch
            )
            metrics_global['psnr'] += psnr_g
            metrics_global['ssim'] += ssim_g
            metrics_global['lpips'] += lpips_g
            
            # 局部指标（PSNR, SSIM）
            psnr_l, ssim_l, _ = self._compute_channel_metrics(
                E_img, H_img, max_val_local, ch
            )
            metrics_local['psnr'] += psnr_l
            metrics_local['ssim'] += ssim_l
        
        # 局部LPIPS需要特殊处理
        if add_poisson_noise:
            # 重复采样计算局部LPIPS
            E_img_for_poisson = np.maximum(0, E_img)
            lpips_local_sum = 0.0
            
            for _ in range(lpips_local_repeat_n):
                E_img_sampled = np.random.poisson(E_img_for_poisson).astype(np.float32)
                for ch in range(num_channels):
                    _, _, lpips_l = self._compute_channel_metrics(
                        E_img_sampled, H_img, max_val_local, ch
                    )
                    lpips_local_sum += lpips_l
            
            metrics_local['lpips'] = lpips_local_sum / lpips_local_repeat_n
        else:
            # 直接计算局部LPIPS
            for ch in range(num_channels):
                _, _, lpips_l = self._compute_channel_metrics(
                    E_img, H_img, max_val_local, ch
                )
                metrics_local['lpips'] += lpips_l
        
        return metrics_global, metrics_local

    def _save_image_channels(self, img, key, image_name, max_val_global):
        """保存图像的所有通道"""
        save_dir = os.path.join(self.opt['path']['images'])
        
        # Clip and normalize
        img_clipped = np.clip(img, 0, max_val_global)
        if max_val_global > 0:
            img_255 = (img_clipped / max_val_global * 255).astype(np.uint8)
        else:
            img_255 = np.zeros_like(img_clipped, dtype=np.uint8)
        
        # Save each channel as RGB
        for ch in range(img.shape[2]):
            gray_img = img_255[:, :, ch]
            rgb_img = np.stack([gray_img, gray_img, gray_img], axis=2)
            rgb_path = os.path.join(save_dir, key, f"{image_name}_ch{ch}.png")
            util.imsave(rgb_img, rgb_path)

    def _create_visualization(self, L_img, E_img, H_img, max_val_local, add_poisson_noise, metrics_local_avg):
        """创建单张图像的可视化"""
        fig = plt.figure(figsize=(8, 20))
        gs = plt.GridSpec(2, 4, height_ratios=[1, 1], width_ratios=[1, 1, 1, 0.05])
        
        titles = {
            'L': 'Input (L)',
            'E': f"Estimated (E){'_poisson' if add_poisson_noise else ''}",
            'H': 'Ground Truth (H)'
        }
        
        sample_imgs = {'L': L_img, 'E': E_img, 'H': H_img}
        vmax_vis, vmin_vis = np.max(H_img), 0
        
        # 使用已经计算好的局部指标
        plt.suptitle(
            f'PSNR(local): {metrics_local_avg["psnr"]:.2f}dB, '
            f'SSIM(local): {metrics_local_avg["ssim"]:.4f}, '
            f'LPIPS(local): {metrics_local_avg["lpips"]:.4f}', 
            fontsize=16
        )

        # 绘制图像
        for row, view in enumerate(['Anterior', 'Posterior']):
            for col, (key, title) in enumerate(titles.items()):
                ax = plt.subplot(gs[row, col])
                im = ax.imshow(sample_imgs[key][:,:,row], cmap='gray', 
                             vmin=vmin_vis, vmax=vmax_vis)
                ax.set_title(f'{title} - {view}')
                ax.axis('off')
        
        # 添加colorbar
        cax = plt.subplot(gs[:, 3])
        plt.colorbar(im, cax=cax)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # 转换为数组
        fig.canvas.draw()
        img_array = np.array(fig.canvas.renderer.buffer_rgba())
        plt.close(fig)
        
        return img_array

    def _compute_metrics_and_visuals(self, all_imgs, image_names, max_val_global, 
                                   loss_sum, add_poisson_noise, lpips_local_repeat_n):
        """计算所有指标和生成可视化"""
        # 初始化累加器
        metrics_sum_global = {'psnr': 0, 'ssim': 0, 'lpips': 0}
        metrics_sum_local = {'psnr': 0, 'ssim': 0, 'lpips': 0}
        poisson_ll_sum = 0
        count = 0
        visuals_list = []
        
        for idx, (L_img, E_img_clean, H_img) in enumerate(
            zip(all_imgs['L'], all_imgs['E'], all_imgs['H'])
        ):
            count += 1
            image_name = image_names[idx]
            max_val_local = np.max(H_img)
            
            # 处理泊松噪声
            if add_poisson_noise:
                E_img_for_poisson = np.maximum(0, E_img_clean)
                E_img = np.random.poisson(E_img_for_poisson).astype(np.float32)
            else:
                E_img = E_img_clean
                poisson_ll_sum += util.pll(E_img, H_img)
            
            # 计算指标
            metrics_global, metrics_local = self._compute_metrics_for_image(
                E_img, H_img, max_val_global, max_val_local, 
                add_poisson_noise, lpips_local_repeat_n
            )
            
            # 累加指标
            for key in metrics_sum_global:
                metrics_sum_global[key] += metrics_global[key]
                metrics_sum_local[key] += metrics_local[key]
            
            # 保存图像
            for key, img in zip(['L', 'E', 'H'], [L_img, E_img, H_img]):
                self._save_image_channels(img, key, image_name, max_val_global)
            
            # 计算当前图像的平均局部指标用于显示
            num_channels = E_img.shape[2]
            metrics_local_avg = {
                'psnr': metrics_local['psnr'] / num_channels if num_channels > 0 else 0,
                'ssim': metrics_local['ssim'] / num_channels if num_channels > 0 else 0,
                'lpips': metrics_local['lpips'] / num_channels if num_channels > 0 else 0
            }
            
            # 创建可视化
            visual = self._create_visualization(
                L_img, E_img, H_img, max_val_local, add_poisson_noise, metrics_local_avg
            )
            visuals_list.append(visual)
        
        # 计算平均值
        total_channels = count * L_img.shape[2] if count > 0 else 0
        
        if total_channels == 0:
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
            }
        
        if not add_poisson_noise:
            metrics_avg['poisson_ll'] = poisson_ll_sum / count if count > 0 else 0
        
        return metrics_avg, visuals_list
