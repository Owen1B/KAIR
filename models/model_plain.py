from collections import OrderedDict
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.optim import Adam

from models.select_network import define_G
from models.model_base import ModelBase
from models.loss import CharbonnierLoss, PoissonLLLoss
from models.loss_ssim import SSIMLoss

from utils.utils_model import test_mode
from utils.utils_regularizers import regularizer_orth, regularizer_clip


class ModelPlain(ModelBase):
    """Train with pixel loss"""
    def __init__(self, opt):
        super(ModelPlain, self).__init__(opt)
        # ------------------------------------
        # define network
        # ------------------------------------
        self.opt_train = self.opt['train']    # training option
        self.netG = define_G(opt)
        self.netG = self.model_to_device(self.netG)
        if self.opt_train['E_decay'] > 0:
            self.netE = define_G(opt).to(self.device).eval()
            
        # ------------------------------------
        # AMP support
        # ------------------------------------
        self.amp_enabled = self.opt_train.get('amp_enabled', False)
        if self.amp_enabled:
            print('AMP (Automatic Mixed Precision) is enabled.')
            self.scaler = torch.amp.GradScaler('cuda')
        else:
            self.scaler = None

    """
    # ----------------------------------------
    # Preparation before training with data
    # Save model during training
    # ----------------------------------------
    """

    # ----------------------------------------
    # initialize training
    # ----------------------------------------
    def init_train(self):
        self.load()                           # load model
        self.netG.train()                     # set training mode,for BN
        self.define_loss()                    # define loss
        self.define_optimizer()               # define optimizer
        self.load_optimizers()                # load optimizer
        self.define_scheduler()               # define scheduler
        self.load_scheduler_states()          # load scheduler states
        self.log_dict = OrderedDict()         # log

    # ----------------------------------------
    # load pre-trained G model
    # ----------------------------------------
    def load(self):
        load_path_G = self.opt['path']['pretrained_netG']
        if load_path_G is not None:
            print('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, strict=self.opt_train['G_param_strict'], param_key='params')
        load_path_E = self.opt['path']['pretrained_netE']
        if self.opt_train['E_decay'] > 0:
            if load_path_E is not None:
                print('Loading model for E [{:s}] ...'.format(load_path_E))
                self.load_network(load_path_E, self.netE, strict=self.opt_train['E_param_strict'], param_key='params_ema')
            else:
                print('Copying model for E ...')
                self.update_E(0)
            self.netE.eval()

    # ----------------------------------------
    # load optimizer
    # ----------------------------------------
    def load_optimizers(self):
        load_path_optimizerG = self.opt['path']['pretrained_optimizerG']
        if load_path_optimizerG is not None and self.opt_train['G_optimizer_reuse']:
            print('Loading optimizerG [{:s}] ...'.format(load_path_optimizerG))
            self.load_optimizer(load_path_optimizerG, self.G_optimizer)
            
        # Load AMP scaler if available
        if self.amp_enabled:
            load_path_scaler = self.opt['path'].get('pretrained_scaler')
            if load_path_scaler is not None:
                print('Loading AMP scaler [{:s}] ...'.format(load_path_scaler))
                self.load_scaler(load_path_scaler)

    # ----------------------------------------
    # load scheduler states
    # ----------------------------------------
    def load_scheduler_states(self):
        load_path_schedulerG = self.opt['path'].get('pretrained_schedulerG') # Use .get for safety
        if load_path_schedulerG is not None and self.schedulers:
            print('Loading schedulerG [{:s}] ...'.format(load_path_schedulerG))
            self.load_scheduler(load_path_schedulerG, self.schedulers[0])

    # ----------------------------------------
    # save model / optimizer(optional)
    # ----------------------------------------
    def save(self, iter_label):
        # 保存 netG 模型前先删除旧的检查点
        self._delete_old_checkpoints('G')
        self.save_network(self.save_dir, self.netG, 'G', iter_label)
        
        if self.opt_train['E_decay'] > 0:
            # 保存 netE 模型前先删除旧的检查点
            self._delete_old_checkpoints('E')
            self.save_network(self.save_dir, self.netE, 'E', iter_label)
            
        if self.opt_train['G_optimizer_reuse']:
            # 保存优化器前先删除旧的检查点
            self._delete_old_checkpoints('optimizerG')
            self.save_optimizer(self.save_dir, self.G_optimizer, 'optimizerG', iter_label)
            
        if self.schedulers: 
            # 保存调度器前先删除旧的检查点
            self._delete_old_checkpoints('schedulerG')
            self.save_scheduler(self.save_dir, self.schedulers[0], 'schedulerG', iter_label)
            
        # Save AMP scaler if enabled
        if self.amp_enabled and self.scaler is not None:
            self._delete_old_checkpoints('scaler')
            self.save_scaler(self.save_dir, iter_label)

    # ----------------------------------------
    # save AMP scaler
    # ----------------------------------------
    def save_scaler(self, save_dir, iter_label):
        import os
        save_filename = '{}_scaler.pth'.format(iter_label)
        save_path = os.path.join(save_dir, save_filename)
        torch.save(self.scaler.state_dict(), save_path)

    # ----------------------------------------
    # load AMP scaler
    # ----------------------------------------
    def load_scaler(self, load_path):
        scaler_state_dict = torch.load(load_path, map_location=lambda storage, loc: storage)
        self.scaler.load_state_dict(scaler_state_dict)
    
    # ----------------------------------------
    # 删除旧的检查点模型
    # ----------------------------------------
    def _delete_old_checkpoints(self, model_type):
        import os
        import re
        
        # 查找目录中所有与指定类型匹配的检查点文件
        checkpoint_files = []
        for filename in os.listdir(self.save_dir):
            if filename.endswith(f'_{model_type}.pth'):
                checkpoint_files.append(filename)
        
        # 如果找到了多个检查点，保留最新的，删除其余的
        if len(checkpoint_files) > 0:
            # 提取迭代数并排序
            iterations = []
            for filename in checkpoint_files:
                match = re.match(r'(\d+)_', filename)
                if match:
                    iterations.append((int(match.group(1)), filename))
            
            # 按迭代次数从大到小排序
            iterations.sort(reverse=True)
            
            # 保留最新的检查点，删除其他的
            for i in range(1, len(iterations)):
                iter_num, filename = iterations[i]

                filepath = os.path.join(self.save_dir, filename)
                os.remove(filepath)

    # ----------------------------------------
    # define loss
    # ----------------------------------------
    def define_loss(self):
        G_lossfn_type = self.opt_train['G_lossfn_type']
        if G_lossfn_type == 'l1':
            self.G_lossfn = nn.L1Loss().to(self.device)
        elif G_lossfn_type == 'l2':
            self.G_lossfn = nn.MSELoss().to(self.device)
        elif G_lossfn_type == 'l2sum':
            self.G_lossfn = nn.MSELoss(reduction='sum').to(self.device)
        elif G_lossfn_type == 'ssim':
            self.G_lossfn = SSIMLoss().to(self.device)
        elif G_lossfn_type == 'charbonnier':
            self.G_lossfn = CharbonnierLoss(self.opt_train['G_charbonnier_eps']).to(self.device)
        elif G_lossfn_type == 'poisson':
            # 从配置文件中读取归一化参数
            normalization_method = self.opt_train.get('G_poisson_normalization_method', 'linear')
            max_pixel = self.opt_train.get('G_poisson_max_pixel', 150.0)
            epsilon = self.opt_train.get('G_poisson_epsilon', 1e-9)
            self.G_lossfn = PoissonLLLoss(
                normalization_method=normalization_method,
                max_pixel=max_pixel,
                epsilon=epsilon
            ).to(self.device)
        else:
            raise NotImplementedError('Loss type [{:s}] is not found.'.format(G_lossfn_type))
        self.G_lossfn_weight = self.opt_train['G_lossfn_weight']

    # ----------------------------------------
    # define optimizer
    # ----------------------------------------
    def define_optimizer(self):
        G_optim_params = []
        for k, v in self.netG.named_parameters():
            if v.requires_grad:
                G_optim_params.append(v)
            else:
                print('Params [{:s}] will not optimize.'.format(k))
        if self.opt_train['G_optimizer_type'] == 'adam':
            self.G_optimizer = Adam(G_optim_params, lr=self.opt_train['G_optimizer_lr'],
                                    betas=self.opt_train['G_optimizer_betas'],
                                    weight_decay=self.opt_train['G_optimizer_wd'])
        else:
            raise NotImplementedError

    # ----------------------------------------
    # define scheduler, only "MultiStepLR"
    # ----------------------------------------
    def define_scheduler(self):
        if self.opt_train['G_scheduler_type'] == 'MultiStepLR':
            self.schedulers.append(lr_scheduler.MultiStepLR(self.G_optimizer,
                                                            self.opt_train['G_scheduler_milestones'],
                                                            self.opt_train['G_scheduler_gamma']
                                                            ))
        elif self.opt_train['G_scheduler_type'] == 'CosineAnnealingWarmRestarts':
            self.schedulers.append(lr_scheduler.CosineAnnealingWarmRestarts(self.G_optimizer,
                                                            self.opt_train['G_scheduler_periods'],
                                                            self.opt_train['G_scheduler_restart_weights'],
                                                            self.opt_train['G_scheduler_eta_min']
                                                            ))
        else:
            raise NotImplementedError

    """
    # ----------------------------------------
    # Optimization during training with data
    # Testing/evaluation
    # ----------------------------------------
    """

    # ----------------------------------------
    # feed L/H data
    # ----------------------------------------
    def feed_data(self, data, need_H=True):
        self.L = data['L'].to(self.device)
        if need_H:
            self.H = data['H'].to(self.device)

    # ----------------------------------------
    # feed L to netG
    # ----------------------------------------
    def netG_forward(self):
        if self.amp_enabled:
            with torch.amp.autocast('cuda'):
                self.E = self.netG(self.L)
        else:
            self.E = self.netG(self.L)

    # ----------------------------------------
    # update parameters and get loss
    # ----------------------------------------
    def optimize_parameters(self, current_step):
        self.G_optimizer.zero_grad()
        
        if self.amp_enabled:
            # AMP training
            with torch.amp.autocast('cuda'):
                self.netG_forward()
                G_loss = self.G_lossfn_weight * self.G_lossfn(self.E, self.H)
            
            # Scale loss and backward
            self.scaler.scale(G_loss).backward()
            
            # Gradient clipping with AMP
            G_optimizer_clipgrad = self.opt_train['G_optimizer_clipgrad'] if self.opt_train['G_optimizer_clipgrad'] else 0
            if G_optimizer_clipgrad > 0:
                self.scaler.unscale_(self.G_optimizer)
                torch.nn.utils.clip_grad_norm_(self.netG.parameters(), max_norm=self.opt_train['G_optimizer_clipgrad'], norm_type=2)
            
            # Optimizer step with AMP
            self.scaler.step(self.G_optimizer)
            self.scaler.update()
        else:
            # Standard training
            self.netG_forward()
            G_loss = self.G_lossfn_weight * self.G_lossfn(self.E, self.H)
            G_loss.backward()

            # Gradient clipping
            G_optimizer_clipgrad = self.opt_train['G_optimizer_clipgrad'] if self.opt_train['G_optimizer_clipgrad'] else 0
            if G_optimizer_clipgrad > 0:
                torch.nn.utils.clip_grad_norm_(self.netG.parameters(), max_norm=self.opt_train['G_optimizer_clipgrad'], norm_type=2)

            self.G_optimizer.step()

        # ------------------------------------
        # regularizer
        # ------------------------------------
        G_regularizer_orthstep = self.opt_train['G_regularizer_orthstep'] if self.opt_train['G_regularizer_orthstep'] else 0
        if G_regularizer_orthstep > 0 and current_step % G_regularizer_orthstep == 0 and current_step % self.opt['train']['checkpoint_save'] != 0:
            self.netG.apply(regularizer_orth)
        G_regularizer_clipstep = self.opt_train['G_regularizer_clipstep'] if self.opt_train['G_regularizer_clipstep'] else 0
        if G_regularizer_clipstep > 0 and current_step % G_regularizer_clipstep == 0 and current_step % self.opt['train']['checkpoint_save'] != 0:
            self.netG.apply(regularizer_clip)

        # self.log_dict['G_loss'] = G_loss.item()/self.E.size()[0]  # if `reduction='sum'`
        self.log_dict['G_loss'] = G_loss.item()

        if self.opt_train['E_decay'] > 0:
            self.update_E(self.opt_train['E_decay'])

    # ----------------------------------------
    # test / inference
    # ----------------------------------------
    def test(self):
        self.netG.eval()
        with torch.no_grad():
            if self.amp_enabled:
                with torch.amp.autocast('cuda'):
                    self.netG_forward()
            else:
                self.netG_forward()
        self.netG.train()

    # ----------------------------------------
    # test / inference x8
    # ----------------------------------------
    def testx8(self):
        self.netG.eval()
        with torch.no_grad():
            self.E = test_mode(self.netG, self.L, mode=3, sf=self.opt['scale'], modulo=1)
        self.netG.train()

    # ----------------------------------------
    # get log_dict
    # ----------------------------------------
    def current_log(self):
        return self.log_dict

    # ----------------------------------------
    # get L, E, H image
    # ----------------------------------------
    def current_visuals(self, need_H=True):
        out_dict = OrderedDict()
        out_dict['L'] = self.L.detach()[0].float().cpu()
        out_dict['E'] = self.E.detach()[0].float().cpu()
        if need_H:
            out_dict['H'] = self.H.detach()[0].float().cpu()
        return out_dict

    # ----------------------------------------
    # get L, E, H batch images
    # ----------------------------------------
    def current_results(self, need_H=True):
        out_dict = OrderedDict()
        out_dict['L'] = self.L.detach().float().cpu()
        out_dict['E'] = self.E.detach().float().cpu()
        if need_H:
            out_dict['H'] = self.H.detach().float().cpu()
        return out_dict

    """
    # ----------------------------------------
    # Information of netG
    # ----------------------------------------
    """

    # ----------------------------------------
    # print network
    # ----------------------------------------
    def print_network(self):
        msg = self.describe_network(self.netG)
        print(msg)

    # ----------------------------------------
    # print params
    # ----------------------------------------
    def print_params(self):
        msg = self.describe_params(self.netG)
        print(msg)

    # ----------------------------------------
    # network information
    # ----------------------------------------
    def info_network(self):
        msg = self.describe_network(self.netG)
        return msg

    # ----------------------------------------
    # params information
    # ----------------------------------------
    def info_params(self):
        msg = self.describe_params(self.netG)
        return msg
