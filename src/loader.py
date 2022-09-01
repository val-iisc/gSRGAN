# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# src/loader.py

import pickle
import json
import glob
import os
import warnings
import random
from os.path import dirname, abspath, exists, join
import matplotlib.pyplot as plt
import pandas as pd
# from torchlars import LARS

from data_utils.load_dataset import *
from metrics.inception_network import InceptionV3
from metrics.resnet50 import resnet50
from metrics.prepare_inception_moments import prepare_inception_moments
from utils.log import make_checkpoint_dir, make_logger
from utils.losses import *
from utils.load_checkpoint import load_checkpoint
from utils.misc import *
from utils.biggan_utils import ema, ema_DP_SyncBN
from sync_batchnorm.batchnorm import convert_model
from worker import make_worker
import torchvision.transforms as transforms

import torch
from torch.utils.data import DataLoader
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from utils.sample import BalancedBatchSampler

def plot_sn_curves(gain_sn, name, run_name):
    
    directory = join('./figures', run_name)
    if not exists(abspath(directory)):
        os.makedirs(directory)

    save_path = join(directory, f'LT_{name}.png')

    gain_lst = []
    for i in [1,2,3,4,5,6]:
        gain = pd.DataFrame(gain_sn[:,i-1,:])
        gain = gain[(gain > 0).all(axis=1)]
        columns = gain.columns
        gain_lst.append(gain)

    fig, ax = plt.subplots(2,3, figsize=(10,10))
    ax[0][0].plot(gain_lst[0], label=columns)
    ax[0][0].set_title(f'SN_{name}_layer1')
    ax[0][0].legend()
    ax[0][1].plot(gain_lst[1], label=columns)
    ax[0][1].set_title(f'SN_{name}_layer2')
    ax[0][1].legend()
    ax[0][2].plot(gain_lst[2], label=columns)
    ax[0][2].set_title(f'SN_{name}_layer3')
    ax[0][2].legend()
    ax[1][0].plot(gain_lst[3], label=columns)
    ax[1][0].set_title(f'SN_{name}_layer4')
    ax[1][0].legend()
    ax[1][1].plot(gain_lst[4], label=columns)
    ax[1][1].set_title(f'SN_{name}_layer5')
    ax[1][1].legend()
    ax[1][2].plot(gain_lst[5], label=columns)
    ax[1][2].set_title(f'SN_{name}_layer6')
    ax[1][2].legend()
    plt.savefig(save_path)
    plt.close()


class LoadEvalModel(object):
    def __init__(self, eval_backbone, world_size, distributed_data_parallel, device):
        super(LoadEvalModel, self).__init__()
        self.eval_backbone = eval_backbone
        self.save_output = SaveOutput()

        if self.eval_backbone == "Inception_V3":
            self.model = InceptionV3().to(device)
        elif self.eval_backbone == "SwAV":
            self.model = torch.hub.load('facebookresearch/swav:main', 'resnet50').to(device)
            #self.model = resnet50().to(device)
            hook_handles = []
            for name, layer in self.model.named_children():
                if name == "fc":
                    handle = layer.register_forward_pre_hook(self.save_output)
                    hook_handles.append(handle)
        elif self.eval_backbone == "VGG16":
            detector_url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
            self.model = torch.hub.load_state_dict_from_url(detector_url).to(device)
        else:
            raise NotImplementedError

        if world_size > 1 and distributed_data_parallel:
            toggle_grad(self.model, on=True)
            self.model = DDP(self.model, device_ids=[device], broadcast_buffers=True)
        elif world_size > 1 and distributed_data_parallel is False:
            self.model = DataParallel(self.model, output_device=device)
        else:
            pass

    def eval(self):
        self.model.eval()

    def get_outputs(self, x):
        if self.eval_backbone == "Inception_V3":
            repres, logits = self.model(x)
        elif self.eval_backbone == "VGG16":
            repres = self.model(x, return_features=True)
            logits = torch.zeros((repres.size(0), 1000), dtype=repres.dtype).to(repres.device)
        else:
            logits = self.model(x)
            #logits = torch.zeros((repres.size(0), 1000), dtype=repres.dtype).to(repres.device)
            repres = self.save_output.outputs[0][0]
            self.save_output.clear()
        return repres, logits

    def __call__(self, x):
        return self.get_outputs(x)

def plot_moving_avg(alphas_r, alphas_f, run_name):

    directory = join('./figures', run_name)
    if not exists(abspath(directory)):
        os.makedirs(directory)

    save_path = join(directory, f'Moving_avg_plots.png')
    fig, ax = plt.subplots(2,5, figsize=(10,10))
    for i in range(2):
        for j in range(5):
            ax[i][j].plot(alphas_r[i*5+j], label=f'Real Class')
            ax[i][j].plot(alphas_f[i*5+j], label=f'Fake Class')
            ax[i][j].set_title(f'Class: {5*i+j+1}')
            ax[i][j].legend()
    
    plt.savefig(save_path)
    plt.close()


def prepare_train_eval(local_rank, gpus_per_node, world_size, run_name, train_configs, model_configs, hdf5_path_train):
    cfgs = dict2clsattr(train_configs, model_configs)
    prev_ada_p, step, best_step, best_fid, best_fid_checkpoint_path, mu, sigma, inception_model = None, 0, 0, None, None, None, None, None

    if cfgs.distributed_data_parallel:
        global_rank = cfgs.nr*(gpus_per_node) + local_rank
        print("Use GPU: {} for training.".format(global_rank))
        setup(global_rank, world_size)
        torch.cuda.set_device(local_rank)
    else:
        global_rank = local_rank

    writer = SummaryWriter(log_dir=join('./logs', run_name)) if local_rank == 0 else None
    if local_rank == 0:
        logger = make_logger(run_name, None)
        logger.info('Run name : {run_name}'.format(run_name=run_name))
        logger.info(json.dumps(train_configs, indent=2))
        logger.info(json.dumps(model_configs, indent=2))
    else:
        logger = None

    ##### load dataset #####
    if local_rank == 0: logger.info('Load train datasets...')

    if cfgs.imbalanced_cifar10:
        print('''######################
                YOU ARE USING IMBALANCED CIFAR-10 DATASET
                ##########################''')

        transform = [transforms.ToTensor(),
                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                         transforms.Resize(cfgs.img_size)]

        if cfgs.random_flip_preprocessing:
            transform += [transforms.RandomHorizontalFlip()]

        transform = transforms.Compose(transform)

        train_dataset = IMBALANCECIFAR10(cfgs.dataset_name, root='./data', train=True, imb_factor=cfgs.imb_factor,
                        download=True, transform=transform)
    
    elif cfgs.imbalanced_cifar100:
        print('''######################
                LONG TAIL CIFAR-100
                ##########################''')

        transform = [transforms.ToTensor(),
                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                         transforms.Resize(cfgs.img_size)]

        if cfgs.random_flip_preprocessing:
            transform += [transforms.RandomHorizontalFlip()]

        transform = transforms.Compose(transform)

        train_dataset = IMBALANCECIFAR100(cfgs.dataset_name, root='./data', train=True, imb_factor=cfgs.imb_factor,
                        download=True, transform=transform)

    elif cfgs.imbalanced_lsun:
        print('''######################
                LONG TAIL LSUN
                ##########################''')


        train_dataset = LoadDataset(cfgs.dataset_name, cfgs.data_path, train=True, download=True, resize_size=cfgs.img_size,
                                    hdf5_path=hdf5_path_train, random_flip=cfgs.random_flip_preprocessing, cfgs=cfgs)
        
        if hdf5_path_train == None:
        
            lsun_classes = ["bedroom_train", "conference_room_train", "dining_room_train", "kitchen_train", "living_room_train"]
            
            tmp_obj = IMBALANCELSUN(root=cfgs.data_path, classes=lsun_classes, imb_factor=cfgs.imb_factor, max_samples = 50000)
            train_dataset.img_num_list = tmp_obj.img_num_list

        
        print(train_dataset.img_num_list)

    else:
        train_dataset = LoadDataset(cfgs.dataset_name, cfgs.data_path, train=True, download=True, resize_size=cfgs.img_size,
                                    hdf5_path=hdf5_path_train, random_flip=cfgs.random_flip_preprocessing, cfgs=cfgs)

        
    if cfgs.reduce_train_dataset < 1.0:
        num_train = int(cfgs.reduce_train_dataset*len(train_dataset))
        train_dataset, _ = torch.utils.data.random_split(train_dataset, [num_train, len(train_dataset) - num_train])
    if local_rank == 0: logger.info('Train dataset size : {dataset_size}'.format(dataset_size=len(train_dataset)))

    if local_rank == 0: logger.info('Load {mode} datasets...'.format(mode=cfgs.eval_type))
    eval_mode = True if cfgs.eval_type == 'train' else False


    if cfgs.imbalanced_cifar10:
        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        eval_dataset = IMBALANCECIFAR10(cfgs.dataset_name, root='./data', train=False, imb_factor=1.0,
                        download=True, transform=transform)
    elif cfgs.imbalanced_cifar100:
        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        eval_dataset = IMBALANCECIFAR100(cfgs.dataset_name, root='./data', train=False, imb_factor=1.0,
                        download=True, transform=transform)
    
    elif cfgs.imbalanced_lsun:
        transform = [transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    transforms.Resize((cfgs.img_size, cfgs.img_size))]
        lsun_classes = [ "bedroom_train", "conference_room_train", "dining_room_train", "kitchen_train", "living_room_train"]
        if hdf5_path_train == None:
            
            eval_dataset = LoadDataset(cfgs.dataset_name, cfgs.data_path, train=False, download=True, resize_size=cfgs.img_size,
                                     hdf5_path=hdf5_path_train, random_flip=cfgs.random_flip_preprocessing, cfgs=cfgs)
        else:    
            eval_dataset = LoadDataset(cfgs.dataset_name, cfgs.data_path, train=False, download=True, resize_size=cfgs.img_size,
                                     hdf5_path="./data/lsun_128_val.hdf5", random_flip=cfgs.random_flip_preprocessing, cfgs=cfgs)

    else:                   
        if eval_mode == True: 
            eval_dataset = LoadDataset(cfgs.dataset_name, cfgs.data_path, train=eval_mode, download=True, resize_size=cfgs.img_size,
                                hdf5_path=hdf5_path_train, random_flip=False, cfgs=cfgs)
        else:
            # For iNat19
            if cfgs.dataset_name == "inaturalist2019" and hdf5_path_train != None:
                eval_dataset = LoadDataset(cfgs.dataset_name, cfgs.data_path, train=eval_mode, download=True, resize_size=cfgs.img_size,
                                hdf5_path="./inaturalist2019_64_val.hdf5", random_flip=False, cfgs=cfgs)
            elif cfgs.dataset_name == "inaturalist2017" and hdf5_path_train != None:
                eval_dataset = LoadDataset(cfgs.dataset_name, cfgs.data_path, train=eval_mode, download=True, resize_size=cfgs.img_size,
                                hdf5_path="./inaturalist2017_64_val.hdf5", random_flip=False, cfgs=cfgs)

            else:
                eval_dataset = LoadDataset(cfgs.dataset_name, cfgs.data_path, train=eval_mode, download=True, resize_size=cfgs.img_size,
                                hdf5_path=None, random_flip=False, cfgs=cfgs)
    
    if local_rank == 0: logger.info('Eval dataset size : {dataset_size}'.format(dataset_size=len(eval_dataset)))

    if cfgs.distributed_data_parallel:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        cfgs.batch_size = cfgs.batch_size//world_size
    else:
        train_sampler = None

    train_dataloader = DataLoader(train_dataset, batch_size=cfgs.batch_size, shuffle=(train_sampler is None), pin_memory=True,
                                  num_workers=cfgs.num_workers, sampler=train_sampler, drop_last=True)

    if cfgs.dataset_name == "inaturalist2019":
        # Sample a balanced batch for evaluation
        eval_dataloader = DataLoader(eval_dataset, batch_size=cfgs.batch_size, shuffle=False, pin_memory=True, num_workers=cfgs.num_workers, drop_last=True, sampler=BalancedBatchSampler(eval_dataset, torch.Tensor(eval_dataset.labels) ))
    else:
        eval_dataloader = DataLoader(eval_dataset, batch_size=cfgs.batch_size, shuffle=False, pin_memory=True, num_workers=cfgs.num_workers, drop_last=False)
    ##### build model #####
    if local_rank == 0: logger.info('Build model...')
    module = __import__('models.{architecture}'.format(architecture=cfgs.architecture), fromlist=['something'])
    if local_rank == 0: logger.info('Modules are located on models.{architecture}.'.format(architecture=cfgs.architecture))
    Gen = module.Generator(cfgs.z_dim, cfgs.shared_dim, cfgs.img_size, cfgs.g_conv_dim, cfgs.g_spectral_norm, cfgs.attention,
                           cfgs.attention_after_nth_gen_block, cfgs.activation_fn, cfgs.conditional_strategy, cfgs.num_classes,
                           cfgs.g_init, cfgs.G_depth, cfgs.mixed_precision, cfgs.sn_batchnorm).to(local_rank)

    Dis = module.Discriminator(cfgs.img_size, cfgs.d_conv_dim, cfgs.d_spectral_norm, cfgs.attention, cfgs.attention_after_nth_dis_block,
                               cfgs.activation_fn, cfgs.conditional_strategy, cfgs.hypersphere_dim, cfgs.num_classes, cfgs.nonlinear_embed,
                               cfgs.normalize_embed, cfgs.d_init, cfgs.D_depth, cfgs.mixed_precision, cfgs.shared_dim).to(local_rank)

    if cfgs.aux_disc != "":
        if cfgs.optimizer == "SGD":
            D_aux_optimizer  = torch.optim.SGD(filter(lambda p: p.requires_grad, D_aux.parameters()), cfgs.d_lr, momentum=cfgs.momentum, nesterov=cfgs.nesterov)
        elif cfgs.optimizer == "RMSprop":
            D_aux_optimizer  = torch.optim.RMSprop(filter(lambda p: p.requires_grad, D_aux.parameters()), cfgs.d_lr, momentum=cfgs.momentum, alpha=cfgs.alpha)
        elif cfgs.optimizer == "Adam":
            D_aux_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, D_aux.parameters()), cfgs.d_lr, [cfgs.beta1, cfgs.beta2], eps=1e-6)
        else:
            raise NotImplementedError
    else:
        D_aux, d_aux_feat, D_aux_optimizer = None, None, None
    

    if cfgs.ema:
        if local_rank == 0: logger.info('Prepare EMA for G with decay of {}.'.format(cfgs.ema_decay))
        Gen_copy = module.Generator(cfgs.z_dim, cfgs.shared_dim, cfgs.img_size, cfgs.g_conv_dim, cfgs.g_spectral_norm, cfgs.attention,
                                    cfgs.attention_after_nth_gen_block, cfgs.activation_fn, cfgs.conditional_strategy, cfgs.num_classes,
                                    initialize=False, G_depth=cfgs.G_depth, mixed_precision=cfgs.mixed_precision).to(local_rank)
        if not cfgs.distributed_data_parallel and world_size > 1 and cfgs.synchronized_bn:
            Gen_ema = ema_DP_SyncBN(Gen, Gen_copy, cfgs.ema_decay, cfgs.ema_start)
        else:
            Gen_ema = ema(Gen, Gen_copy, cfgs.ema_decay, cfgs.ema_start)
    else:
        Gen_copy, Gen_ema = None, None

    if local_rank == 0: logger.info(count_parameters(Gen))
    if local_rank == 0: logger.info(Gen)

    if local_rank == 0: logger.info(count_parameters(Dis))
    if local_rank == 0: logger.info(Dis)


    ### define loss functions and optimizers
    G_loss = {'vanilla': loss_dcgan_gen, 'least_square': loss_lsgan_gen, 'hinge': loss_hinge_gen, 'wasserstein': loss_wgan_gen}
    D_loss = {'vanilla': loss_dcgan_dis, 'least_square': loss_lsgan_dis, 'hinge': loss_hinge_dis, 'wasserstein': loss_wgan_dis}

    if cfgs.optimizer == "SGD":
        G_optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, Gen.parameters()), cfgs.g_lr, momentum=cfgs.momentum, nesterov=cfgs.nesterov)
        D_optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, Dis.parameters()), cfgs.d_lr, momentum=cfgs.momentum, nesterov=cfgs.nesterov)
    elif cfgs.optimizer == "RMSprop":
        G_optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, Gen.parameters()), cfgs.g_lr, momentum=cfgs.momentum, alpha=cfgs.alpha)
        D_optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, Dis.parameters()), cfgs.d_lr, momentum=cfgs.momentum, alpha=cfgs.alpha)
    elif cfgs.optimizer == "Adam":
        G_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, Gen.parameters()), cfgs.g_lr, [cfgs.beta1, cfgs.beta2], eps=1e-6)
        D_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, Dis.parameters()), cfgs.d_lr, [cfgs.beta1, cfgs.beta2], eps=1e-6)
    else:
        raise NotImplementedError

    if cfgs.LARS_optimizer:
        G_optimizer = LARS(optimizer=G_optimizer, eps=1e-8, trust_coef=0.001)
        D_optimizer = LARS(optimizer=D_optimizer, eps=1e-8, trust_coef=0.001)

    ##### load checkpoints if needed #####
    if cfgs.checkpoint_folder is None:
        checkpoint_dir = make_checkpoint_dir(cfgs.checkpoint_folder, run_name)
    else:

        when = "current" if cfgs.load_current is True else "best"
        #import pdb; pdb.set_trace()
        if not exists(abspath(cfgs.checkpoint_folder)):
            raise NotADirectoryError
        checkpoint_dir = make_checkpoint_dir(cfgs.checkpoint_folder, run_name)
        g_checkpoint_dir = glob.glob(join(checkpoint_dir,"model=G-{when}-weights-step*.pth".format(when=when)))[0]
        d_checkpoint_dir = glob.glob(join(checkpoint_dir,"model=D-{when}-weights-step*.pth".format(when=when)))[0]
        Gen, G_optimizer, trained_seed, run_name, step, prev_ada_p = load_checkpoint(Gen, G_optimizer, g_checkpoint_dir)
        Dis, D_optimizer, trained_seed, run_name, step, prev_ada_p, best_step, best_fid, best_fid_checkpoint_path =\
            load_checkpoint(Dis, D_optimizer, d_checkpoint_dir, metric=True)
        
        if local_rank == 0: logger = make_logger(run_name, None)
        if cfgs.ema:
            g_ema_checkpoint_dir = glob.glob(join(checkpoint_dir, "model=G_ema-{when}-weights-step*.pth".format(when=when)))[0]
            Gen_copy = load_checkpoint(Gen_copy, None, g_ema_checkpoint_dir, ema=True)
            Gen_ema.source, Gen_ema.target = Gen, Gen_copy

        writer = SummaryWriter(log_dir=join('./logs', run_name)) if global_rank == 0 else None
        if cfgs.train_configs['train'] and cfgs.seed != trained_seed:
            cfgs.seed = trained_seed
            fix_all_seed(cfgs.seed) 

        if local_rank == 0: logger.info('Generator checkpoint is {}'.format(g_checkpoint_dir))
        if local_rank == 0: logger.info('Discriminator checkpoint is {}'.format(d_checkpoint_dir))
        if cfgs.freeze_layers > -1 :
            prev_ada_p, step, best_step, best_fid, best_fid_checkpoint_path = None, 0, 0, None, None


    ##### wrap models with DP and convert BN to Sync BN #####
    if world_size > 1:
        if cfgs.distributed_data_parallel:
            if cfgs.synchronized_bn:
                process_group = torch.distributed.new_group([w for w in range(world_size)])
                Gen = torch.nn.SyncBatchNorm.convert_sync_batchnorm(Gen, process_group)
                Dis = torch.nn.SyncBatchNorm.convert_sync_batchnorm(Dis, process_group)
                if cfgs.ema:
                    Gen_copy = torch.nn.SyncBatchNorm.convert_sync_batchnorm(Gen_copy, process_group)

            Gen = DDP(Gen, device_ids=[local_rank])
            Dis = DDP(Dis, device_ids=[local_rank])
            if cfgs.ema:
                Gen_copy = DDP(Gen_copy, device_ids=[local_rank])
        else:
            Gen = DataParallel(Gen, output_device=local_rank)
            Dis = DataParallel(Dis, output_device=local_rank)
            if cfgs.ema:
                Gen_copy = DataParallel(Gen_copy, output_device=local_rank)

            if cfgs.synchronized_bn:
                Gen = convert_model(Gen).to(local_rank)
                Dis = convert_model(Dis).to(local_rank)
                if cfgs.ema:
                    Gen_copy = convert_model(Gen_copy).to(local_rank)

    ##### load the inception network and prepare first/secend moments for calculating FID #####
    if cfgs.eval:
        inception_model = LoadEvalModel(cfgs.eval_backbone, world_size,\
                                        cfgs.distributed_data_parallel, local_rank)
        
        mu, sigma = prepare_inception_moments(dataloader=eval_dataloader,
                                              generator=Gen,
                                              eval_mode=cfgs.eval_type,
                                              inception_model=inception_model,
                                              splits=1,
                                              run_name=run_name,
                                              logger=logger,
                                              device=local_rank)

    worker = make_worker(
        cfgs=cfgs,
        run_name=run_name,
        best_step=best_step,
        logger=logger,
        writer=writer,
        n_gpus=world_size,
        gen_model=Gen,
        dis_model=Dis,
        inception_model=inception_model,
        Gen_copy=Gen_copy,
        Gen_ema=Gen_ema,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        G_optimizer=G_optimizer,
        D_optimizer=D_optimizer,
        G_loss=G_loss[cfgs.adv_loss],
        D_loss=D_loss[cfgs.adv_loss],
        prev_ada_p=prev_ada_p,
        global_rank=global_rank,
        local_rank=local_rank,
        bn_stat_OnTheFly=cfgs.bn_stat_OnTheFly,
        checkpoint_dir=checkpoint_dir,
        mu=mu,
        sigma=sigma,
        best_fid=best_fid,
        best_fid_checkpoint_path=best_fid_checkpoint_path,
        # Adding for the baseline (Ensembling off the shelf models)
        d_aux = D_aux,
        d_aux_feat = d_aux_feat,
        d_aux_optimizer = D_aux_optimizer,
        vgg_model = None
    )

    if cfgs.train_configs['train']:
        step = worker.train(current_step=step, total_step=cfgs.total_step)


    if cfgs.eval:
        
        is_save = worker.evaluation(step=step, standing_statistics=cfgs.standing_statistics, standing_step=cfgs.standing_step)

    if cfgs.save_images:
        worker.save_images(is_generate=True, png=True, npz=True, standing_statistics=cfgs.standing_statistics, standing_step=cfgs.standing_step)

    if cfgs.image_visualization:
        worker.run_image_visualization_lt(ncol=cfgs.ncol, standing_statistics=cfgs.standing_statistics, standing_step=cfgs.standing_step, step=best_step)

    if cfgs.k_nearest_neighbor:
        worker.run_nearest_neighbor(nrow=cfgs.nrow, ncol=cfgs.ncol, standing_statistics=cfgs.standing_statistics, standing_step=cfgs.standing_step)

    if cfgs.interpolation:
        worker.run_linear_interpolation(nrow=cfgs.nrow, ncol=cfgs.ncol, fix_z=True, fix_y=False,
                                        standing_statistics=cfgs.standing_statistics, standing_step=cfgs.standing_step)
        worker.run_linear_interpolation(nrow=cfgs.nrow, ncol=cfgs.ncol, fix_z=False, fix_y=True,
                                        standing_statistics=cfgs.standing_statistics, standing_step=cfgs.standing_step)

    if cfgs.frequency_analysis:
        worker.run_frequency_analysis(num_images=len(train_dataset),
                                      standing_statistics=cfgs.standing_statistics, standing_step=cfgs.standing_step)

    if cfgs.tsne_analysis:
        worker.run_tsne(dataloader=eval_dataloader,
                        standing_statistics=cfgs.standing_statistics, standing_step=cfgs.standing_step, cfgs=cfgs)
    
    if cfgs.tsne_analysis_lt:
        worker.run_tsne_lt(dataloader=train_dataloader,
                        standing_statistics=cfgs.standing_statistics, standing_step=cfgs.standing_step)
    
    if cfgs.corr_mat:
        worker.run_correlation_matrix(standing_statistics=cfgs.standing_statistics)
    if cfgs.attention_maps:
        worker.run_attention_maps_visualize(ncol=cfgs.ncol, standing_statistics=cfgs.standing_statistics, standing_step=cfgs.standing_step, step=best_step)
