# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# src/worker.py

import numpy as np
import sys
import glob
import random
from scipy import ndimage
from sklearn.manifold import TSNE
from os.path import join
from PIL import Image
from tqdm import tqdm
from datetime import datetime
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

from metrics.IS import calculate_incep_score
from metrics.FID import calculate_fid_score, get_bottom
from metrics.features import generate_images_and_stack_features
from metrics.prdc import calculate_pr_dc
from metrics.IntraFID import calculate_intra_fid_score
from metrics.F_beta import calculate_f_beta_score
from metrics.Accuracy import calculate_accuracy


from utils.biggan_utils import interp
from utils.sample import sample_latents, sample_1hot, make_mask, target_class_sampler
from utils.misc import *
from utils.losses import calc_derv4gp, calc_derv4dra, calc_derv, latent_optimise, set_temperature
from utils.losses import Conditional_Contrastive_loss, Proxy_NCA_loss, NT_Xent_loss
from utils.diff_aug import DiffAugment
from utils.cr_diff_aug import CR_DiffAug
from utils.model_ops import ConditionalBatchNorm2d, ConditionalBatchNorm2d_for_skip_and_shared

import torch
import torch.nn as nn
from torch.nn import DataParallel
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import torch.nn.functional as F
import torchvision
from torchvision import transforms



SAVE_FORMAT = 'step={step:0>3}-Inception_mean={Inception_mean:<.4}-Inception_std={Inception_std:<.4}-FID={FID:<.5}.pth'

LOG_FORMAT = (
    "Step: {step:>7} "
    "Progress: {progress:<.1%} "
    "Elapsed: {elapsed} "
    "SN_Regularizer_Loss: {SN_Regularizer_Loss:<.6} "
    "SN_Regularizer_Loss_Dis: {SN_Regularizer_Loss_Dis:<.6} "
    "Fro_Regularizer_Loss: {Fro_Regularizer_Loss:<.6}"
    "Discriminator_loss: {dis_loss:<.6} "
    "Generator_loss: {gen_loss:<.6} "
)


class make_worker(object):
    def __init__(self, cfgs, run_name, best_step, logger, writer, n_gpus, gen_model, dis_model, inception_model, Gen_copy,
                 Gen_ema, train_dataset, eval_dataset, train_dataloader, eval_dataloader, G_optimizer, D_optimizer, G_loss,
                 D_loss, prev_ada_p, global_rank, local_rank, bn_stat_OnTheFly, checkpoint_dir, mu, sigma, best_fid,
                 best_fid_checkpoint_path, d_aux, d_aux_feat, d_aux_optimizer, vgg_model=None):
        

        self.cfgs = cfgs
        self.run_name = run_name
        self.best_step = best_step
        self.seed = cfgs.seed
        self.dataset_name = cfgs.dataset_name
        self.eval_type = cfgs.eval_type
        self.logger = logger
        self.writer = writer
        self.num_workers = cfgs.num_workers
        self.n_gpus = n_gpus

        self.sn_regularize = cfgs.sn_regularize # added
        
        self.fro_regularize = cfgs.fro_regularize # added
        
        self.lam = [1] * cfgs.num_classes
        self.gen_model = gen_model
        self.dis_model = dis_model
        self.inception_model = inception_model
        self.pr_model = inception_model if vgg_model is None else vgg_model
        self.Gen_copy = Gen_copy
        self.Gen_ema = Gen_ema

        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader

        self.freeze_layers = cfgs.freeze_layers

        self.conditional_strategy = cfgs.conditional_strategy
        self.pos_collected_numerator = cfgs.pos_collected_numerator
        self.z_dim = cfgs.z_dim
        self.num_classes = cfgs.num_classes
        self.hypersphere_dim = cfgs.hypersphere_dim
        self.d_spectral_norm = cfgs.d_spectral_norm
        self.g_spectral_norm = cfgs.g_spectral_norm

        self.G_optimizer = G_optimizer
        self.D_optimizer = D_optimizer
        self.batch_size = cfgs.batch_size
        self.g_steps_per_iter = cfgs.g_steps_per_iter
        self.d_steps_per_iter = cfgs.d_steps_per_iter
        self.accumulation_steps = cfgs.accumulation_steps
        self.total_step = cfgs.total_step

        self.G_loss = G_loss
        self.D_loss = D_loss
        self.contrastive_lambda = cfgs.contrastive_lambda
        self.margin = cfgs.margin
        self.tempering_type = cfgs.tempering_type
        self.tempering_step = cfgs.tempering_step
        self.start_temperature = cfgs.start_temperature
        self.end_temperature = cfgs.end_temperature
        self.weight_clipping_for_dis = cfgs.weight_clipping_for_dis
        self.weight_clipping_bound = cfgs.weight_clipping_bound
        self.gradient_penalty_for_dis = cfgs.gradient_penalty_for_dis
        self.gradient_penalty_lambda = cfgs.gradient_penalty_lambda
        self.deep_regret_analysis_for_dis = cfgs.deep_regret_analysis_for_dis
        self.regret_penalty_lambda = cfgs.regret_penalty_lambda
        self.cr = cfgs.cr
        self.cr_lambda = cfgs.cr_lambda
        self.bcr = cfgs.bcr
        self.real_lambda = cfgs.real_lambda
        self.fake_lambda = cfgs.fake_lambda
        self.zcr = cfgs.zcr
        self.gen_lambda = cfgs.gen_lambda
        self.dis_lambda = cfgs.dis_lambda
        self.sigma_noise = cfgs.sigma_noise

        self.diff_aug = cfgs.diff_aug
        self.ada = cfgs.ada
        self.prev_ada_p = prev_ada_p
        self.ada_target = cfgs.ada_target
        self.ada_length = cfgs.ada_length
        self.prior = cfgs.prior
        self.truncated_factor = cfgs.truncated_factor
        self.ema = cfgs.ema
        self.latent_op = cfgs.latent_op
        self.latent_op_rate = cfgs.latent_op_rate
        self.latent_op_step = cfgs.latent_op_step
        self.latent_op_step4eval = cfgs.latent_op_step4eval
        self.latent_op_alpha = cfgs.latent_op_alpha
        self.latent_op_beta = cfgs.latent_op_beta
        self.latent_norm_reg_weight = cfgs.latent_norm_reg_weight

        self.global_rank = global_rank
        self.local_rank = local_rank
        self.bn_stat_OnTheFly = bn_stat_OnTheFly
        self.print_every = cfgs.print_every
        self.save_every = cfgs.save_every
        self.checkpoint_dir = checkpoint_dir
        self.evaluate = cfgs.eval
        self.mu = mu
        self.sigma = sigma
        self.best_fid = best_fid
        self.best_fid_checkpoint_path = best_fid_checkpoint_path
        self.distributed_data_parallel = cfgs.distributed_data_parallel
        self.mixed_precision = cfgs.mixed_precision
        self.synchronized_bn = cfgs.synchronized_bn

        self.lc_regularizer = cfgs.lc_regularizer
        self.lc_lambda = cfgs.lc_lambda
        self.lc_gamma = cfgs.lc_gamma
        self.plot_discriminator_predictions = cfgs.plot_discriminator_predictions

        self.start_time = datetime.now()
        self.l2_loss = torch.nn.MSELoss()
        self.ce_loss = torch.nn.CrossEntropyLoss()
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
        self.policy = "color,translation,cutout"
        self.sampler = define_sampler(self.dataset_name, self.conditional_strategy, self.batch_size, self.num_classes)
        self.counter = 0
        self.plot_param = cfgs.plot_param
        self.bool_param_plot = cfgs.bool_param_plot


        if self.cfgs.num_classes > 20: # Added # Plot the least 20 classes
            self.keys = sorted(range(len(self.train_dataset.img_num_list)), key = lambda sub: self.train_dataset.img_num_list[sub])[:20]
        
        else:
            self.keys = range(0,self.num_classes)


        if self.distributed_data_parallel: self.group = dist.new_group([n for n in range(self.n_gpus)])
        if self.mixed_precision: self.scaler = torch.cuda.amp.GradScaler()
        if self.ada: self.adtv_aug = Adaptive_Augment(self.prev_ada_p, self.ada_target, self.ada_length, self.batch_size, self.local_rank)

        if self.conditional_strategy in ['ProjGAN', 'ContraGAN', 'Proxy_NCA_GAN']:
            if isinstance(self.dis_model, DataParallel) or isinstance(self.dis_model, DistributedDataParallel):
                self.embedding_layer = self.dis_model.module.embedding
            else:
                self.embedding_layer = self.dis_model.embedding

        if self.conditional_strategy == 'ContraGAN':
            self.contrastive_criterion = Conditional_Contrastive_loss(self.local_rank, self.batch_size, self.pos_collected_numerator)
        elif self.conditional_strategy == 'Proxy_NCA_GAN':
            self.NCA_criterion = Proxy_NCA_loss(self.local_rank, self.embedding_layer, self.num_classes, self.batch_size)
        elif self.conditional_strategy == 'NT_Xent_GAN':
            self.NT_Xent_criterion = NT_Xent_loss(self.local_rank, self.batch_size)
        else:
            pass

        if self.dataset_name == "imagenet":
            self.num_eval = {'train':50000, 'valid':50000}
        elif self.dataset_name == "tiny_imagenet":
            self.num_eval = {'train':50000, 'valid':10000}
        elif self.dataset_name in ["cifar10", "cifar100","lsun", "inaturalist2019", "inaturalist2017"]  :
            self.num_eval = {'train':50000, 'test':10000}
        elif self.dataset_name == "custom":
            num_train_images, num_eval_images = len(self.train_dataset.data), len(self.eval_dataset.data)
            self.num_eval = {'train':num_train_images, 'valid':num_eval_images}
            self.num_eval = {'train':50000, 'test':10000}
        else:
            raise NotImplementedError
        
    ################################################################################################################################
    def train(self, current_step, total_step):
        gain_sn = []
        bias_sn = []
        self.dis_model.train()
        self.gen_model.train()
        if self.Gen_copy is not None:
            self.Gen_copy.train()

        if self.global_rank == 0: self.logger.info('Start training....')
        step_count = current_step
        train_iter = iter(self.train_dataloader)

        self.ada_aug_p = self.adtv_aug.initialize() if self.ada else 'No'

        alphas_r, alphas_f = [], []

        with dummy_context_mgr() as profiler:

            while step_count <= total_step:
                alpha_r_b, alpha_f_b = [], []

                # ================== TRAIN D ================== #
                toggle_grad(self.dis_model, on=True, freeze_layers=self.freeze_layers)
                toggle_grad(self.gen_model, on=False, freeze_layers=-1)
                t = set_temperature(self.conditional_strategy, self.tempering_type, self.start_temperature, self.end_temperature, step_count, self.tempering_step, total_step)
                for step_index in range(self.d_steps_per_iter):
                    alpha_r_s, alpha_f_s = [0]*self.num_classes, [0]*self.num_classes

                    self.D_optimizer.zero_grad()
                    for acml_index in range(self.accumulation_steps):
                        try:
                            real_images, real_labels = next(train_iter)
                        except StopIteration:
                            train_iter = iter(self.train_dataloader)
                            real_images, real_labels = next(train_iter)

                        real_images, real_labels = real_images.to(self.local_rank), real_labels.to(self.local_rank)
                        with torch.cuda.amp.autocast() if self.mixed_precision else dummy_context_mgr() as mpc:
                            if self.diff_aug:
                                real_images = DiffAugment(real_images, policy=self.policy)
                            if self.ada:
                                real_images, _ = augment(real_images, self.ada_aug_p)

                            if self.zcr:
                                zs, fake_labels, zs_t = sample_latents(self.prior, self.batch_size, self.z_dim, 1, self.num_classes,
                                                                    self.sigma_noise, self.local_rank)
                            
                            else:
                                zs, fake_labels = sample_latents(self.prior, self.batch_size, self.z_dim, 1, self.num_classes,
                                                                None, self.local_rank)
                            if self.latent_op:
                                zs = latent_optimise(zs, fake_labels, self.gen_model, self.dis_model, self.conditional_strategy,
                                                    self.latent_op_step, self.latent_op_rate, self.latent_op_alpha, self.latent_op_beta,
                                                    False, self.local_rank)
                            
                            
                            fake_images = self.gen_model(zs, fake_labels)
                            if self.diff_aug:
                                fake_images = DiffAugment(fake_images, policy=self.policy)
                            if self.ada:
                                fake_images, _ = augment(fake_images, self.ada_aug_p)

                            if self.conditional_strategy == "ACGAN":
                                cls_out_real, dis_out_real = self.dis_model(real_images, real_labels)
                                cls_out_fake, dis_out_fake = self.dis_model(fake_images, fake_labels)
                            elif self.conditional_strategy == "ProjGAN" or self.conditional_strategy == "no":
                                dis_out_real = self.dis_model(real_images, real_labels)
                                dis_out_fake = self.dis_model(fake_images, fake_labels)
                            elif self.conditional_strategy in ["NT_Xent_GAN", "Proxy_NCA_GAN", "ContraGAN"]:
                                cls_proxies_real, cls_embed_real, dis_out_real = self.dis_model(real_images, real_labels)
                                cls_proxies_fake, cls_embed_fake, dis_out_fake = self.dis_model(fake_images, fake_labels)
                            else:
                                raise NotImplementedError

                            dis_acml_loss = self.D_loss(dis_out_real, dis_out_fake)
                            if self.conditional_strategy == "ACGAN":
                                dis_acml_loss += (self.ce_loss(cls_out_real, real_labels) + self.ce_loss(cls_out_fake, fake_labels))
                            elif self.conditional_strategy == "NT_Xent_GAN":
                                real_images_aug = CR_DiffAug(real_images)
                                _, cls_embed_real_aug, dis_out_real_aug = self.dis_model(real_images_aug, real_labels)
                                dis_acml_loss += self.contrastive_lambda*self.NT_Xent_criterion(cls_embed_real, cls_embed_real_aug, t)
                            elif self.conditional_strategy == "Proxy_NCA_GAN":
                                dis_acml_loss += self.contrastive_lambda*self.NCA_criterion(cls_embed_real, cls_proxies_real, real_labels)
                            elif self.conditional_strategy == "ContraGAN":
                                real_cls_mask = make_mask(real_labels, self.num_classes, self.local_rank)
                                dis_acml_loss += self.contrastive_lambda*self.contrastive_criterion(cls_embed_real, cls_proxies_real,
                                                                                                    real_cls_mask, real_labels, t, self.margin)
                            else:
                                pass

                            if self.cr:
                                real_images_aug = CR_DiffAug(real_images)
                                if self.conditional_strategy == "ACGAN":
                                    cls_out_real_aug, dis_out_real_aug = self.dis_model(real_images_aug, real_labels)
                                    cls_consistency_loss = self.l2_loss(cls_out_real, cls_out_real_aug)
                                elif self.conditional_strategy == "ProjGAN" or self.conditional_strategy == "no":
                                    dis_out_real_aug = self.dis_model(real_images_aug, real_labels)
                                elif self.conditional_strategy in ["NT_Xent_GAN", "Proxy_NCA_GAN", "ContraGAN"]:
                                    _, cls_embed_real_aug, dis_out_real_aug = self.dis_model(real_images_aug, real_labels)
                                    cls_consistency_loss = self.l2_loss(cls_embed_real, cls_embed_real_aug)
                                else:
                                    raise NotImplementedError

                                consistency_loss = self.l2_loss(dis_out_real, dis_out_real_aug)
                                if self.conditional_strategy in ["ACGAN", "NT_Xent_GAN", "Proxy_NCA_GAN", "ContraGAN"]:
                                    consistency_loss += cls_consistency_loss
                                dis_acml_loss += self.cr_lambda*consistency_loss

                            if self.bcr:
                                real_images_aug = CR_DiffAug(real_images)
                                fake_images_aug = CR_DiffAug(fake_images)
                                if self.conditional_strategy == "ACGAN":
                                    cls_out_real_aug, dis_out_real_aug = self.dis_model(real_images_aug, real_labels)
                                    cls_out_fake_aug, dis_out_fake_aug = self.dis_model(fake_images_aug, fake_labels)
                                    cls_bcr_real_loss = self.l2_loss(cls_out_real, cls_out_real_aug)
                                    cls_bcr_fake_loss = self.l2_loss(cls_out_fake, cls_out_fake_aug)
                                elif self.conditional_strategy == "ProjGAN" or self.conditional_strategy == "no":
                                    dis_out_real_aug = self.dis_model(real_images_aug, real_labels)
                                    dis_out_fake_aug = self.dis_model(fake_images_aug, fake_labels)
                                elif self.conditional_strategy in ["ContraGAN", "Proxy_NCA_GAN", "NT_Xent_GAN"]:
                                    cls_proxies_real_aug, cls_embed_real_aug, dis_out_real_aug = self.dis_model(real_images_aug, real_labels)
                                    cls_proxies_fake_aug, cls_embed_fake_aug, dis_out_fake_aug = self.dis_model(fake_images_aug, fake_labels)
                                    cls_bcr_real_loss = self.l2_loss(cls_embed_real, cls_embed_real_aug)
                                    cls_bcr_fake_loss = self.l2_loss(cls_embed_fake, cls_embed_fake_aug)
                                else:
                                    raise NotImplementedError

                                bcr_real_loss = self.l2_loss(dis_out_real, dis_out_real_aug)
                                bcr_fake_loss = self.l2_loss(dis_out_fake, dis_out_fake_aug)
                                if self.conditional_strategy in ["ACGAN", "NT_Xent_GAN", "Proxy_NCA_GAN", "ContraGAN"]:
                                    bcr_real_loss += cls_bcr_real_loss
                                    bcr_fake_loss += cls_bcr_fake_loss
                                dis_acml_loss += self.real_lambda*bcr_real_loss + self.fake_lambda*bcr_fake_loss

                            if self.zcr:
                                fake_images_zaug = self.gen_model(zs_t, fake_labels)
                                if self.conditional_strategy == "ACGAN":
                                    cls_out_fake_zaug, dis_out_fake_zaug = self.dis_model(fake_images_zaug, fake_labels)
                                    cls_zcr_dis_loss = self.l2_loss(cls_out_fake, cls_out_fake_zaug)
                                elif self.conditional_strategy == "ProjGAN" or self.conditional_strategy == "no":
                                    dis_out_fake_zaug = self.dis_model(fake_images_zaug, fake_labels)
                                elif self.conditional_strategy in ["ContraGAN", "Proxy_NCA_GAN", "NT_Xent_GAN"]:
                                    cls_proxies_fake_zaug, cls_embed_fake_zaug, dis_out_fake_zaug = self.dis_model(fake_images_zaug, fake_labels)
                                    cls_zcr_dis_loss = self.l2_loss(cls_embed_fake, cls_embed_fake_zaug)
                                else:
                                    raise NotImplementedError

                                zcr_dis_loss = self.l2_loss(dis_out_fake, dis_out_fake_zaug)
                                if self.conditional_strategy in ["ACGAN", "NT_Xent_GAN", "Proxy_NCA_GAN", "ContraGAN"]:
                                    zcr_dis_loss += cls_zcr_dis_loss
                                dis_acml_loss += self.dis_lambda*zcr_dis_loss

                            if self.gradient_penalty_for_dis:
                                dis_acml_loss += self.gradient_penalty_lambda*calc_derv4gp(self.dis_model, self.conditional_strategy, real_images,
                                                                                        fake_images, real_labels, self.local_rank)
                            if self.deep_regret_analysis_for_dis:
                                dis_acml_loss += self.regret_penalty_lambda*calc_derv4dra(self.dis_model, self.conditional_strategy, real_images,
                                                                                        real_labels, self.local_rank)
                            if self.ada:
                                self.ada_aug_p = self.adtv_aug.update(dis_out_real)


                            
                            if self.lc_regularizer: # ADDED
                                with torch.no_grad():
                                    if step_count==0:
                                        alpha_r = 0
                                        alpha_f = 0
                                    elif self.best_fid_checkpoint_path is not None:
                                        alpha_r = dis_out_real.mean()
                                        alpha_f = dis_out_fake.mean()
                                    else:
                                        alpha_r = self.lc_gamma * alpha_r + (1 - self.lc_gamma) * dis_out_real
                                        alpha_f = self.lc_gamma * alpha_f + (1 - self.lc_gamma) * dis_out_fake
                                
                                lc_reg = torch.mean(F.relu(dis_out_real - alpha_f).pow(2)) + \
                                    torch.mean(F.relu(dis_out_fake - alpha_r).pow(2))               
                                #import pdb; pdb.set_trace()
                                dis_acml_loss += self.lc_lambda * lc_reg

                            dis_acml_loss = dis_acml_loss/self.accumulation_steps
                            
                            
                            if self.plot_discriminator_predictions: # ADDED
                                dis_out_real_ = dis_out_real
                                dis_out_fake_ = dis_out_fake
                                with torch.no_grad():
                                    for i in range(self.num_classes):
                                        if i not in real_labels: # [ batch_size x 1 ]
                                            alpha_r_s[i] = None
                                            continue
                                        
                                        idx = ((real_labels == i).nonzero(as_tuple=True)[0])
                                        alpha_r_s[i] = torch.mean(dis_out_real_[idx]).item()
                                    
                                   
                                    alpha_r_b.append(alpha_r_s)
                                    

                                    for i in range(self.num_classes):
                                        if i not in fake_labels:
                                            alpha_f_s[i] = None
                                            continue

                                        idx = ((fake_labels == i).nonzero(as_tuple=True)[0])
                                        alpha_f_s[i] = torch.mean(dis_out_fake_[idx]).item()
                                    
                                    alpha_f_b.append(alpha_f_s)

                        if self.mixed_precision:
                            self.scaler.scale(dis_acml_loss).backward()
                        else:
                            dis_acml_loss.backward()

                    if self.mixed_precision:
                        self.scaler.step(self.D_optimizer)
                        self.scaler.update()
                    else:
                        self.D_optimizer.step()

                    if self.weight_clipping_for_dis:
                        for p in self.dis_model.parameters():
                            p.data.clamp_(-self.weight_clipping_bound, self.weight_clipping_bound)

                if step_count % self.print_every == 0 and step_count !=0 and self.global_rank == 0:
                    if self.d_spectral_norm:
                        dis_sigmas = calculate_all_sn(self.dis_model)
                        if self.global_rank == 0: self.writer.add_scalars('SN_of_dis', dis_sigmas, step_count)
                
                # Added
                if self.plot_discriminator_predictions:
                    # [ 5 x 10 ]
                    alpha_r_b, alpha_f_b = np.array(alpha_r_b), np.array(alpha_f_b)     # Remove None
                    
                    temp_alpha_r_b = []
                    temp_alpha_f_b = []
                    for cls in range(self.num_classes):
                        # real fake
                        cnt = [0, 0]
                        sum_ = [0, 0]
                        for dis in range(self.d_steps_per_iter):
                            if(alpha_r_b[dis][cls] is not None):
                                sum_[0] += alpha_r_b[dis][cls]
                                cnt[0] += 1
                            if(alpha_f_b[dis][cls] is not None):
                                sum_[1] += alpha_f_b[dis][cls]
                                cnt[1] += 1
                        if(cnt[0]==0):
                            temp_alpha_r_b.append(None)
                        else:
                            temp_alpha_r_b.append(sum_[0]/cnt[0])
                        if(cnt[1]==0):
                            temp_alpha_f_b.append(None)
                        else:
                            temp_alpha_f_b.append(sum_[1]/cnt[1])
                    
                    temp_alpha_r_b = np.array(temp_alpha_r_b)
                    temp_alpha_f_b = np.array(temp_alpha_f_b)

                    if(len(alphas_r)==0):
                        alphas_r, alphas_f = [[0] for i in range(self.num_classes)], [[0] for i in range(self.num_classes)]
                        
                        for i in range(self.num_classes):
                            if(temp_alpha_r_b[i] is None):
                                alphas_r[i].append(alphas_r[i][-1])
                            else:
                                alphas_r[i].append(temp_alpha_r_b[i])

                            if(temp_alpha_f_b[i] is None):
                                alphas_f[i].append(alphas_f[i][-1])
                            else:
                                alphas_f[i].append(temp_alpha_f_b[i])
                        
                    else:
                        for i in range(self.num_classes):
                            if(temp_alpha_r_b[i] is None):
                                alphas_r[i].append(alphas_r[i][-1])
                            else:
                                tmp_r = self.lc_gamma * alphas_r[i][-1] + (1 - self.lc_gamma) * temp_alpha_r_b[i]
                                alphas_r[i].append(tmp_r)

                            if(temp_alpha_f_b[i] is None):
                                alphas_f[i].append(alphas_f[i][-1])
                            else:
                                tmp_f = self.lc_gamma * alphas_f[i][-1] + (1 - self.lc_gamma) * temp_alpha_f_b[i]
                                alphas_f[i].append(tmp_f)
                    
                    # Putting on TensorBoard
                    for idx in range(self.num_classes):
                        if((idx+1)==1 or (idx+1)%10==0):
                            if self.global_rank == 0: self.writer.add_scalars(f'Moving Average : Class {idx+1}', {'Real' : alphas_r[idx][-1],
                                                                                    'Fake' : alphas_f[idx][-1]
                                                                                    },step_count)

                # ================== TRAIN G ================== #
                toggle_grad(self.dis_model, False, freeze_layers=-1)
                toggle_grad(self.gen_model, True, freeze_layers=-1)
                for step_index in range(self.g_steps_per_iter):
                    self.G_optimizer.zero_grad()
                    for acml_step in range(self.accumulation_steps):
                        with torch.cuda.amp.autocast() if self.mixed_precision else dummy_context_mgr() as mpc:
                            if self.zcr:
                                zs, fake_labels, zs_t = sample_latents(self.prior, self.batch_size, self.z_dim, 1, self.num_classes,
                                                                    self.sigma_noise, self.local_rank)
                                
                            else:
                                zs, fake_labels = sample_latents(self.prior, self.batch_size, self.z_dim, 1, self.num_classes,
                                                                None, self.local_rank)
                            if self.latent_op:
                                zs, transport_cost = latent_optimise(zs, fake_labels, self.gen_model, self.dis_model, self.conditional_strategy,
                                                                    self.latent_op_step, self.latent_op_rate, self.latent_op_alpha,
                                                                    self.latent_op_beta, True, self.local_rank)
                            
                            fake_images = self.gen_model(zs, fake_labels)
                            # Added 
                            # fake_labels : 64 x 1
                            # gain_sn_tmp : layers x classes
                            # cn_gain     : layers x classes

                            if self.sn_regularize == True or (self.bool_param_plot and step_count % self.plot_param == 0): # Added for no sn calc
                                if isinstance(self.dis_model, DataParallel) or isinstance(self.dis_model, DistributedDataParallel):
                                    gain_sn_tmp, bias_sn_tmp, cn_gain, cn_bias, gamma_param, beta_param= self.gen_model.module.gen_sn(fake_labels, self.cfgs)
                                
                                else:
                                    gain_sn_tmp, bias_sn_tmp, cn_gain, cn_bias, gamma_param, beta_param = self.gen_model.gen_sn(fake_labels, self.cfgs)

                            

                            if(self.bool_param_plot and step_count % self.plot_param == 0):

                                keys = self.keys #lsun-added #lsun-added
                                
                                if self.cfgs.bool_tensorboard_plots:
                                    with torch.no_grad():
                                        for i in range(len(gain_sn_tmp)):
                                            '''
                                                tensorboard SN plots
                                            '''
                                            dct = {}


                                            for k in keys:

                                                dct[f'Class {k+1}'] = gain_sn_tmp[i][k].item()

                                            if(self.cfgs.num_classes==100):  
                                                dct[f'Class {100}'] = gain_sn_tmp[i][99].item()
                                            
                                            if self.global_rank == 0: self.writer.add_scalars(f'Gain_SN_Layer{i+1}', dct, step_count)

                            if self.diff_aug:
                                fake_images = DiffAugment(fake_images, policy=self.policy)
                            if self.ada:
                                fake_images, _ = augment(fake_images, self.ada_aug_p)

                            if self.conditional_strategy == "ACGAN":
                                cls_out_fake, dis_out_fake = self.dis_model(fake_images, fake_labels)
                            elif self.conditional_strategy == "ProjGAN" or self.conditional_strategy == "no":
                                dis_out_fake = self.dis_model(fake_images, fake_labels)
                            elif self.conditional_strategy in ["NT_Xent_GAN", "Proxy_NCA_GAN", "ContraGAN"]:
                                fake_cls_mask = make_mask(fake_labels, self.num_classes, self.local_rank)
                                cls_proxies_fake, cls_embed_fake, dis_out_fake = self.dis_model(fake_images, fake_labels)
                            else:
                                raise NotImplementedError

                            gen_acml_loss = self.G_loss(dis_out_fake)
                            

                            if self.fro_regularize:
                                fro_reg_loss = 0
                                for layer in self.gen_model.modules():
                                    if isinstance(layer, ConditionalBatchNorm2d) or isinstance(layer, ConditionalBatchNorm2d_for_skip_and_shared):
                                        for name,param in layer.named_parameters():
                                            
                                            fro_reg_loss += torch.linalg.norm(param, ord = 2) # Replacing forbinius norm with the spectral norm for ablation

                                fro_loss = self.cfgs.fro_reg_weight * fro_reg_loss
                                 
                                gen_acml_loss += fro_loss

                            

                            if self.sn_regularize:

                                cnt_layers = len(gain_sn_tmp)
                                cnt_classes = self.cfgs.num_classes

                                # Class balanced loss based on effective number of samples
                                if self.cfgs.class_balanced_loss_based_on_effective_number_of_samples:
                                    beta = self.cfgs.beta

                                    n = self.train_dataset.img_num_list
                                    En = np.array([(1 - beta**x) / (1 - beta) for x in n]) # Effective Number of Samples
                                    
                                    # lam = 125.0 / En # inversally prop to number of effective samples
                                    K = cnt_classes / (np.sum(1/En))
                                    lam = torch.tensor((K * self.cfgs.sn_reg_weight) / En, dtype=torch.float64).to(self.local_rank) # inversally prop to number of effective samples
                                else:
                                    lam = self.lam
                                

                                r_g = torch.tensor(0.0).to(self.local_rank)
                                r_b = torch.tensor(0.0).to(self.local_rank)
                                sn_tensor = torch.stack(gain_sn_tmp, dim=0)
                                bs_tensor = torch.stack(bias_sn_tmp, dim=0)
                                r_g = torch.mul(torch.pow(sn_tensor,2), lam).sum()
                                r_b = torch.mul(torch.pow(bs_tensor, 2), lam).sum()
                                reg_loss = (r_g + r_b)/(cnt_layers * cnt_classes)
                                gen_acml_loss += reg_loss
                                
                            

                            if self.latent_op:
                                gen_acml_loss += transport_cost*self.latent_norm_reg_weight

                            if self.zcr:
                                fake_images_zaug = self.gen_model(zs_t, fake_labels, tmp=2)
                                zcr_gen_loss = -1 * self.l2_loss(fake_images, fake_images_zaug)
                                gen_acml_loss += self.gen_lambda*zcr_gen_loss

                            if self.conditional_strategy == "ACGAN":
                                gen_acml_loss += self.ce_loss(cls_out_fake, fake_labels)
                            elif self.conditional_strategy == "ContraGAN":
                                gen_acml_loss += self.contrastive_lambda*self.contrastive_criterion(cls_embed_fake, cls_proxies_fake, fake_cls_mask, fake_labels, t, self.margin)
                            elif self.conditional_strategy == "Proxy_NCA_GAN":
                                gen_acml_loss += self.contrastive_lambda*self.NCA_criterion(cls_embed_fake, cls_proxies_fake, fake_labels)
                            elif self.conditional_strategy == "NT_Xent_GAN":
                                fake_images_aug = CR_DiffAug(fake_images)
                                _, cls_embed_fake_aug, dis_out_fake_aug = self.dis_model(fake_images_aug, fake_labels)
                                gen_acml_loss += self.contrastive_lambda*self.NT_Xent_criterion(cls_embed_fake, cls_embed_fake_aug, t)
                            else:
                                pass
                            
                            gen_acml_loss = gen_acml_loss/self.accumulation_steps

                        if self.mixed_precision:
                            self.scaler.scale(gen_acml_loss).backward()
                        else:
                            gen_acml_loss.backward()

                    if self.mixed_precision:
                        self.scaler.step(self.G_optimizer)
                        self.scaler.update()
                    else:
                        self.G_optimizer.step()

                    
                    if self.ema:
                        self.Gen_ema.update(step_count)

                    step_count += 1
                    

                reg_loss_ = 0. # Added
                reg_loss_dis_ = 0.
                fro_loss_ = 0.
                if(self.fro_regularize):
                    fro_loss_ = fro_loss.item()
                if(self.sn_regularize):
                    reg_loss_ = reg_loss.item()

                if step_count % self.print_every == 0 and self.global_rank == 0:
                   
                    log_message = LOG_FORMAT.format(step=step_count,
                                                    progress=step_count/total_step,
                                                    elapsed=elapsed_time(self.start_time),
                                                    SN_Regularizer_Loss=reg_loss_,
                                                    SN_Regularizer_Loss_Dis = reg_loss_dis_,
                                                    Fro_Regularizer_Loss = fro_loss_,
                                                    dis_loss=dis_acml_loss.item(),
                                                    gen_loss=gen_acml_loss.item(),
                                                    )
                    self.logger.info(log_message)

                    if self.g_spectral_norm:
                        gen_sigmas = calculate_all_sn(self.gen_model)
                        if self.global_rank == 0: self.writer.add_scalars('SN_of_gen', gen_sigmas, step_count)
                    
                    if self.global_rank == 0: self.writer.add_scalars('Losses', {'discriminator': dis_acml_loss.item(),
                                                    'generator': gen_acml_loss.item(),
                                                    'SN_Regularizer': reg_loss_}, step_count)
                    if self.ada:
                        if self.global_rank == 0: self.writer.add_scalar('ada_p', self.ada_aug_p, step_count)

                if step_count % self.save_every == 0 or step_count == total_step:
                    if self.evaluate:
                        is_best = self.evaluation(step_count, False, "N/A")
                        if self.global_rank == 0: self.save(step_count, is_best)
                    else:
                        if self.global_rank == 0: self.save(step_count, False)

                if self.cfgs.distributed_data_parallel:
                    dist.barrier(self.group)



        return step_count-1
    ################################################################################################################################


    ################################################################################################################################
    def save(self, step, is_best):
        when = "best" if is_best is True else "current"
        self.dis_model.eval()
        self.gen_model.eval()
        if self.Gen_copy is not None:
            self.Gen_copy.eval()

        if isinstance(self.gen_model, DataParallel) or isinstance(self.gen_model, DistributedDataParallel):
            gen, dis = self.gen_model.module, self.dis_model.module
            if self.Gen_copy is not None:
                gen_copy = self.Gen_copy.module
        else:
            gen, dis = self.gen_model, self.dis_model
            if self.Gen_copy is not None:
                gen_copy = self.Gen_copy

        g_states = {'seed': self.seed, 'run_name': self.run_name, 'step': step, 'best_step': self.best_step,
                    'state_dict': gen.state_dict(), 'optimizer': self.G_optimizer.state_dict(), 'ada_p': self.ada_aug_p}

        d_states = {'seed': self.seed, 'run_name': self.run_name, 'step': step, 'best_step': self.best_step,
                    'state_dict': dis.state_dict(), 'optimizer': self.D_optimizer.state_dict(), 'ada_p': self.ada_aug_p,
                    'best_fid': self.best_fid, 'best_fid_checkpoint_path': self.checkpoint_dir}

        if len(glob.glob(join(self.checkpoint_dir,"model=G-{when}-weights-step*.pth".format(when=when)))) >= 1:
            find_and_remove(glob.glob(join(self.checkpoint_dir,"model=G-{when}-weights-step*.pth".format(when=when)))[0])
            find_and_remove(glob.glob(join(self.checkpoint_dir,"model=D-{when}-weights-step*.pth".format(when=when)))[0])

        g_checkpoint_output_path = join(self.checkpoint_dir, "model=G-{when}-weights-step={step}.pth".format(when=when, step=str(step)))
        d_checkpoint_output_path = join(self.checkpoint_dir, "model=D-{when}-weights-step={step}.pth".format(when=when, step=str(step)))

        torch.save(g_states, g_checkpoint_output_path)
        torch.save(d_states, d_checkpoint_output_path)

        if when == "best" or 1: # Added checkpoint
            if len(glob.glob(join(self.checkpoint_dir,"model=G-current-weights-step*.pth"))) >= 1:
                find_and_remove(glob.glob(join(self.checkpoint_dir,"model=G-current-weights-step*.pth"))[0])
                find_and_remove(glob.glob(join(self.checkpoint_dir,"model=D-current-weights-step*.pth"))[0])

            g_checkpoint_output_path_ = join(self.checkpoint_dir, "model=G-current-weights-step={step}.pth".format(step=str(step)))
            d_checkpoint_output_path_ = join(self.checkpoint_dir, "model=D-current-weights-step={step}.pth".format(step=str(step)))

            torch.save(g_states, g_checkpoint_output_path_)
            torch.save(d_states, d_checkpoint_output_path_)

        if self.Gen_copy is not None:
            g_ema_states = {'state_dict': gen_copy.state_dict()}
            if len(glob.glob(join(self.checkpoint_dir, "model=G_ema-{when}-weights-step*.pth".format(when=when)))) >= 1:
                find_and_remove(glob.glob(join(self.checkpoint_dir, "model=G_ema-{when}-weights-step*.pth".format(when=when)))[0])

            g_ema_checkpoint_output_path = join(self.checkpoint_dir, "model=G_ema-{when}-weights-step={step}.pth".format(when=when, step=str(step)))

            torch.save(g_ema_states, g_ema_checkpoint_output_path)

            if when == "best":
                if len(glob.glob(join(self.checkpoint_dir,"model=G_ema-current-weights-step*.pth".format(when=when)))) >= 1:
                    find_and_remove(glob.glob(join(self.checkpoint_dir,"model=G_ema-current-weights-step*.pth".format(when=when)))[0])

                g_ema_checkpoint_output_path_ = join(self.checkpoint_dir, "model=G_ema-current-weights-step={step}.pth".format(when=when, step=str(step)))

                torch.save(g_ema_states, g_ema_checkpoint_output_path_)

        if self.logger:
            if self.global_rank == 0: self.logger.info("Save model to {}".format(self.checkpoint_dir))

        self.dis_model.train()
        self.gen_model.train()
        if self.Gen_copy is not None:
            self.Gen_copy.train()
    ################################################################################################################################


    ################################################################################################################################
    def evaluation(self, step, standing_statistics, standing_step):
        if standing_statistics: self.counter += 1
        with torch.no_grad() if self.latent_op is False else dummy_context_mgr() as mpc:
            if self.global_rank == 0: self.logger.info("Start Evaluation ({step} Step): {run_name}".format(step=step, run_name=self.run_name))
            is_best = False
            num_split, num_run4PR, num_cluster4PR, beta4PR = 1, 10, 20, 8
            num_random_restarts = 3 # fix restarts

            self.dis_model.eval()
            generator = change_generator_mode(self.gen_model, self.Gen_copy, self.bn_stat_OnTheFly, standing_statistics, standing_step,
                                                self.prior, self.batch_size, self.z_dim, self.num_classes, self.local_rank, training=False, counter=self.counter)
            
            if self.cfgs.evaluation_checkpoint != False:
                
                fid_scores, kl_div_scores, kl_scores = np.zeros(num_random_restarts), np.zeros(num_random_restarts), np.zeros(num_random_restarts)

                for i in range(num_random_restarts):
                    
                    fid_scores[i], _, _ = calculate_fid_score(self.eval_dataloader, generator, self.dis_model, self.inception_model, 50000,
                                                    self.truncated_factor, self.prior, self.latent_op, self.latent_op_step4eval, self.latent_op_alpha,
                                                    self.latent_op_beta, self.local_rank, self.logger, self.mu, self.sigma, self.run_name)
                    self.logger.info("{i}: FID score {scr}".format(i = i, scr = fid_scores[i]))



                    kl_scores[i], _ = calculate_incep_score(self.eval_dataloader, generator, self.dis_model, self.inception_model, 50000,
                                                        self.truncated_factor, self.prior, self.latent_op, self.latent_op_step4eval, self.latent_op_alpha,
                                                        self.latent_op_beta, num_split, self.local_rank, self.logger)
                    
                    self.logger.info("{i}: Inception score {scr}".format(i = i, scr = kl_scores[i]))
                    nearest_k = 5

                self.logger.info("FID score using Using {type} moments score: Avg Scores: {scr} Std Dev: {std_dev}".format(type = self.eval_type, scr = str(np.mean(fid_scores)), std_dev = str(np.std(fid_scores))))

                self.logger.info("Inception score using Using {type} moments score: Avg Scores: {scr} Std Dev: {std_dev}".format(type = self.eval_type, scr = str(np.mean(kl_scores)), std_dev = str(np.std(kl_scores))))

                fid_score = fid_scores[-1]    
            else:
                fid_score, self.m1, self.s1 = calculate_fid_score(self.eval_dataloader, generator, self.dis_model, self.inception_model, self.num_eval[self.eval_type],
                                                                self.truncated_factor, self.prior, self.latent_op, self.latent_op_step4eval, self.latent_op_alpha,
                                                                self.latent_op_beta, self.local_rank, self.logger, self.mu, self.sigma, self.run_name)
   
                
                if self.cfgs.compute_prdc:
                    nearest_k = 5

                    fake_feats = generate_images_and_stack_features(
                                                                   generator=generator,
                                                                   discriminator=self.dis_model,
                                                                   eval_model=self.inception_model,
                                                                   num_generate=self.num_eval[self.eval_type],
                                                                   batch_size=self.cfgs.batch_size,
                                                                   z_prior=self.prior,
                                                                   truncation_factor=self.cfgs.truncated_factor,
                                                                   num_classes=self.cfgs.num_classes,
                                                                   world_size=1,
                                                                   DDP=False,
                                                                   device=self.local_rank,
                                                                   logger=self.logger,
                                                                   disable_tqdm=self.global_rank != 0,
                                                                   latent_op = self.cfgs.latent_op,
                                                                   latent_op_step = self.cfgs.latent_op_step,
                                                                   latent_op_alpha = self.cfgs.latent_op_alpha,
                                                                   latent_op_beta = self.cfgs.latent_op_beta
                                                                   )


                    prc, rec, dns, cvg = calculate_pr_dc(fake_feats=fake_feats,
                                                          data_loader=self.eval_dataloader,
                                                          eval_model=self.inception_model, # eval_model =? inception model
                                                          num_generate=self.num_eval[self.eval_type],
                                                          cfgs=self.cfgs,
                                                          quantize=True,
                                                          nearest_k=nearest_k,
                                                          world_size=1, # Number of gpus
                                                          DDP=False,     # Distributed data parallel
                                                          disable_tqdm=True)
                    if self.global_rank == 0:
                        # type=self.RUN.ref_dataset
                        ref_dataset = "eval"
                        self.logger.info("Improved Precision (Step: {step}, Using {type} images): {prc}".format(
                            step=step, type=ref_dataset, prc=prc))
                        self.logger.info("Improved Recall (Step: {step}, Using {type} images): {rec}".format(
                            step=step, type=ref_dataset, rec=rec))
                        self.logger.info("Density (Step: {step}, Using {type} images): {dns}".format(
                            step=step, type=ref_dataset, dns=dns))
                        self.logger.info("Coverage (Step: {step}, Using {type} images): {cvg}".format(
                            step=step, type=ref_dataset, cvg=cvg))
                        # if writing:
                        self.writer.add_scalars('Improved Precision', {'Improved Precision': prc}, step)
                        self.writer.add_scalars('Improved Recall', {'Improved Recall': rec}, step)
                        self.writer.add_scalars('Density', {'Density': dns}, step)
                        self.writer.add_scalars('Coverage', {'Coverage': cvg}, step)
                       

                if False: # INtra FID Calculation Code Disabled due to time consuming
                    intra_fid_score, _, _ = calculate_intra_fid_score(self.eval_dataloader, generator, self.dis_model, self.inception_model, self.num_eval[self.eval_type],
                                                                    self.truncated_factor, self.prior, self.latent_op, self.latent_op_step4eval, self.latent_op_alpha,
                                                                    self.latent_op_beta, self.local_rank, self.logger, self.mu, self.sigma, self.run_name)


                kl_score, kl_std = calculate_incep_score(self.eval_dataloader, generator, self.dis_model, self.inception_model, self.num_eval[self.eval_type],
                                                        self.truncated_factor, self.prior, self.latent_op, self.latent_op_step4eval, self.latent_op_alpha,
                                                        self.latent_op_beta, num_split, self.local_rank, self.logger)
                
                # precision, recall, f_beta, f_beta_inv = calculate_f_beta_score(self.eval_dataloader, generator, self.dis_model, self.inception_model, self.num_eval[self.eval_type],
                #                                                             num_run4PR, num_cluster4PR, beta4PR, self.truncated_factor, self.prior, self.latent_op,
                #                                                             self.latent_op_step4eval, self.latent_op_alpha, self.latent_op_beta, self.local_rank, self.logger)
                #PR_Curve = plot_pr_curve(precision, recall, self.run_name, self.logger, logging=True)


                if self.conditional_strategy in ['ProjGAN', 'ContraGAN', 'Proxy_NCA_GAN']:
                    if self.dataset_name == "cifar10":
                        classes = torch.tensor([c for c in range(self.num_classes)], dtype=torch.long).to(self.local_rank)
                        labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
                    elif self.dataset_name == "cifar100":
                        classes = torch.tensor([c for c in range(self.num_classes)], dtype=torch.long).to(self.local_rank)
                        labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
                    else:
                        if self.num_classes > 10:
                            classes = torch.tensor(random.sample(range(0, self.num_classes), 10), dtype=torch.long).to(self.local_rank)
                        else:
                            classes = torch.tensor([c for c in range(self.num_classes)], dtype=torch.long).to(self.local_rank)
                        labels = classes.detach().cpu().numpy()
                    proxies = self.embedding_layer(classes)
                    sim_p = self.cosine_similarity(proxies.unsqueeze(1), proxies.unsqueeze(0))
                    #sim_heatmap = plot_sim_heatmap(sim_p.detach().cpu().numpy(), labels, labels, self.run_name, self.logger, logging=True)

                if self.D_loss.__name__ != "loss_wgan_dis":
                    real_train_acc, fake_acc = calculate_accuracy(self.train_dataloader, generator, self.dis_model, self.D_loss, self.num_eval[self.eval_type],
                                                                self.truncated_factor, self.prior, self.latent_op, self.latent_op_step, self.latent_op_alpha,
                                                                self.latent_op_beta, self.local_rank, cr=self.cr, logger=self.logger, eval_generated_sample=True)

                    if self.eval_type == 'train':
                        acc_dict = {'real_train': real_train_acc, 'fake': fake_acc}
                    else:
                        real_eval_acc = calculate_accuracy(self.eval_dataloader, generator, self.dis_model, self.D_loss, self.num_eval[self.eval_type],
                                                        self.truncated_factor, self.prior, self.latent_op, self.latent_op_step, self.latent_op_alpha,
                                                        self. latent_op_beta, self.local_rank, cr=self.cr, logger=self.logger, eval_generated_sample=False)
                        acc_dict = {'real_train': real_train_acc, 'real_valid': real_eval_acc, 'fake': fake_acc}

                    if self.global_rank == 0: self.writer.add_scalars('Accuracy', acc_dict, step)

                if self.best_fid is None:
                    self.best_fid, self.best_step, is_best = fid_score, step, True
                
                else:
                    if fid_score <= self.best_fid:
                        self.best_fid, self.best_step, is_best = fid_score, step, True

                if self.global_rank == 0:
                    self.writer.add_scalars('FID score', {'using {type} moments'.format(type=self.eval_type):fid_score}, step)
                    self.writer.add_scalars('IS score', {'{num} generated images'.format(num=str(self.num_eval[self.eval_type])):kl_score}, step)

                    self.logger.info('FID score (Step: {step}, Using {type} moments): {FID}'.format(step=step, type=self.eval_type, FID=fid_score))
                #     #if self.cfgs.compute_intra_fid:
                #     #    self.logger.info('Intra FID score (Step: {step}, Using {type} moments): {FID}'.format(step=step, type=self.eval_type, FID=intra_fid_score))
                    self.logger.info('Inception score (Step: {step}, {num} generated images): {IS}'.format(step=step, num=str(self.num_eval[self.eval_type]), IS=kl_score))
                if self.train:
                    self.logger.info('Best FID score (Step: {step}, Using {type} moments): {FID}'.format(step=self.best_step, type=self.eval_type, FID=self.best_fid))
                    # Added
            img_name = "current_fid"
        
            if self.best_fid == fid_score:
                img_name="best_fid"

            self.run_image_visualization(nrow=self.cfgs.nrow, ncol=self.cfgs.ncol, standing_statistics=False, standing_step="N/A", img_name=img_name, step=step, fid_score=fid_score)

            self.dis_model.train()
            generator = change_generator_mode(self.gen_model, self.Gen_copy, self.bn_stat_OnTheFly, standing_statistics, standing_step,
                                            self.prior, self.batch_size, self.z_dim, self.num_classes, self.local_rank, training=True, counter=self.counter)
                        

        return is_best
    ################################################################################################################################


    ################################################################################################################################
    def save_images(self, is_generate, standing_statistics, standing_step, png=True, npz=True):
        if self.global_rank == 0: self.logger.info('Start save images....')
        if standing_statistics: self.counter += 1
        with torch.no_grad() if self.latent_op is False else dummy_context_mgr() as mpc:
            self.dis_model.eval()
            generator = change_generator_mode(self.gen_model, self.Gen_copy, self.bn_stat_OnTheFly, standing_statistics, standing_step,
                                              self.prior, self.batch_size, self.z_dim, self.num_classes, self.local_rank, training=False, counter=self.counter)

            if png:
                save_images_png(self.run_name, self.eval_dataloader, self.num_eval[self.eval_type], self.num_classes, generator,
                                self.dis_model, is_generate, self.truncated_factor, self.prior, self.latent_op, self.latent_op_step,
                                self.latent_op_alpha, self.latent_op_beta, self.local_rank)
            if npz:
                save_images_npz(self.run_name, self.eval_dataloader, self.num_eval[self.eval_type], self.num_classes, generator,
                                self.dis_model, is_generate, self.truncated_factor, self.prior, self.latent_op, self.latent_op_step,
                                self.latent_op_alpha, self.latent_op_beta, self.local_rank)

            generator = change_generator_mode(self.gen_model, self.Gen_copy, self.bn_stat_OnTheFly, standing_statistics, standing_step,
                                              self.prior, self.batch_size, self.z_dim, self.num_classes, self.local_rank, training=True, counter=self.counter)
    ################################################################################################################################

    ################################################################################################################################

    def run_image_visualization_lt(self, ncol, standing_statistics, standing_step, img_name=None, step=None):
        if self.global_rank == 0: self.logger.info('Start visualize lt images....')
        if standing_statistics: self.counter += 1
        assert self.batch_size % 8 ==0, "batch size should be devided by 8!"
        with torch.no_grad() if self.latent_op is False else dummy_context_mgr() as mpc:
            generator = change_generator_mode(self.gen_model, self.Gen_copy, self.bn_stat_OnTheFly, standing_statistics, standing_step,
                                              self.prior, self.batch_size, self.z_dim, self.num_classes, self.local_rank, training=False, counter=self.counter)
            

            num_batches = len(self.train_dataloader.dataset)//self.batch_size
            iv_iter = iter(self.train_dataloader)
            real={}

            for i in range(num_batches):
                real_images, real_labels = next(iv_iter)
                real_images, real_labels = real_images.to(self.local_rank), real_labels.to(self.local_rank)

                if i == 0:
                    real["images"] = real_images.detach().cpu()
                    real["labels"] = real_labels
                else:
                    real["images"] = torch.cat([real["images"], real_images.detach().cpu()], dim=0)
                    real["labels"] = torch.cat([real["labels"], real_labels])


            labels = torch.tensor(range(self.num_classes)).to(self.local_rank)
            counts = torch.tensor(self.train_dataset.img_num_list).to(self.local_rank)
            j, n = 0, self.batch_size//8
            _ , indices = torch.sort(counts, descending=True)
            head_cls_indices = labels[indices[j:j+n]]
            print('head_cls_indices',head_cls_indices)

            _ , indices = torch.sort(counts, descending=False)
            tail_cls_indices = labels[indices[j:j+100]]
            
            start_tail_indices = tail_cls_indices.sort()[0][:32]
            print('start_tail_classes', start_tail_indices)

            tail_cls_indices = tail_cls_indices[j:j+n]
            print('tail_cls_indices',tail_cls_indices)

           
            types = ["tail_", 'head_', 'tail_start']

            for classes_type in types:
                if classes_type == "head_":
                    cls_indices = head_cls_indices
                elif classes_type == "tail_":
                    cls_indices = tail_cls_indices
                elif classes_type == "tail_start":
                    cls_indices = start_tail_indices
                
                if self.zcr:
                    zs, _, zs_t = sample_latents(self.prior, self.batch_size, self.z_dim, 1, self.num_classes,
                                                        self.sigma_noise, self.local_rank, sampler=self.sampler)
                else:
                    zs, _ = sample_latents(self.prior, self.batch_size, self.z_dim, 1, self.num_classes, None,
                                                    self.local_rank, sampler=self.sampler)

                if self.latent_op:
                    zs = latent_optimise(zs, fake_labels, self.gen_model, self.dis_model, self.conditional_strategy,
                                            self.latent_op_step, self.latent_op_rate, self.latent_op_alpha, self.latent_op_beta,
                                            False, self.local_rank, sampler=self.sampler)

                fake_labels = []
                for idx in cls_indices:
                    fake_labels += [idx]*8
                fake_labels = torch.tensor(fake_labels, dtype=torch.long).to(self.local_rank)
                # print(fake_labels)

                generated_images = generator(zs, fake_labels, evaluation=True)
                # print(generated_images.shape)

                plot_img_canvas((generated_images.detach().cpu()+1)/2, "./figures/{run_name}/{classes_type}/canvas_generated.png".\
                                format(run_name=self.run_name, classes_type=classes_type), ncol, self.logger, logging=True)
                
                self.writer.add_images(img_name, (generated_images.detach().cpu()+1)/2, global_step=step)

                #visualize specific classes
                for i in range(cls_indices.cpu().shape[0]):
                    if i==0:
                        real_canvas = real['images'][real['labels'].cpu()==cls_indices[i].cpu()][:8]
                        N, C, W, H = real_canvas.shape
                        if real_canvas.shape[0]<8:
                            real_canvas = torch.cat([real_canvas, torch.zeros((8-N, C, H, W),dtype=real_canvas.dtype)], dim=0)

                    else:
                        real_imgs = real['images'][real['labels'].cpu()==cls_indices[i].cpu()][:8]
                        N, C, W, H = real_imgs.shape
                        if real_imgs.shape[0]<8:
                            real_imgs = torch.cat([real_imgs, torch.zeros((8-N, C, H, W),dtype=real_imgs.dtype)], dim=0)

                        
                        real_canvas = torch.cat([real_canvas, real_imgs], dim=0)
                
                plot_img_canvas((real_canvas+1)/2, "./figures/{run_name}/{classes_type}/canvas_real.png".\
                                format(run_name=self.run_name, classes_type=classes_type), ncol, self.logger, logging=True)
                
                
            generator = change_generator_mode(self.gen_model, self.Gen_copy, self.bn_stat_OnTheFly, standing_statistics, standing_step,
                                              self.prior, self.batch_size, self.z_dim, self.num_classes, self.local_rank, training=True, counter=self.counter)
    ################################################################################################################################
    

    ################################################################################################################################
    def run_image_visualization(self, nrow, ncol, standing_statistics, standing_step, img_name=None, step=None, fid_score=None):
        if self.global_rank == 0: self.logger.info('Start visualize images....')
        if standing_statistics: self.counter += 1
        assert self.batch_size % 8 ==0, "batch size should be devided by 8!"
        with torch.no_grad() if self.latent_op is False else dummy_context_mgr() as mpc:
            generator = change_generator_mode(self.gen_model, self.Gen_copy, self.bn_stat_OnTheFly, standing_statistics, standing_step,
                                              self.prior, self.batch_size, self.z_dim, self.num_classes, self.local_rank, training=False, counter=self.counter)

            if self.zcr:
                zs, fake_labels, zs_t = sample_latents(self.prior, self.batch_size, self.z_dim, 1, self.num_classes,
                                                     self.sigma_noise, self.local_rank, sampler=self.sampler)
            else:
                zs, fake_labels = sample_latents(self.prior, self.batch_size, self.z_dim, 1, self.num_classes, None,
                                                 self.local_rank, sampler=self.sampler)

            if self.latent_op:
                zs = latent_optimise(zs, fake_labels, self.gen_model, self.dis_model, self.conditional_strategy,
                                        self.latent_op_step, self.latent_op_rate, self.latent_op_alpha, self.latent_op_beta,
                                        False, self.local_rank, sampler=self.sampler)

            generated_images = generator(zs, fake_labels, evaluation=True)

            plot_img_canvas((generated_images.detach().cpu()+1)/2, "./figures/{run_name}/generated_canvas_{img_name}.png".\
                            format(run_name=self.run_name, img_name=img_name), ncol, self.logger, logging=True)

            self.writer.add_images(img_name, (generated_images.detach().cpu()+1)/2, global_step=step)

            generator = change_generator_mode(self.gen_model, self.Gen_copy, self.bn_stat_OnTheFly, standing_statistics, standing_step,
                                              self.prior, self.batch_size, self.z_dim, self.num_classes, self.local_rank, training=True, counter=self.counter)
    ################################################################################################################################


    ################################################################################################################################
    def run_nearest_neighbor(self, nrow, ncol, standing_statistics, standing_step):
        if self.global_rank == 0: self.logger.info('Start nearest neighbor analysis....')
        if standing_statistics: self.counter += 1
        assert self.batch_size % 8 ==0, "batch size should be devided by 8!"
        with torch.no_grad() if self.latent_op is False else dummy_context_mgr() as mpc:
            generator = change_generator_mode(self.gen_model, self.Gen_copy, self.bn_stat_OnTheFly, standing_statistics, standing_step,
                                              self.prior, self.batch_size, self.z_dim, self.num_classes, self.local_rank, training=False, counter=self.counter)

            resnet50_model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained=True)
            resnet50_conv = nn.Sequential(*list(resnet50_model.children())[:-1]).to(self.local_rank)
            if self.n_gpus > 1:
                resnet50_conv = DataParallel(resnet50_conv, output_device=self.local_rank)
            resnet50_conv.eval()

            for c in tqdm(range(self.num_classes)):
                fake_images, fake_labels = generate_images_for_KNN(self.batch_size, c, generator, self.dis_model, self.truncated_factor, self.prior, self.latent_op,
                                                                   self.latent_op_step, self.latent_op_alpha, self.latent_op_beta, self.local_rank)
                fake_image = torch.unsqueeze(fake_images[0], dim=0)
                fake_anchor_embedding = torch.squeeze(resnet50_conv((fake_image+1)/2))

                num_samples, target_sampler = target_class_sampler(self.train_dataset, c)
                batch_size = self.batch_size if num_samples >= self.batch_size else num_samples
                train_dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size, shuffle=False, sampler=target_sampler,
                                                               num_workers=self.num_workers, pin_memory=True)
                train_iter = iter(train_dataloader)
                for batch_idx in range(num_samples//batch_size):
                    real_images, real_labels = next(train_iter)
                    real_images = real_images.to(self.local_rank)
                    real_embeddings = torch.squeeze(resnet50_conv((real_images+1)/2))
                    if batch_idx == 0:
                        distances = torch.square(real_embeddings - fake_anchor_embedding).mean(dim=1).detach().cpu().numpy()
                        holder = real_images.detach().cpu().numpy()
                    else:
                        distances = np.concatenate([distances, torch.square(real_embeddings - fake_anchor_embedding).mean(dim=1).detach().cpu().numpy()], axis=0)
                        holder = np.concatenate([holder, real_images.detach().cpu().numpy()], axis=0)

                nearest_indices = (-distances).argsort()[-(ncol-1):][::-1]
                if c % nrow == 0:
                    canvas = np.concatenate([fake_image.detach().cpu().numpy(), holder[nearest_indices]], axis=0)
                elif c % nrow == nrow-1:
                    row_images = np.concatenate([fake_image.detach().cpu().numpy(), holder[nearest_indices]], axis=0)
                    canvas = np.concatenate((canvas, row_images), axis=0)
                    plot_img_canvas((torch.from_numpy(canvas)+1)/2, "./figures/{run_name}/Fake_anchor_{ncol}NN_{cls}_classes.png".\
                                    format(run_name=self.run_name, ncol=ncol, cls=c+1), ncol, self.logger, logging=False)
                else:
                    row_images = np.concatenate([fake_image.detach().cpu().numpy(), holder[nearest_indices]], axis=0)
                    canvas = np.concatenate((canvas, row_images), axis=0)

            generator = change_generator_mode(self.gen_model, self.Gen_copy, self.bn_stat_OnTheFly, standing_statistics, standing_step,
                                              self.prior, self.batch_size, self.z_dim, self.num_classes, self.local_rank, training=True, counter=self.counter)
    ################################################################################################################################


    ################################################################################################################################
    def run_linear_interpolation(self, nrow, ncol, fix_z, fix_y, standing_statistics, standing_step, num_images=100):
        if self.global_rank == 0: self.logger.info('Start linear interpolation analysis....')
        if standing_statistics: self.counter += 1
        assert self.batch_size % 8 ==0, "batch size should be devided by 8!"
        with torch.no_grad() if self.latent_op is False else dummy_context_mgr() as mpc:
            generator = change_generator_mode(self.gen_model, self.Gen_copy, self.bn_stat_OnTheFly, standing_statistics, standing_step,
                                              self.prior, self.batch_size, self.z_dim, self.num_classes, self.local_rank, training=False, counter=self.counter)
            shared = generator.module.shared if isinstance(generator, DataParallel) or isinstance(generator, DistributedDataParallel) else generator.shared
            assert int(fix_z)*int(fix_y) != 1, "unable to switch fix_z and fix_y on together!"

            for num in tqdm(range(num_images)):
                if fix_z:
                    zs = torch.randn(nrow, 1, self.z_dim, device=self.local_rank)
                    zs = zs.repeat(1, ncol, 1).view(-1, self.z_dim)
                    name = "fix_z"
                else:
                    zs = interp(torch.randn(nrow, 1, self.z_dim, device=self.local_rank),
                                torch.randn(nrow, 1, self.z_dim, device=self.local_rank),
                                ncol - 2).view(-1, self.z_dim)

                if fix_y:
                    ys = sample_1hot(nrow, self.num_classes, device=self.local_rank)
                    ys = shared(ys).view(nrow, 1, -1)
                    ys = ys.repeat(1, ncol, 1).view(nrow * (ncol), -1)
                    name = "fix_y"
                else:
                    ys = interp(shared(sample_1hot(nrow, self.num_classes)).view(nrow, 1, -1),
                                shared(sample_1hot(nrow, self.num_classes)).view(nrow, 1, -1),
                                ncol-2).view(nrow * (ncol), -1)

                interpolated_images = generator(zs, None, shared_label=ys, evaluation=True)

                plot_img_canvas((interpolated_images.detach().cpu()+1)/2, "./figures/{run_name}/{num}_Interpolated_images_{fix_flag}.png".\
                                format(num=num, run_name=self.run_name, fix_flag=name), ncol, self.logger, logging=False)

            generator = change_generator_mode(self.gen_model, self.Gen_copy, self.bn_stat_OnTheFly, standing_statistics, standing_step,
                                              self.prior, self.batch_size, self.z_dim, self.num_classes, self.local_rank, training=True, counter=self.counter)
    ################################################################################################################################


    ################################################################################################################################
    def run_frequency_analysis(self, num_images, standing_statistics, standing_step):
        if self.global_rank == 0: self.logger.info('Start frequency analysis....')
        if standing_statistics: self.counter += 1
        with torch.no_grad() if self.latent_op is False else dummy_context_mgr() as mpc:
            generator = change_generator_mode(self.gen_model, self.Gen_copy, self.bn_stat_OnTheFly, standing_statistics, standing_step,
                                              self.prior, self.batch_size, self.z_dim, self.num_classes, self.local_rank, training=False, counter=self.counter)

            train_iter = iter(self.train_dataloader)
            num_batches = num_images//self.batch_size
            for i in range(num_batches):
                if self.zcr:
                    zs, fake_labels, zs_t = sample_latents(self.prior, self.batch_size, self.z_dim, 1, self.num_classes,
                                                           self.sigma_noise, self.local_rank)


                else:
                    zs, fake_labels = sample_latents(self.prior, self.batch_size, self.z_dim, 1, self.num_classes,
                                                     None, self.local_rank)

                if self.latent_op:
                    zs = latent_optimise(zs, fake_labels, self.gen_model, self.dis_model, self.conditional_strategy,
                                         self.latent_op_step, self.latent_op_rate, self.latent_op_alpha, self.latent_op_beta,
                                         False, self.local_rank)

                real_images, real_labels = next(train_iter)
                fake_images = generator(zs, fake_labels, evaluation=True).detach().cpu().numpy()

                real_images = np.asarray((real_images + 1)*127.5, np.uint8)
                fake_images = np.asarray((fake_images + 1)*127.5, np.uint8)

                if i == 0:
                    real_array = real_images
                    fake_array = fake_images
                else:
                    real_array = np.concatenate([real_array, real_images], axis = 0)
                    fake_array = np.concatenate([fake_array, fake_images], axis = 0)

            N, C, H, W = np.shape(real_array)
            real_r, real_g, real_b = real_array[:,0,:,:], real_array[:,1,:,:], real_array[:,2,:,:]
            real_gray = 0.2989 * real_r + 0.5870 * real_g + 0.1140 * real_b
            fake_r, fake_g, fake_b = fake_array[:,0,:,:], fake_array[:,1,:,:], fake_array[:,2,:,:]
            fake_gray = 0.2989 * fake_r + 0.5870 * fake_g + 0.1140 * fake_b
            for j in tqdm(range(N)):
                real_gray_f = np.fft.fft2(real_gray[j] - ndimage.median_filter(real_gray[j], size= H//8))
                fake_gray_f = np.fft.fft2(fake_gray[j] - ndimage.median_filter(fake_gray[j], size=H//8))

                real_gray_f_shifted = np.fft.fftshift(real_gray_f)
                fake_gray_f_shifted = np.fft.fftshift(fake_gray_f)

                if j == 0:
                    real_gray_spectrum = 20*np.log(np.abs(real_gray_f_shifted))/N
                    fake_gray_spectrum = 20*np.log(np.abs(fake_gray_f_shifted))/N
                else:
                    real_gray_spectrum += 20*np.log(np.abs(real_gray_f_shifted))/N
                    fake_gray_spectrum += 20*np.log(np.abs(fake_gray_f_shifted))/N

            plot_spectrum_image(real_gray_spectrum, fake_gray_spectrum, self.run_name, self.logger, logging=True)

            generator = change_generator_mode(self.gen_model, self.Gen_copy, self.bn_stat_OnTheFly, standing_statistics, standing_step,
                                              self.prior, self.batch_size, self.z_dim, self.num_classes, self.local_rank, training=True, counter=self.counter)
    ################################################################################################################################


    ################################################################################################################################
    def run_tsne(self, dataloader, standing_statistics, standing_step, cfgs):
        if self.global_rank == 0: self.logger.info('Start tsne analysis....')
        if standing_statistics: self.counter += 1
        with torch.no_grad() if self.latent_op is False else dummy_context_mgr() as mpc:
            generator = change_generator_mode(self.gen_model, self.Gen_copy, self.bn_stat_OnTheFly, standing_statistics, standing_step,
                                              self.prior, self.batch_size, self.z_dim, self.num_classes, self.local_rank, training=False, counter=self.counter)
            if isinstance(self.gen_model, DataParallel) or isinstance(self.gen_model, DistributedDataParallel):
                dis_model = self.dis_model.module
            else:
                dis_model = self.dis_model

            save_output = SaveOutput()
            hook_handles = []
            real, fake = {}, {}
            tsne_iter = iter(dataloader)
            num_batches = len(dataloader.dataset)//self.batch_size
            for name, layer in dis_model.named_children():
                if name == "linear1":
                    handle = layer.register_forward_pre_hook(save_output)
                    hook_handles.append(handle)

            for i in range(num_batches):
                if self.zcr:
                    zs, fake_labels, zs_t = sample_latents(self.prior, self.batch_size, self.z_dim, 1, self.num_classes,
                                                           self.sigma_noise, self.local_rank)
                else:
                    zs, fake_labels = sample_latents(self.prior, self.batch_size, self.z_dim, 1, self.num_classes,
                                                     None, self.local_rank)

                if self.latent_op:
                    zs = latent_optimise(zs, fake_labels, self.gen_model, self.dis_model, self.conditional_strategy,
                                         self.latent_op_step, self.latent_op_rate, self.latent_op_alpha, self.latent_op_beta,
                                         False, self.local_rank)

                real_images, real_labels = next(tsne_iter)
                real_images, real_labels = real_images.to(self.local_rank), real_labels.to(self.local_rank)
                fake_images = generator(zs, fake_labels, evaluation=True)

                if self.conditional_strategy == "ACGAN":
                    cls_out_real, dis_out_real = self.dis_model(real_images, real_labels)
                elif self.conditional_strategy == "ProjGAN" or self.conditional_strategy == "no":
                    dis_out_real = self.dis_model(real_images, real_labels)
                elif self.conditional_strategy in ["NT_Xent_GAN", "Proxy_NCA_GAN", "ContraGAN"]:
                    cls_proxies_real, cls_embed_real, dis_out_real = self.dis_model(real_images, real_labels)
                else:
                    raise NotImplementedError

                if i == 0:
                    real["embeds"] = save_output.outputs[0][0].detach().cpu().numpy()
                    real["labels"] = real_labels.detach().cpu().numpy()
                else:
                    real["embeds"] = np.concatenate([real["embeds"], save_output.outputs[0][0].cpu().detach().numpy()], axis=0)
                    real["labels"] = np.concatenate([real["labels"], real_labels.detach().cpu().numpy()])

                save_output.clear()

                if self.conditional_strategy == "ACGAN":
                    cls_out_fake, dis_out_fake = self.dis_model(fake_images, fake_labels)
                elif self.conditional_strategy == "ProjGAN" or self.conditional_strategy == "no":
                    dis_out_fake = self.dis_model(fake_images, fake_labels)
                elif self.conditional_strategy in ["NT_Xent_GAN", "Proxy_NCA_GAN", "ContraGAN"]:
                    cls_proxies_fake, cls_embed_fake, dis_out_fake = self.dis_model(fake_images, fake_labels)
                else:
                    raise NotImplementedError

                if i == 0:
                    fake["embeds"] = save_output.outputs[0][0].detach().cpu().numpy()
                    fake["labels"] = fake_labels.detach().cpu().numpy()
                else:
                    fake["embeds"] = np.concatenate([fake["embeds"], save_output.outputs[0][0].cpu().detach().numpy()], axis=0)
                    fake["labels"] = np.concatenate([fake["labels"], fake_labels.detach().cpu().numpy()])

                save_output.clear()

            # t-SNE
            tsne = TSNE(n_components=1, verbose=1, perplexity=50, n_iter=5000)
            if self.num_classes > 10:
                 cls_indices = np.random.permutation(self.num_classes)[:10]
                 real["embeds"] = real["embeds"][np.isin(real["labels"], cls_indices)]
                 real["labels"] = real["labels"][np.isin(real["labels"], cls_indices)]
                 fake["embeds"] = fake["embeds"][np.isin(fake["labels"], cls_indices)]
                 fake["labels"] = fake["labels"][np.isin(fake["labels"], cls_indices)]

            real_tsne_results = tsne.fit_transform(real["embeds"])
            plot_tsne_scatter_plot(real, real_tsne_results, "real", self.run_name, self.logger, logging=True, cfgs=cfgs)
            fake_tsne_results = tsne.fit_transform(fake["embeds"])
            plot_tsne_scatter_plot(fake, fake_tsne_results, "fake", self.run_name, self.logger, logging=True, cfgs=cfgs)

            generator = change_generator_mode(self.gen_model, self.Gen_copy, self.bn_stat_OnTheFly, standing_statistics, standing_step,
                                              self.prior, self.batch_size, self.z_dim, self.num_classes, self.local_rank, training=True, counter=self.counter)
    ################################################################################################################################
