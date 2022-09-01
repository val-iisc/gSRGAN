# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# src/utils/model_ops.py


import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
from torch.nn import init
import numpy as np
from utils.misc import *
import torch.nn.functional as F
from torch.nn.functional import normalize
# import torch_batch_svd



def init_weights(modules, initialize):
    for module in modules():
        if (isinstance(module, nn.Conv2d)
                or isinstance(module, nn.ConvTranspose2d)
                or isinstance(module, nn.Linear)):
            if initialize == 'ortho':
                init.orthogonal_(module.weight)
                if module.bias is not None:
                    module.bias.data.fill_(0.)
            elif initialize == 'N02':
                init.normal_(module.weight, 0, 0.02)
                if module.bias is not None:
                    module.bias.data.fill_(0.)
            elif initialize in ['glorot', 'xavier']:
                init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    module.bias.data.fill_(0.)
            else:
                print('Init style not recognized...')
        elif isinstance(module, nn.Embedding):
            if initialize == 'ortho':
                init.orthogonal_(module.weight)
            elif initialize == 'N02':
                init.normal_(module.weight, 0, 0.02)
            elif initialize in ['glorot', 'xavier']:
                init.xavier_uniform_(module.weight)
            else:
                print('Init style not recognized...')
        else:
            pass


def conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                     stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

def deconv2d(in_channels, out_channels, kernel_size, stride=2, padding=0, dilation=1, groups=1, bias=True):
    return nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

def linear(in_features, out_features, bias=True):
    return nn.Linear(in_features=in_features, out_features=out_features, bias=bias)

def embedding(num_embeddings, embedding_dim):
    return nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)

def snconv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
    return spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias), eps=1e-6)

def sndeconv2d(in_channels, out_channels, kernel_size, stride=2, padding=0, dilation=1, groups=1, bias=True):
    return spectral_norm(nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                            stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias), eps=1e-6)

def snlinear(in_features, out_features, bias=True):
    return spectral_norm(nn.Linear(in_features=in_features, out_features=out_features, bias=bias), eps=1e-6)

def sn_embedding(num_embeddings, embedding_dim):
    return spectral_norm(nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim), eps=1e-6)

def batchnorm_2d(in_features, eps=1e-4, momentum=0.1, affine=True):
    return nn.BatchNorm2d(in_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=False) # Change breaking

def get_dimensions(x):
        a, b = 0, np.inf
        for i in range(int(np.sqrt(x)+1)):
            if(i==0):
                continue
            if((x/i).is_integer()):
                if(i>a and (x/i)<b):
                    a = i
                    b = int(x/i)
        return a, b

class ConditionalBatchNorm2d(nn.Module):
    # https://github.com/voletiv/self-attention-GAN-pytorch
    def __init__(self, num_features, num_classes, spectral_norm):
        super().__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.bn = batchnorm_2d(num_features, eps=1e-4, momentum=0.1, affine=False)
        self.n_power_iterations = 1
        # Spectral norm of Embedding [always on if TRUE]
        if spectral_norm:
            self.embed0 = sn_embedding(num_classes, num_features)
            self.embed1 = sn_embedding(num_classes, num_features)
        else:
            self.embed0 = embedding(num_classes, num_features)
            self.embed1 = embedding(num_classes, num_features)
            
        self.h, self.w = get_dimensions(self.num_features)

        with torch.no_grad():
            
            self.u0 = None
            self.v0 = None
            self.u1 = None
            self.v1 = None

    def forward(self, x, y):

        embed0_y = self.embed0(y)
        embed1_y = self.embed1(y)
        
        gain = (1 + embed0_y).view(-1, self.num_features, 1, 1)
        bias = embed1_y.view(-1, self.num_features, 1, 1)
        out = self.bn(x)
        return out * gain + bias

    def plot_sn(self, y, cfgs):
        """
        Main Function for calculation of SN Norm for the generator. (SNGAN)

        y: class labels
        cfgs: config files
        
        """
        # emb_0 : gain,     emb_1 : bias
        #lbls = torch.tensor([i for i in range(cfgs.num_classes)], device='cuda')
        emb_0 = self.embed0.weight
        emb_1 = self.embed1.weight

        # fro_emb_0 = torch.norm(emb_0, p='fro')
        # fro_emb_1 = torch.norm(emb_1, p='fro')
        
        if cfgs.shuffle_emb:
            emb_0 = emb_0[:, torch.randperm(emb_0.size()[1])]
            emb_1 = emb_1[:, torch.randperm(emb_1.size()[1])]

        gamma = emb_0
        beta = emb_1
        
        # Add power iteration code
        if cfgs.num_power_iter > 0:
            num_iter = cfgs.num_power_iter
            if self.u0 is None:
                self.u0 = torch.randn((self.num_classes, self.h)).to(y.device).unsqueeze(2)
                self.v0 = torch.randn((self.num_classes, self.w)).to(y.device).unsqueeze(2)
                self.u1 = torch.randn((self.num_classes, self.h)).to(y.device).unsqueeze(2)
                self.v1 = torch.randn((self.num_classes, self.w)).to(y.device).unsqueeze(2)
                num_iter = 10 * cfgs.num_power_iter # Based on NVAE

            with torch.no_grad():
                u0 = self.u0
                v0 = self.v0
                u1 = self.u1
                v1 = self.v1
                

                for _ in range(num_iter):
                    # Spectral norm of weight equals to `u^T W v`, where `
                    # u` and `v`
                    # are the first left and right singular vectors.
                    # This power iteration produces approximations of `u` and `v`.
                    
                    v0 = normalize(torch.matmul(emb_0.reshape((-1,self.h, self.w)).permute(0, 2, 1), u0), dim=1, eps=1e-3, out=v0)
                    u0 = normalize(torch.matmul(emb_0.reshape((-1,self.h, self.w)), v0), dim=1, eps=1e-3, out=u0)
                    v1 = normalize(torch.matmul(emb_1.reshape((-1,self.h, self.w)).permute(0, 2, 1), u1), dim=1, eps=1e-3, out=v1)
                    u1 = normalize(torch.matmul(emb_1.reshape((-1,self.h, self.w)), v0), dim=1, eps=1e-3, out=u1)

                if num_iter > 0:
                    # See above on why we need to clone
                    u0 = u0.clone(memory_format=torch.contiguous_format)
                    v0 = v0.clone(memory_format=torch.contiguous_format)
                    u1 = u1.clone(memory_format=torch.contiguous_format)
                    v1 = v1.clone(memory_format=torch.contiguous_format)

            sigma1 = (u1.squeeze(2) * torch.matmul(emb_1.reshape((-1,self.h, self.w)), v1).squeeze(2)).sum(-1)
            sigma0 = (u0.squeeze(2) * torch.matmul(emb_0.reshape((-1,self.h, self.w)), v0).squeeze(2)).sum(-1)

            tmp_sn_0,tmp_sn_1, condition_number_0, condition_number_1 = sigma0, sigma1, 0, 0 
        else:

            a, b = get_dimensions(self.num_features)
            

            _, diag_0, _ = torch.linalg.svd(emb_0.reshape(-1,a,b).double())
            _, diag_1, _ = torch.linalg.svd(emb_1.reshape(-1,a,b).double())



            if cfgs.sn_regularize:
                tmp_sn_0 = torch.max(diag_0, dim=-1)[0]
                tmp_sn_1 = torch.max(diag_1, dim=-1)[0]
            else:
                tmp_sn_0 = torch.max(diag_0, dim=-1)[0].detach().cpu().numpy()
                tmp_sn_1 = torch.max(diag_1, dim=-1)[0].detach().cpu().numpy()
            
            with torch.no_grad():
                sigma_max_0 = torch.max(diag_0, dim=-1)[0]
                sigma_max_1 = torch.max(diag_1, dim=-1)[0]
                sigma_min_0 = torch.min(diag_0, dim=-1)[0]
                sigma_min_1 = torch.min(diag_1, dim=-1)[0]

                condition_number_0 = (sigma_max_0/sigma_min_0)
                condition_number_1 = (sigma_max_1/sigma_min_1)

        

        return tmp_sn_0, tmp_sn_1, condition_number_0, condition_number_1, gamma, beta
        

class ConditionalBatchNorm2d_for_skip_and_shared(nn.Module):
    # https://github.com/voletiv/self-attention-GAN-pytorch
    def __init__(self, num_features, z_dims_after_concat, spectral_norm):
        super().__init__()
        self.num_features = num_features
        self.bn = batchnorm_2d(num_features, eps=1e-4, momentum=0.1, affine=False)
        print(num_features)

        if spectral_norm:
            self.gain = snlinear(z_dims_after_concat, num_features, bias=False)
            self.bias = snlinear(z_dims_after_concat, num_features, bias=False)
        else:
            self.gain = linear(z_dims_after_concat, num_features, bias=False)
            self.bias = linear(z_dims_after_concat, num_features, bias=False)

        with torch.no_grad():
            
            self.u0 = None
            self.v0 = None
            self.u1 = None
            self.v1 = None
        

    def forward(self, x, y):
        # x : [ bs , channel, H, W ]
        gain = (1 + self.gain(y)).view(y.size(0), -1, 1, 1)
        # gain/bias : [ bs , channel, 1, 1 ]
        bias = self.bias(y).view(y.size(0), -1, 1, 1)
        out = self.bn(x) 
        return out * gain + bias



    def plot_sn(self, embed, y, cfgs, noise_len=None):
        """
        Main Function for calculation of SN Norm for the generator.

        embed: embedding matrix for SN Norm calculation
        y: class labels
        cfgs: config files
        noise_len: length of chunk used for GAN conditioning in bigGAN
        """

        # embed : 100 x 128
        # emb_0 : gain,     emb_1 : bias
        # embed : 100 X 148 [ zero-padded ]
        embed = F.pad(input=embed, pad=(0,noise_len,0,0), mode='constant', value=0)
        
        emb_0 = self.gain(embed)
        emb_1 = self.bias(embed)

        if cfgs.shuffle_emb:
            emb_0 = emb_0[:, torch.randperm(emb_0.size()[1])]
            emb_1 = emb_1[:, torch.randperm(emb_1.size()[1])]
        
        if cfgs.num_power_iter > 0: # Calculate SN Norm by Power Iteration (It's significantly faster than SVD)
            num_iter = cfgs.num_power_iter
            if self.u0 is None:
                self.h, self.w = get_dimensions(emb_1.shape[1])
                self.u0 = torch.randn((cfgs.num_classes, self.h)).to(y.device).unsqueeze(2)
                self.v0 = torch.randn((cfgs.num_classes, self.w)).to(y.device).unsqueeze(2)
                self.u1 = torch.randn((cfgs.num_classes, self.h)).to(y.device).unsqueeze(2)
                self.v1 = torch.randn((cfgs.num_classes, self.w)).to(y.device).unsqueeze(2)
                num_iter = 10 * cfgs.num_power_iter # Based on NVAE


            with torch.no_grad():
                    u0 = self.u0
                    v0 = self.v0
                    u1 = self.u1
                    v1 = self.v1
                    
                    
                    for _ in range(num_iter):
                        # Spectral norm of weight equals to `u^T W v`, where `
                        # u` and `v`
                        # are the first left and right singular vectors.
                        # This power iteration produces approximations of `u` and `v`.
                        #print(self.embed0.weight.reshape((-1,self.h, self.w)).permute(0, 2, 1).shape, u0.shape)
                        v0 = normalize(torch.matmul(emb_0.reshape((-1,self.h, self.w)).permute(0, 2, 1), u0), dim=1, eps=1e-3, out=v0)
                        u0 = normalize(torch.matmul(emb_0.reshape((-1,self.h, self.w)), v0), dim=1, eps=1e-3, out=u0)
                        v1 = normalize(torch.matmul(emb_1.reshape((-1,self.h, self.w)).permute(0, 2, 1), u1), dim=1, eps=1e-3, out=v1)
                        u1 = normalize(torch.matmul(emb_1.reshape((-1,self.h, self.w)), v0), dim=1, eps=1e-3, out=u1)

                    if num_iter > 0:
                        # See above on why we need to clone
                        u0 = u0.clone(memory_format=torch.contiguous_format)
                        v0 = v0.clone(memory_format=torch.contiguous_format)
                        u1 = u1.clone(memory_format=torch.contiguous_format)
                        v1 = v1.clone(memory_format=torch.contiguous_format)

                    
            

            sigma1 = (u1.squeeze(2) * torch.matmul(emb_1.reshape((-1,self.h, self.w)), v1).squeeze(2)).sum(-1)
            sigma0 = (u0.squeeze(2) * torch.matmul(emb_0.reshape((-1,self.h, self.w)), v0).squeeze(2)).sum(-1)
            # Condition Number calculation is disabled for power iteration based calculation
            tmp_sn_0,tmp_sn_1, condition_number_0, condition_number_1 = sigma0, sigma1, 0, 0 
        else:
            # Calculate of SN Norm based on the SVD Based Method
            a, b = get_dimensions(emb_0.shape[1])

            # diag_i : [ 100 x 16 ]
            _, diag_0, _ = torch.linalg.svd(emb_0.reshape(-1,a,b).double()) # lsun added: 16,24[cifar-100] -> 30,32 [lsun]
            _, diag_1, _ = torch.linalg.svd(emb_1.reshape(-1,a,b).double())

            # tmp_sn_i : [ 100 ]
            if cfgs.sn_regularize: 
                tmp_sn_0 = torch.max(diag_0, dim=-1)[0]
                tmp_sn_1 = torch.max(diag_1, dim=-1)[0]
            else:
                tmp_sn_0 = torch.max(diag_0, dim=-1)[0].detach().cpu().numpy()
                tmp_sn_1 = torch.max(diag_1, dim=-1)[0].detach().cpu().numpy()

            # with torch.no_grad():
            sigma_max_0 = torch.max(diag_0, dim=-1)[0]
            sigma_max_1 = torch.max(diag_1, dim=-1)[0]
            sigma_min_0 = torch.min(diag_0, dim=-1)[0]
            sigma_min_1 = torch.min(diag_1, dim=-1)[0]

            condition_number_0 = (sigma_max_0/sigma_min_0)
            condition_number_1 = (sigma_max_1/sigma_min_1)


        return tmp_sn_0, tmp_sn_1, condition_number_0, condition_number_1, emb_0, emb_1
    

class Self_Attn(nn.Module):
    # https://github.com/voletiv/self-attention-GAN-pytorch
    def __init__(self, in_channels, spectral_norm):
        super(Self_Attn, self).__init__()
        self.in_channels = in_channels

        if spectral_norm:
            self.conv1x1_theta = snconv2d(in_channels=in_channels, out_channels=in_channels//8, kernel_size=1, stride=1, padding=0, bias=False)
            self.conv1x1_phi = snconv2d(in_channels=in_channels, out_channels=in_channels//8, kernel_size=1, stride=1, padding=0, bias=False)
            self.conv1x1_g = snconv2d(in_channels=in_channels, out_channels=in_channels//2, kernel_size=1, stride=1, padding=0, bias=False)
            self.conv1x1_attn = snconv2d(in_channels=in_channels//2, out_channels=in_channels, kernel_size=1, stride=1, padding=0, bias=False)
        else:
            self.conv1x1_theta = conv2d(in_channels=in_channels, out_channels=in_channels//8, kernel_size=1, stride=1, padding=0, bias=False)
            self.conv1x1_phi = conv2d(in_channels=in_channels, out_channels=in_channels//8, kernel_size=1, stride=1, padding=0, bias=False)
            self.conv1x1_g = conv2d(in_channels=in_channels, out_channels=in_channels//2, kernel_size=1, stride=1, padding=0, bias=False)
            self.conv1x1_attn = conv2d(in_channels=in_channels//2, out_channels=in_channels, kernel_size=1, stride=1, padding=0, bias=False)

        self.maxpool = nn.MaxPool2d(2, stride=2, padding=0)
        self.softmax  = nn.Softmax(dim=-1)
        self.sigma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """
            inputs :
                x : input feature maps(B X C X H X W)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        _, ch, h, w = x.size()
        # Theta path
        theta = self.conv1x1_theta(x)
        theta = theta.view(-1, ch//8, h*w)
        # Phi path
        phi = self.conv1x1_phi(x)
        phi = self.maxpool(phi)
        phi = phi.view(-1, ch//8, h*w//4)
        # Attn map
        attn = torch.bmm(theta.permute(0, 2, 1), phi)
        attn = self.softmax(attn)
        # g path
        g = self.conv1x1_g(x)
        g = self.maxpool(g)
        g = g.view(-1, ch//2, h*w//4)
        # Attn_g
        attn_g = torch.bmm(g, attn.permute(0, 2, 1))
        attn_g = attn_g.view(-1, ch//2, h, w)
        attn_g = self.conv1x1_attn(attn_g)
        
        return x + self.sigma*attn_g

def find_modules(nn_module, type):
    return [module for module in nn_module.modules() if isinstance(module, type)]

class RecorderBigGAN(nn.Module):
    def __init__(self, disc, device = None):
        super().__init__()
        self.discriminator = disc

        self.data = None
        self.recordings = []
        self.hooks = []
        self.hook_registered = False
        self.ejected = False
        self.device = device

    def _hook(self, _, input, output):
        self.recordings.append(output.clone().detach())

    def _register_hook(self):
        modules = find_modules(self.discriminator, Self_Attn)
        for module in modules:
            handle = module.softmax.register_forward_hook(self._hook)
            self.hooks.append(handle)
        self.hook_registered = True

    def eject(self):
        self.ejected = True
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        return self.discriminator

    def clear(self):
        self.recordings.clear()

    def record(self, attn):
        recording = attn.clone().detach()
        self.recordings.append(recording)

    def forward(self, img, labels = None):
        assert not self.ejected, 'recorder has been ejected, cannot be used anymore'
        self.clear()
        if not self.hook_registered:
            self._register_hook()
        
        pred = self.discriminator(img, labels)

        # move all recordings to one device before stacking
        target_device = self.device if self.device is not None else img.device
        recordings = tuple(map(lambda t: t.to(target_device), self.recordings))
       
        attns = torch.stack(recordings, dim = 0)
        return pred, attns