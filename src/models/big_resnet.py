# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# models/big_resnet.py

from utils.model_ops import *
from utils.misc import *

import torch
import torch.nn as nn
import torch.nn.functional as F


class GenBlock(nn.Module):
    def __init__(self, in_channels, out_channels, g_spectral_norm, activation_fn, conditional_bn, z_dims_after_concat, sn_batchnorm):
        super(GenBlock, self).__init__()
        self.conditional_bn = conditional_bn

        if self.conditional_bn:
            self.bn1 = ConditionalBatchNorm2d_for_skip_and_shared(num_features=in_channels, z_dims_after_concat=z_dims_after_concat,
                                                                  spectral_norm=g_spectral_norm)
            self.bn2 = ConditionalBatchNorm2d_for_skip_and_shared(num_features=out_channels, z_dims_after_concat=z_dims_after_concat,
                                                                  spectral_norm=g_spectral_norm)
        else:
            self.bn1 = batchnorm_2d(in_features=in_channels)
            self.bn2 = batchnorm_2d(in_features=out_channels)

        if activation_fn == "ReLU":
            self.activation = nn.ReLU(inplace=True)
        elif activation_fn == "Leaky_ReLU":
            self.activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif activation_fn == "ELU":
            self.activation = nn.ELU(alpha=1.0, inplace=True)
        elif activation_fn == "GELU":
            self.activation = nn.GELU()
        else:
            raise NotImplementedError

        if g_spectral_norm:
            self.conv2d0 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
            self.conv2d1 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
            self.conv2d2 = snconv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        else:
            self.conv2d0 = conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
            self.conv2d1 = conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
            self.conv2d2 = conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)


    def forward(self, x, label):
        x0 = x
        if self.conditional_bn:
            x = self.bn1(x, label) # nans
        else:
            x = self.bn1(x)

        x = self.activation(x)
        x = F.interpolate(x, scale_factor=2, mode='nearest') # upsample
        x = self.conv2d1(x)
        if self.conditional_bn:
            x = self.bn2(x, label)
        else:
            x = self.bn2(x)
        x = self.activation(x)
        x = self.conv2d2(x)

        x0 = F.interpolate(x0, scale_factor=2, mode='nearest') # upsample
        x0 = self.conv2d0(x0)

        out = x + x0
        return out
    
    # Hook Added
    def _forward_hook(self, module, input, output):
        print('Forward Hook')
        print(type(input))
        print(len(input))
        print(type(output))
        print(input[0].shape)
        print(output.shape)
        print()

    def _backward_hook(self, module, grad_input, grad_output):
        print('Backward Hook')
        print(type(grad_input))
        print(len(grad_input))
        print(type(grad_output))
        print(len(grad_output))
        print(grad_input[0].shape)
        print(grad_input[1].shape)
        print(grad_output[0].shape)
        print()

    # Added
    def genblock_sn(self, embed, label, cfgs, noise_len):

        g1, b1, cn_g_1, cn_b_1, gamma_1, beta_1 = self.bn1.plot_sn(embed, label, cfgs, noise_len)
        g2, b2, cn_g_2, cn_b_2, gamma_2, beta_2 = self.bn2.plot_sn(embed, label, cfgs, noise_len)
        return g1, b1, g2, b2, cn_g_1, cn_b_1, cn_g_2, cn_b_2, gamma_1, beta_1, gamma_2, beta_2


class Generator(nn.Module):
    """Generator."""
    def __init__(self, z_dim, shared_dim, img_size, g_conv_dim, g_spectral_norm, attention, attention_after_nth_gen_block, activation_fn,
                 conditional_strategy, num_classes, initialize, G_depth, mixed_precision, sn_batchnorm=False):
        super(Generator, self).__init__()
        g_in_dims_collection = {"32": [g_conv_dim*4, g_conv_dim*4, g_conv_dim*4],
                                "64": [g_conv_dim*16, g_conv_dim*8, g_conv_dim*4, g_conv_dim*2],
                                "128": [g_conv_dim*16, g_conv_dim*16, g_conv_dim*8, g_conv_dim*4, g_conv_dim*2],
                                "256": [g_conv_dim*16, g_conv_dim*16, g_conv_dim*8, g_conv_dim*8, g_conv_dim*4, g_conv_dim*2],
                                "512": [g_conv_dim*16, g_conv_dim*16, g_conv_dim*8, g_conv_dim*8, g_conv_dim*4, g_conv_dim*2, g_conv_dim]}

        g_out_dims_collection = {"32": [g_conv_dim*4, g_conv_dim*4, g_conv_dim*4],
                                 "64": [g_conv_dim*8, g_conv_dim*4, g_conv_dim*2, g_conv_dim],
                                 "128": [g_conv_dim*16, g_conv_dim*8, g_conv_dim*4, g_conv_dim*2, g_conv_dim],
                                 "256": [g_conv_dim*16, g_conv_dim*8, g_conv_dim*8, g_conv_dim*4, g_conv_dim*2, g_conv_dim],
                                 "512": [g_conv_dim*16, g_conv_dim*8, g_conv_dim*8, g_conv_dim*4, g_conv_dim*2, g_conv_dim, g_conv_dim]}
        bottom_collection = {"32": 4, "64": 4, "128": 4, "256": 4, "512": 4}

        self.z_dim = z_dim
        self.shared_dim = shared_dim
        self.num_classes = num_classes
        self.mixed_precision = mixed_precision
        conditional_bn = True if conditional_strategy in ["ACGAN", "ProjGAN", "ContraGAN", "Proxy_NCA_GAN", "NT_Xent_GAN"] else False
        self.img_size = img_size
        self.in_dims =  g_in_dims_collection[str(img_size)]
        self.out_dims = g_out_dims_collection[str(img_size)]
        self.bottom = bottom_collection[str(img_size)]
        self.n_blocks = len(self.in_dims)
        self.chunk_size = z_dim//(self.n_blocks+1)
        self.z_dims_after_concat = self.chunk_size + self.shared_dim
        print(self.z_dim, self.n_blocks)
        assert self.z_dim % (self.n_blocks+1) == 0, "z_dim should be divided by the number of blocks "

        if g_spectral_norm:
            self.linear0 = snlinear(in_features=self.chunk_size, out_features=self.in_dims[0]*self.bottom*self.bottom)
        else:
            self.linear0 = linear(in_features=self.chunk_size, out_features=self.in_dims[0]*self.bottom*self.bottom)

        self.shared = embedding(self.num_classes, self.shared_dim) # culprit ?

        self.blocks = []
        for index in range(self.n_blocks):
            self.blocks += [[GenBlock(in_channels=self.in_dims[index],
                                      out_channels=self.out_dims[index],
                                      g_spectral_norm=g_spectral_norm,
                                      activation_fn=activation_fn,
                                      conditional_bn=conditional_bn,
                                      z_dims_after_concat=self.z_dims_after_concat,
                                      sn_batchnorm=sn_batchnorm)]]

            if index+1 == attention_after_nth_gen_block and attention is True:
                self.blocks += [[Self_Attn(self.out_dims[index], g_spectral_norm)]]

        self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])

        self.bn4 = batchnorm_2d(in_features=self.out_dims[-1])

        if activation_fn == "ReLU":
            self.activation = nn.ReLU(inplace=True)
        elif activation_fn == "Leaky_ReLU":
            self.activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif activation_fn == "ELU":
            self.activation = nn.ELU(alpha=1.0, inplace=True)
        elif activation_fn == "GELU":
            self.activation = nn.GELU()
        else:
            raise NotImplementedError

        if g_spectral_norm:
            self.conv2d5 = snconv2d(in_channels=self.out_dims[-1], out_channels=3, kernel_size=3, stride=1, padding=1)
        else:
            self.conv2d5 = conv2d(in_channels=self.out_dims[-1], out_channels=3, kernel_size=3, stride=1, padding=1)

        self.tanh = nn.Tanh()

        # Weight init
        if initialize is not False:
            init_weights(self.modules, initialize)

        # self.register_forward_hook(self._forward_hook) # Hook added
        # self.register_full_backward_hook(self._backward_hook)

    def forward(self, z, label, shared_label=None, evaluation=False, tmp=None):
        self.z = z
        with torch.cuda.amp.autocast() if self.mixed_precision is True and evaluation is False else dummy_context_mgr() as mp:
            zs = torch.split(z, self.chunk_size, 1)
            z = zs[0]
            if shared_label is None:
                shared_label = self.shared(label)
                # print(shared_label)
            else:
                pass
            # shared_label : 128, noise_chunk : 20
            labels = [torch.cat([shared_label, item], 1) for item in zs[1:]]
            # print(label.shape, shared_label.shape)
            act = self.linear0(z)
            act = act.view(-1, self.in_dims[0], self.bottom, self.bottom)
            counter = 0
            for index, blocklist in enumerate(self.blocks):
                for block in blocklist:
                    if isinstance(block, Self_Attn):
                        act = block(act)
                    else:
                        act = block(act, labels[counter]) #->1e38
                        counter +=1

            act = self.bn4(act) #->nans
            act = self.activation(act)
            act = self.conv2d5(act)
            out = self.tanh(act)
        return out
    
    # hook added
    def _forward_hook(self, module, input, output):
        print('Forward Hook')
        print(type(input))
        print(len(input))
        print(type(output))
        print(input[0].shape)
        print(output.shape)
        print()

    def _backward_hook(self, module, grad_input, grad_output):
        print('Backward Hook')
        print(type(grad_input))
        print(len(grad_input))
        print(type(grad_output))
        print(len(grad_output))
        print(grad_input)
        print(grad_output)
        if(grad_input[0] is not None):
            print(grad_input[0].shape)
            # print(grad_input[0].shape)
            print(grad_input[1].shape)
        print(grad_output[0].shape)
        print()
    

    #Added
    def gen_sn(self, label, cfgs, shared_label=None, z=None):
        """
        Computes Spectral Norm for the whole generator model.
        label : batch_size x num_classes
        cfgs : list of configs for each generator block
        shared_label : batch_size x shared_dim
        z: batch_size x z_dim
        """
        # label : [ 64 x 1 ]
        #z = self.z
        
        g_sn_for_all_blocks = [] ## Spectral Norm for gamma matrix in batch norm
        b_sn_for_all_blocks = [] ## Spectral Norm for beta matrix in batch norm
        cn_g_for_all_blocks = [] ## Condition Number for gain matrix
        cn_b_for_all_blocks = [] ## Condition Number for Bias Matrix ()
        gamma_all_blocks = [] ## gamma vector 
        beta_all_blocks = [] ## Beta vector

        with torch.cuda.amp.autocast() if self.mixed_precision is True else dummy_context_mgr() as mp:

            if shared_label is None:
                shared_label = self.shared(label)
            else:
                pass
            
            lbls = torch.tensor([i for i in range(cfgs.num_classes)], device=shared_label.device)

            # shrd_lbls: [ 100 x 128 ]
            shrd_embs = self.shared(lbls)
            # shared_label : 128, item : 20
            

            counter = 0
            for index, blocklist in enumerate(self.blocks):
                for block in blocklist:
                    if not isinstance(block, Self_Attn):
                        g1, b1, g2, b2, cn_g_1, cn_b_1, cn_g_2, cn_b_2, gamma_1, beta_1, gamma_2, beta_2  = block.genblock_sn(shrd_embs, label, cfgs, self.chunk_size)
                        
                        cn_g_for_all_blocks.append(cn_g_1)
                        cn_g_for_all_blocks.append(cn_g_2)
                        cn_b_for_all_blocks.append(cn_b_1)
                        cn_b_for_all_blocks.append(cn_b_2)
                        g_sn_for_all_blocks.append(g1)
                        g_sn_for_all_blocks.append(g2)
                        b_sn_for_all_blocks.append(b1)
                        b_sn_for_all_blocks.append(b2)
                        gamma_all_blocks.append(gamma_1)
                        gamma_all_blocks.append(gamma_2)
                        beta_all_blocks.append(beta_1)
                        beta_all_blocks.append(beta_2)
                        counter += 1

        return g_sn_for_all_blocks, b_sn_for_all_blocks, cn_g_for_all_blocks, cn_b_for_all_blocks, gamma_all_blocks, beta_all_blocks

class DiscOptBlock(nn.Module):
    def __init__(self, in_channels, out_channels, d_spectral_norm, activation_fn):
        super(DiscOptBlock, self).__init__()
        self.d_spectral_norm = d_spectral_norm

        if d_spectral_norm:
            self.conv2d0 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
            self.conv2d1 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
            self.conv2d2 = snconv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        else:
            self.conv2d0 = conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
            self.conv2d1 = conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
            self.conv2d2 = conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)

            self.bn0 = batchnorm_2d(in_features=in_channels)
            self.bn1 = batchnorm_2d(in_features=out_channels)

        if activation_fn == "ReLU":
            self.activation = nn.ReLU(inplace=True)
        elif activation_fn == "Leaky_ReLU":
            self.activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif activation_fn == "ELU":
            self.activation = nn.ELU(alpha=1.0, inplace=True)
        elif activation_fn == "GELU":
            self.activation = nn.GELU()
        else:
            raise NotImplementedError

        self.average_pooling = nn.AvgPool2d(2)

    
        # self.conv2d1.register_forward_hook(self._forward_hook) # Hook added
        # self.conv2d1.register_full_backward_hook(self._backward_hook)

     # hook added
    def _forward_hook(self, module, input, output):
        in_ = tuple([i.clone() for i in input])
        out_ = output.clone()

        # input = input.clone()
        # output = output.clone()
        print('Forward Hook')
        print(type(in_))
        print(len(in_))
        print(type(out_))
        print(in_[0].shape)
        print(out_.shape)
        print()

    def _backward_hook(self, module, grad_input, grad_output):
        grad_input = grad_input.clone()
        grad_output = grad_output.clone()
        print('Backward Hook')
        print(type(grad_input))
        print(len(grad_input))
        print(type(grad_output))
        print(len(grad_output))
        # print(grad_input.shape)
        print(grad_input[0].shape)
        # print(grad_input[0].shape)
        # print(grad_input[1].shape)
        print(grad_output[0].shape)
        print()


    def forward(self, x):
        x0 = x
        x = self.conv2d1(x)
        if self.d_spectral_norm is False:
            x = self.bn1(x)
        x = self.activation(x)
        x = self.conv2d2(x)
        x = self.average_pooling(x)

        x0 = self.average_pooling(x0)
        if self.d_spectral_norm is False:
            x0 = self.bn0(x0)
        x0 = self.conv2d0(x0)

        out = x + x0
        return out


class DiscBlock(nn.Module):
    def __init__(self, in_channels, out_channels, d_spectral_norm, activation_fn, downsample=True):
        super(DiscBlock, self).__init__()
        self.d_spectral_norm = d_spectral_norm
        self.downsample = downsample

        if activation_fn == "ReLU":
            self.activation = nn.ReLU(inplace=True)
        elif activation_fn == "Leaky_ReLU":
            self.activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif activation_fn == "ELU":
            self.activation = nn.ELU(alpha=1.0, inplace=True)
        elif activation_fn == "GELU":
            self.activation = nn.GELU()
        else:
            raise NotImplementedError

        self.ch_mismatch = False
        if in_channels != out_channels:
            self.ch_mismatch = True

        if d_spectral_norm:
            if self.ch_mismatch or downsample:
                self.conv2d0 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
            self.conv2d1 = snconv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
            self.conv2d2 = snconv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        else:
            if self.ch_mismatch or downsample:
                self.conv2d0 = conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
            self.conv2d1 = conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
            self.conv2d2 = conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)

            if self.ch_mismatch or downsample:
                self.bn0 = batchnorm_2d(in_features=in_channels)
            self.bn1 = batchnorm_2d(in_features=in_channels)
            self.bn2 = batchnorm_2d(in_features=out_channels)

        self.average_pooling = nn.AvgPool2d(2)
        
        # self.conv2d1.register_forward_hook(self._forward_hook) # Hook added for debugging
        # self.conv2d1.register_full_backward_hook(self._backward_hook)

     # hook added
    def _forward_hook(self, module, input, output):
        print('Forward Hook')
        print(type(input))
        print(len(input))
        print(type(output))
        print(input[0].shape)
        print(output.shape)
        print()

    def _backward_hook(self, module, grad_input, grad_output):
        print('Backward Hook')
        print(type(grad_input))
        print(len(grad_input))
        print(type(grad_output))
        print(len(grad_output))

        print(grad_input[0].shape)

        print(grad_output[0].shape)
        print()

    def forward(self, x):
        x0 = x

        if self.d_spectral_norm is False:
            x = self.bn1(x)
        x = self.activation(x)
        x = self.conv2d1(x)
        if self.d_spectral_norm is False:
            x = self.bn2(x)
        x = self.activation(x)
        x = self.conv2d2(x)
        if self.downsample:
            x = self.average_pooling(x)

        if self.downsample or self.ch_mismatch:
            if self.d_spectral_norm is False:
                x0 = self.bn0(x0)
            x0 = self.conv2d0(x0)
            if self.downsample:
                x0 = self.average_pooling(x0)

        out = x + x0
        return out


class Discriminator(nn.Module):
    """Discriminator."""
    def __init__(self, img_size, d_conv_dim, d_spectral_norm, attention, attention_after_nth_dis_block, activation_fn, conditional_strategy,
                 hypersphere_dim, num_classes, nonlinear_embed, normalize_embed, initialize, D_depth, mixed_precision, shared_dim):
        super(Discriminator, self).__init__()
        d_in_dims_collection = {"32": [3] + [d_conv_dim*2, d_conv_dim*2, d_conv_dim*2],
                                "64": [3] + [d_conv_dim, d_conv_dim*2, d_conv_dim*4, d_conv_dim*8],
                                "128": [3] +[d_conv_dim, d_conv_dim*2, d_conv_dim*4, d_conv_dim*8, d_conv_dim*16],
                                "256": [3] +[d_conv_dim, d_conv_dim*2, d_conv_dim*4, d_conv_dim*8, d_conv_dim*8, d_conv_dim*16],
                                "512": [3] +[d_conv_dim, d_conv_dim, d_conv_dim*2, d_conv_dim*4, d_conv_dim*8, d_conv_dim*8, d_conv_dim*16]}

        d_out_dims_collection = {"32": [d_conv_dim*2, d_conv_dim*2, d_conv_dim*2, d_conv_dim*2],
                                 "64": [d_conv_dim, d_conv_dim*2, d_conv_dim*4, d_conv_dim*8, d_conv_dim*16],
                                 "128": [d_conv_dim, d_conv_dim*2, d_conv_dim*4, d_conv_dim*8, d_conv_dim*16, d_conv_dim*16],
                                 "256": [d_conv_dim, d_conv_dim*2, d_conv_dim*4, d_conv_dim*8, d_conv_dim*8, d_conv_dim*16, d_conv_dim*16],
                                 "512": [d_conv_dim, d_conv_dim, d_conv_dim*2, d_conv_dim*4, d_conv_dim*8, d_conv_dim*8, d_conv_dim*16, d_conv_dim*16]}

        d_down = {"32": [True, True, False, False],
                  "64": [True, True, True, True, False],
                  "128": [True, True, True, True, True, False],
                  "256": [True, True, True, True, True, True, False],
                  "512": [True, True, True, True, True, True, True, False]}

        self.nonlinear_embed = nonlinear_embed
        self.normalize_embed = normalize_embed
        self.conditional_strategy = conditional_strategy
        self.mixed_precision = mixed_precision
        self.shared_dim = shared_dim
        self.num_classes = num_classes

        self.in_dims  = d_in_dims_collection[str(img_size)]
        self.out_dims = d_out_dims_collection[str(img_size)]
        down = d_down[str(img_size)]

        self.shared = embedding(self.num_classes, self.shared_dim)



        self.blocks = []
        for index in range(len(self.in_dims)):
            if index == 0:
                self.blocks += [[DiscOptBlock(in_channels=self.in_dims[index],
                                              out_channels=self.out_dims[index],
                                              d_spectral_norm=d_spectral_norm,
                                              activation_fn=activation_fn)]]
            else:
                self.blocks += [[DiscBlock(in_channels=self.in_dims[index],
                                           out_channels=self.out_dims[index],
                                           d_spectral_norm=d_spectral_norm,
                                           activation_fn=activation_fn,
                                           downsample=down[index])]]

            if index+1 == attention_after_nth_dis_block and attention is True:
                self.blocks += [[Self_Attn(self.out_dims[index], d_spectral_norm)]]

        self.blocks = nn.ModuleList([nn.ModuleList(block) for block in self.blocks])

        if activation_fn == "ReLU":
            self.activation = nn.ReLU(inplace=True)
        elif activation_fn == "Leaky_ReLU":
            self.activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif activation_fn == "ELU":
            self.activation = nn.ELU(alpha=1.0, inplace=True)
        elif activation_fn == "GELU":
            self.activation = nn.GELU()
        else:
            raise NotImplementedError

        if d_spectral_norm:
            self.linear1 = snlinear(in_features=self.out_dims[-1], out_features=1)
            if self.conditional_strategy in ['ContraGAN', 'Proxy_NCA_GAN', 'NT_Xent_GAN']:
                self.linear2 = snlinear(in_features=self.out_dims[-1], out_features=hypersphere_dim)
                if self.nonlinear_embed:
                    self.linear3 = snlinear(in_features=hypersphere_dim, out_features=hypersphere_dim)
                self.embedding = sn_embedding(num_classes, hypersphere_dim)
            elif self.conditional_strategy == 'ProjGAN':
                self.embedding = sn_embedding(num_classes, self.out_dims[-1])
            elif self.conditional_strategy == 'ACGAN':
                self.linear4 = snlinear(in_features=self.out_dims[-1], out_features=num_classes)
            else:
                pass
        else:
            self.linear1 = linear(in_features=self.out_dims[-1], out_features=1)
            if self.conditional_strategy in ['ContraGAN', 'Proxy_NCA_GAN', 'NT_Xent_GAN']:
                self.linear2 = linear(in_features=self.out_dims[-1], out_features=hypersphere_dim)
                if self.nonlinear_embed:
                    self.linear3 = linear(in_features=hypersphere_dim, out_features=hypersphere_dim)
                self.embedding = embedding(num_classes, hypersphere_dim)
            elif self.conditional_strategy == 'ProjGAN':
                self.embedding = embedding(num_classes, self.out_dims[-1])
            elif self.conditional_strategy == 'ACGAN':
                self.linear4 = linear(in_features=self.out_dims[-1], out_features=num_classes)
            else:
                pass
        
        # self.register_forward_hook(self._forward_hook) # Hook added
        # self.register_full_backward_hook(self._backward_hook)

        # Weight init
        if initialize is not False:
            init_weights(self.modules, initialize)

        with torch.no_grad():
            
            self.u0 = None
            self.v0 = None
        
        self.n_power_iterations = 1
        self.h, self.w = self.get_dimensions(self.out_dims[-1])
    
    def get_dimensions(self, x):
        a, b = 0, np.inf
        for i in range(int(np.sqrt(x)+1)):
            if(i==0):
                continue
            if((x/i).is_integer()):
                if(i>a and (x/i)<b):
                    a = i
                    b = int(x/i)
        return a, b

    # hook added
    def _forward_hook(self, module, input, output):
        print('Forward Hook')
        print(type(input))
        print(len(input))
        print(type(output))
        print(input[0].shape)
        print(output.shape)
        print()

    def _backward_hook(self, module, grad_input, grad_output):
        print('Backward Hook')
        print(type(grad_input))
        print(len(grad_input))
        print(type(grad_output))
        print(len(grad_output))
        print(grad_input)
        print(grad_output)
        if(grad_input[0] is not None):
            print(grad_input[0].shape)
            # print(grad_input[0].shape)
            print(grad_input[1].shape)
        print(grad_output[0].shape)
        print()


    def forward(self, x, label = None, evaluation=False):
        with torch.cuda.amp.autocast() if self.mixed_precision is True and evaluation is False else dummy_context_mgr() as mp:
            h = x
            for index, blocklist in enumerate(self.blocks):
                for block in blocklist:
                    h = block(h)
            h = self.activation(h)
            h = torch.sum(h, dim=[2,3])     # h : [ 64, 1536 ]

            if self.conditional_strategy == 'no':
                authen_output = torch.squeeze(self.linear1(h))
                return authen_output

            elif self.conditional_strategy in ['ContraGAN', 'Proxy_NCA_GAN', 'NT_Xent_GAN']:
                authen_output = torch.squeeze(self.linear1(h))
                cls_proxy = self.embedding(label)
                cls_embed = self.linear2(h)
                if self.nonlinear_embed:
                    cls_embed = self.linear3(self.activation(cls_embed))# sigma_max_1 = torch.max(diag_1, dim=-1)[0]
                if self.normalize_embed:
                    cls_proxy = F.normalize(cls_proxy, dim=1)
                    cls_embed = F.normalize(cls_embed, dim=1)
                return cls_proxy, cls_embed, authen_output

            elif self.conditional_strategy == 'ProjGAN':
                authen_output = torch.squeeze(self.linear1(h))
                proj = torch.sum(torch.mul(self.embedding(label), h), 1)
                outputs =  proj + authen_output
                # return outputs   
                return outputs.unsqueeze(1)
            elif self.conditional_strategy == 'ACGAN':
                authen_output = torch.squeeze(self.linear1(h))
                cls_output = self.linear4(h)
                return cls_output, authen_output

            else:
                raise NotImplementedError

    



class MLPD(nn.Module):
    def __init__(self, in_ch=768, out_ch=256):
        super().__init__()

        self.decoder = nn.Sequential(nn.Linear(in_ch, out_ch), nn.LeakyReLU(),
                                                  nn.Linear(out_ch, 1))

    def forward(self, input):
        
        return self.decoder(input)



