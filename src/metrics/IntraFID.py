import os
import math

import numpy as np
from tqdm import tqdm
import torch
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel

from metrics.FID import generate_images
from metrics.FID import calculate_frechet_distance


def get_activations_with_label(data_loader, generator, discriminator, inception_model, n_generate, truncated_factor, prior, is_generate,
                    latent_op, latent_op_step, latent_op_alpha, latent_op_beta, device, tqdm_disable=False, run_name=None):
    """Calculates the activations of the pool_3 layer for all images.
    Params:
    -- data_loader      : data_loader of training images
    -- generator        : instance of GANs' generator
    -- inception_model  : Instance of inception model

    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    if is_generate is True:
        batch_size = data_loader.batch_size
        total_instance = n_generate
        n_batches = math.ceil(float(total_instance) / float(batch_size))
    else:
        batch_size = data_loader.batch_size
        if data_loader.sampler is None:
            total_instance = len(data_loader.dataset)
        else:
            total_instance = data_loader.sampler.length
        
        n_batches = math.ceil(float(total_instance) / float(batch_size))
        data_iter = iter(data_loader)
    
    
    num_classes = generator.module.num_classes if isinstance(generator, DataParallel) or isinstance(generator, DistributedDataParallel) else generator.num_classes
    pred_arr = np.empty((total_instance, 2048))
    label_arr = []

    for i in tqdm(range(0, n_batches), disable=tqdm_disable):
        start = i*batch_size
        end = start + batch_size
        if is_generate is True:
            images, labels = generate_images(batch_size, generator, discriminator, truncated_factor, prior, latent_op,
                                             latent_op_step, latent_op_alpha, latent_op_beta, device)
            images = images.to(device)

            with torch.no_grad():
                embeddings, logits = inception_model(images)

            if total_instance >= batch_size:
                pred_arr[start:end] = embeddings.cpu().data.numpy().reshape(batch_size, -1)
            else:
                pred_arr[start:] = embeddings[:total_instance].cpu().data.numpy().reshape(total_instance, -1)

            total_instance -= images.shape[0]
        else:
            try:
                feed_list = next(data_iter)
                images = feed_list[0]
                labels = feed_list[1]
                images = images.to(device)
                with torch.no_grad():
                    embeddings, logits = inception_model(images)

                if total_instance >= batch_size:
                    pred_arr[start:end] = embeddings.cpu().data.numpy().reshape(batch_size, -1)
                else:
                    pred_arr[start:] = embeddings[:total_instance].cpu().data.numpy().reshape(total_instance, -1)
                total_instance -= images.shape[0]

            except StopIteration:
                label_arr.append(labels[:total_instance].cpu().data.numpy())
                break
        label_arr.append(labels.cpu().data.numpy())
    label_arr = np.concatenate(label_arr)[:len(pred_arr)]
    return pred_arr, label_arr


def calculate_intra_activation_statistics(data_loader, generator, discriminator, inception_model, n_generate, truncated_factor, prior,
                                    is_generate, latent_op, latent_op_step, latent_op_alpha, latent_op_beta, device, tqdm_disable, run_name=None,method="fine"):
    act, labels = get_activations_with_label(data_loader, generator, discriminator, inception_model, n_generate, truncated_factor, prior,
                          is_generate, latent_op, latent_op_step, latent_op_alpha, latent_op_beta, device, tqdm_disable, run_name)
    num_classes = len(np.unique(labels))
    mu, sigma = [], []
    if method == "coarse":
        img_num_list = data_loader.dataset.img_num_list
        
        many, mid, few = np.empty(shape=(0, act[0].shape[0])), np.empty(shape=(0, act[0].shape[0])), np.empty(shape=(0, act[0].shape[0]))
        for i in range(num_classes):
            if img_num_list[i] > 100:
                many = np.append(many, act[labels == i], axis=0)
            elif img_num_list[i] <= 100 and img_num_list[i] > 40:
                mid = np.append(mid, act[labels == i], axis=0)
            else:
                few = np.append(few, act[labels == i], axis=0)
        # Calc stats
        
        mu.append(np.mean(many, axis=0))
        mu.append(np.mean(mid, axis=0))
        mu.append(np.mean(few, axis=0))
        sigma.append(np.cov(many, rowvar=False))
        sigma.append(np.cov(mid, rowvar=False))
        sigma.append(np.cov(few, rowvar=False))
    
    else:
        for i in tqdm(range(num_classes)):
            mu.append(np.mean(act[labels == i], axis=0))
            sigma.append(np.cov(act[labels == i], rowvar=False))
    return mu, sigma


def calculate_intra_fid_score(data_loader, generator, discriminator, inception_model, n_generate, truncated_factor, prior,
                        latent_op, latent_op_step, latent_op_alpha, latent_op_beta, device, logger, pre_cal_mean=None, pre_cal_std=None, run_name=None, method="fine"):
    disable_tqdm = device != 0
    inception_model.eval()

    if device == 0: logger.info("Calculating Intra-FID Score....")
    if pre_cal_mean is not None and pre_cal_std is not None and False:
        m1, s1 = pre_cal_mean, pre_cal_std
    else:
        m1, s1 = calculate_intra_activation_statistics(data_loader, generator, discriminator, inception_model, n_generate, truncated_factor,
                                                 prior, False, False, 0, latent_op_alpha, latent_op_beta, device, tqdm_disable=disable_tqdm, method=method)

    m2, s2 = calculate_intra_activation_statistics(data_loader, generator, discriminator, inception_model, n_generate, truncated_factor, prior,
                                             True, latent_op, latent_op_step, latent_op_alpha, latent_op_beta, device, tqdm_disable=disable_tqdm, run_name=run_name, method=method)

    # print(len(m1), len(m2[0])) 2048 2048
    intra_fid = []
    for i in tqdm(range(len(m1))):
        intra_fid.append(calculate_frechet_distance(m1[i], s1[i], m2[i], s2[i]))
        #logger.info("Class {i} : Intra FID Score {f}".format(i = i, f = intra_fid[-1]))
    for i in tqdm(range(len(m1))):
        logger.info("Class {i} : Intra FID Score {f}".format(i = i, f = intra_fid[i]))
    
    intra_fid = np.mean(intra_fid)

    return intra_fid, m1, s1