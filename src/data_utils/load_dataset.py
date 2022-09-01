# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# src/data_utils/load_dataset.py


import os
import json
import h5py as h5
import numpy as np
import random
from scipy import io
from PIL import ImageOps, Image
from collections import Counter

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, STL10, CIFAR100, LSUN
from torchvision.datasets import ImageFolder



class RandomCropLongEdge(object):
    """
    this code is borrowed from https://github.com/ajbrock/BigGAN-PyTorch
    MIT License
    Copyright (c) 2019 Andy Brock
    """
    def __call__(self, img):
        size = (min(img.size), min(img.size))
        # Only step forward along this edge if it's the long edge
        i = (0 if size[0] == img.size[0]
            else np.random.randint(low=0,high=img.size[0] - size[0]))
        j = (0 if size[1] == img.size[1]
            else np.random.randint(low=0,high=img.size[1] - size[1]))
        return transforms.functional.crop(img, j, i, size[0], size[1])

    def __repr__(self):
        return self.__class__.__name__


class CenterCropLongEdge(object):
    """
    this code is borrowed from https://github.com/ajbrock/BigGAN-PyTorch
    MIT License
    Copyright (c) 2019 Andy Brock
    """
    def __call__(self, img):
        return transforms.functional.center_crop(img, min(img.size))

    def __repr__(self):
        return self.__class__.__name__


class LT_Dataset(Dataset):
    
    def __init__(self, root, txt, transform=None):
        self.img_path = []
        self.labels = []
        self.transform = transform
        with open(txt) as f:
            for line in f:
                if "iNat17" in root:
                    self.img_path.append(os.path.join(root, line.split('\t')[0]))
                    self.labels.append(int(line.split('\t')[1]))
                else:
                    self.img_path.append(os.path.join(root, line.split()[0]))
                    self.labels.append(int(line.split()[1]))
        # import glob
        # for name in glob.glob("./images_inat/*.jpg"):
        #     self.img_path.append(os.path.join(name))
        #     print(name.split("/"))
        #     self.labels.append(int(name.split("/")[-1][:3]))
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, index):

        path = self.img_path[index]
        label = self.labels[index]
        
        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label


class LoadDataset(Dataset):
    def __init__(self, dataset_name, data_path, train, download, resize_size, hdf5_path=None, random_flip=False, cfgs=None):
        super(LoadDataset, self).__init__()
        self.dataset_name = dataset_name
        self.data_path = data_path
        self.train = train
        self.download = download
        self.resize_size = resize_size
        self.hdf5_path = hdf5_path
        self.random_flip = random_flip
        self.norm_mean = [0.5,0.5,0.5]
        self.norm_std = [0.5,0.5,0.5]
        self.img_num_list = None
        self.transforms = []
        self.cfgs = cfgs

        if self.hdf5_path is None:
            if self.dataset_name in ['cifar10', 'tiny_imagenet', 'cifar100']:
                self.transforms = []
            elif self.dataset_name in ['imagenet']:
                if train:
                    self.transforms = [RandomCropLongEdge(), transforms.Resize(self.resize_size)]
                else:
                    self.transforms = [CenterCropLongEdge(), transforms.Resize(self.resize_size)]
            if self.dataset_name in ["lsun", 'custom'] : #Added
                self.transforms = [transforms.Resize((self.resize_size, self.resize_size))]
            if self.dataset_name == "inaturalist2019" or self.dataset_name == "imagenet_lt" or self.dataset_name == "inaturalist2017":
                self.transforms = [CenterCropLongEdge(), transforms.Resize((self.resize_size, self.resize_size))]
        else:
            self.transforms = [transforms.ToPILImage()]

        if random_flip:
            self.transforms += [transforms.RandomHorizontalFlip()]

        self.transforms += [transforms.ToTensor(), transforms.Normalize(self.norm_mean, self.norm_std)]
        self.transforms = transforms.Compose(self.transforms)

        self.load_dataset()

    def load_dataset(self):
        self.labels=None

        if self.dataset_name == 'cifar10':
            if self.hdf5_path is not None:
                print('Loading %s into memory...' % self.hdf5_path)
                with h5.File(self.hdf5_path, 'r') as f:
                    self.data = f['imgs'][:]
                    self.labels = f['labels'][:]

            else:
                self.data = CIFAR10(root=os.path.join('data', self.dataset_name),
                                    train=self.train,
                                    download=self.download)

        elif self.dataset_name == 'cifar100':
            if self.hdf5_path is not None:
                print('Loading %s into memory...' % self.hdf5_path)
                with h5.File(self.hdf5_path, 'r') as f:
                    self.data = f['imgs'][:]
                    self.labels = f['labels'][:]
            else:
                self.data = CIFAR100(root=os.path.join('data', self.dataset_name),
                                    train=self.train,
                                    download=self.download)

        elif self.dataset_name == "lsun": #Added
            if self.hdf5_path is not None:
                print('Loading %s into memory...' % self.hdf5_path)
                with h5.File(self.hdf5_path, 'r') as f:
                    self.data = f['imgs'][:]
                    self.labels = f['labels'][:]
                
                #self.img_num_list = self.data.img_num_list
                counts = Counter(self.labels)
                self.img_num_list = [0] * len(counts)
                for i in range(len(counts)):
                    self.img_num_list[i] = counts[i]
            
            else:
                # lsun_classes = [ "bedroom_train", "conference_room_train", "dining_room_train", "kitchen_train", "living_room_train"]
                # self.data = LSUN(root=self.data_path,
                #                     # train=self.train,
                #                     # download=self.download)
                #                     classes=lsun_classes)
                lsun_classes = [ "bedroom_train", "conference_room_train", "dining_room_train", "kitchen_train", "living_room_train"]
                # lsun_classes = ["bedroom_train"]
                if self.train == True:
                    self.data = IMBALANCELSUN(root=self.data_path, classes=lsun_classes, imb_factor=self.cfgs.imb_factor,
                                            max_samples = 50000)
                else:
                    self.data = IMBALANCELSUN(root=self.data_path, imb_factor=1.0, classes=lsun_classes,
                                            max_samples = 2000)

            


        elif self.dataset_name == 'imagenet':
            if self.hdf5_path is not None:
                print('Loading %s into memory...' % self.hdf5_path)
                with h5.File(self.hdf5_path, 'r') as f:
                    self.data = f['imgs'][:]
                    self.labels = f['labels'][:]
            else:
                mode = 'train' if self.train == True else 'valid'
                root = os.path.join('data','ILSVRC2012', mode)
                self.data = ImageFolder(root=root)

        elif self.dataset_name == "tiny_imagenet":
            if self.hdf5_path is not None:
                print('Loading %s into memory...' % self.hdf5_path)
                with h5.File(self.hdf5_path, 'r') as f:
                    self.data = f['imgs'][:]
                    self.labels = f['labels'][:]
            else:
                mode = 'train' if self.train == True else 'valid'
                root = os.path.join('data','TINY_ILSVRC2012', mode)
                self.data = ImageFolder(root=root)

        elif self.dataset_name == "inaturalist2019":
            if self.hdf5_path is not None:
                print('Loading %s into memory...' % self.hdf5_path)
                with h5.File(self.hdf5_path, 'r') as f:
                    self.data = f['imgs'][:]
                    self.labels = f['labels'][:]

            else:
                txt = '../iNat19/iNaturalist19_train.txt' if self.train == True else '../iNat19/iNaturalist19_val.txt'
                root = self.data_path
                self.data = LT_Dataset(root, txt)
                self.labels = self.data.labels
                
            counts = dict(Counter(self.labels))
            
            self.img_num_list = [0] * len(counts)
            for i in range(len(counts)):
                self.img_num_list[i] = counts[i]

        
        elif self.dataset_name == "inaturalist2017":
            if self.hdf5_path is not None:
                print('Loading %s into memory...' % self.hdf5_path)
                with h5.File(self.hdf5_path, 'r') as f:
                    self.data = f['imgs'][:]
                    self.labels = f['labels'][:]


            else:
                txt = os.path.join(self.data_path,'iNaturalist17_train.txt') if self.train == True else os.path.join(self.data_path,'iNaturalist17_val.txt')
                root = self.data_path
                self.data = LT_Dataset(root, txt)
                self.labels = self.data.labels
                
            counts = dict(Counter(self.labels))
            
            self.img_num_list = [0] * len(counts)
            for i in range(len(counts)):
                self.img_num_list[i] = counts[i]
        
        elif self.dataset_name == "imagenet_lt":
            if self.hdf5_path is not None:
                print('Loading %s into memory...' % self.hdf5_path)
                with h5.File(self.hdf5_path, 'r') as f:
                    self.data = f['imgs'][:]
                    self.labels = f['labels'][:]
            else:
                txt = './imagenet_lt/ImageNet_LT_train.txt' if self.train == True else './imagenet_lt/ImageNet_LT_val.txt'
                root = self.data_path
                self.data = LT_Dataset(root, txt)
                self.labels = self.data.labels
                
            counts = Counter(self.labels)
            
            
            self.img_num_list = [0] * len(counts)
            for i in range(len(counts)):
                self.img_num_list[i] = counts[i]
                
        elif self.dataset_name == "custom":
            if self.hdf5_path is not None:
                print('Loading %s into memory...' % self.hdf5_path)
                with h5.File(self.hdf5_path, 'r') as f:
                    self.data = f['imgs'][:]
                    self.labels = f['labels'][:]
            else:
                mode = 'train' if self.train == True else 'valid'
                root = os.path.join(self.data_path, mode)
                self.data = ImageFolder(root=root)
            
                self.labels = []
                for _,lbls in self.data:
                    self.labels.append(lbls)
        
            counts = Counter(self.labels)
            self.img_num_list = [0] * len(counts)
            for i in range(len(counts)):
                self.img_num_list[i] = counts[i]
            


        else:
            raise NotImplementedError

        if self.cfgs.per_samples != False and self.dataset_name != "inaturalist2019" and self.dataset_name != "inaturalist2017" and self.dataset_name != "lsun" and \
        self.dataset_name != "imagenet_lt":
            num_samples = int(self.cfgs.per_samples * len(self.data) / 100)
            np.random.seed(42)
            train_idx = np.random.choice(
                len(self.data), num_samples, replace=False)
            
            self.labels = self.data.targets
            self.data = np.array(self.data)[train_idx]
            self.labels = list(np.asarray(self.labels)[train_idx])
            
            counts = Counter(self.labels)
            self.img_num_list = [0] * len(counts)
            for i in range(len(counts)):
                self.img_num_list[i] = counts[i]


    def __len__(self):
        if self.hdf5_path is not None:
            num_dataset = self.data.shape[0]
        else:
            num_dataset = len(self.data)
        return num_dataset


    def __getitem__(self, index):
        if self.hdf5_path is None:
            img, label = self.data[index]
            img, label = self.transforms(img), int(label)
        else:
            img, label = np.transpose(self.data[index], (1,2,0)), int(self.labels[index])
            img = self.transforms(img)
        return img, label


class IMBALANCECIFAR10(torchvision.datasets.CIFAR10):
    cls_num = 10

    def __init__(self,dataset_name, root, imb_type='exp', imb_factor=0.01, rand_number=0, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super(IMBALANCECIFAR10, self).__init__(root, train, transform, target_transform, download)
        self.dataset_name = dataset_name
        np.random.seed(rand_number)
        self.img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
        print(self.img_num_list)

        self.gen_imbalanced_data(self.img_num_list)

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets
        
    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        
        return cls_num_list


class IMBALANCECIFAR100(IMBALANCECIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    This is a subclass of the `CIFAR10` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }
    cls_num = 100



class IMBALANCELSUN(torchvision.datasets.LSUN):
    def __init__(self, root, imb_type='exp', imb_factor=0.01, rand_number=0, classes="val",
                 transform=None, target_transform=None, max_samples = None):
        super(IMBALANCELSUN, self).__init__(root, classes, transform, target_transform,)
        np.random.seed(rand_number)
        self.dataset_name = 'lsun'
        self.max_samples = max_samples
        self.img_num_list = self.get_img_num_per_cls(len(self.classes), imb_type, imb_factor)
        self.gen_imbalanced_data(self.img_num_list)
        
    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = self.max_samples
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0 + 1e-9)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_list):

        modified_self_indices = []
        count = 0
        for c in img_num_list:
            count += c
            modified_self_indices.append(count)

        self.indices = modified_self_indices
        self.length = count

