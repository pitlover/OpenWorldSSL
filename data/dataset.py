from typing import Optional, Dict
import os
from os.path import join
import torch
import torch.nn as nn
import numpy as np
import pickle
import torchvision.transforms as tv
from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data import Dataset


class OpenWorldDataset(Dataset):
    def __init__(self,
                 dataset_name: str,
                 label_num: int,
                 label_ratio: float,
                 data_dir: str,
                 is_train: bool,
                 is_label: bool
                 ):
        super().__init__()

        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.is_train = is_train
        self.is_label = is_label
        self.label_num = label_num
        self.label_ratio = label_ratio

        label_args = dict(label_num=label_num, label_ratio=label_ratio)

        if dataset_name == "cifar10":
            self.num_class = 10
            dataset_class = Cifar10
        elif dataset_name == "cifar100":
            self.num_class = 100
            raise NotImplementedError("Not implement yet Cifar100!")
        elif dataset_name == "imagenet":
            self.num_class = 100
            raise NotImplementedError("Not implement yet Imagenet!")

        else:
            raise ValueError("Unknown dataset: {}".format(dataset_name))

        self.dataset = dataset_class(
            data_dir=self.data_dir,
            is_train=self.is_train,
            is_label=self.is_label,
            num_class=self.num_class,
            transform=self.get_transform,
            **label_args
        )

    @property
    def get_transform(self) -> tv.Compose:
        if "cifar" in self.dataset_name:
            dict_transform = {
                'train': tv.Compose([
                    tv.RandomCrop(32, padding=4),
                    tv.RandomHorizontalFlip(),
                    tv.ToTensor(),
                    tv.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
                ]),
                'test': tv.Compose([
                    tv.ToTensor(),
                    tv.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
                ])
            }
        elif "imagnet" in self.dataset_name:
            dict_transform = {
                "train": tv.Compose([
                    tv.RandomResizedCrop(224, scale=(0.5, 1.0)),
                    tv.RandomHorizontalFlip(),
                    tv.ToTensor(),
                    tv.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]),
                "test": tv.Compose([
                    tv.Resize(256),
                    tv.CenterCrop(224),
                    tv.ToTensor(),
                    tv.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
            }

        return dict_transform['train'] if self.is_train else dict_transform['test']

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Dict:
        img, aug_img, label, img_path = self.dataset[index]

        ret = {
            "index": index,
            "img": img,
            "aug_img": aug_img,
            "label": label,
            "img_path": img_path
        }

        return ret


class Cifar10(CIFAR10):
    def __init__(self,
                 data_dir: str,
                 is_train: bool,
                 transform: tv.Compose,
                 num_class: int,
                 is_label: bool,
                 label_num: int,
                 label_ratio: float,
                 rand_number: int = 0,
                 download: bool = True,
                 unlabeled_index=None,
                 ):
        super().__init__()

        downloaded_list = self.train_list
        self.data = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = join(self.data_dir, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        labeled_classes = range(label_num)
        np.random.seed(rand_number)

        if labeled:
            self.labeled_idxs, self.unlabeled_idxs = self.get_labeled_index(labeled_classes, labeled_ratio)
            self.shrink_data(self.labeled_idxs)
        else:
            self.shrink_data(unlabeled_idxs)

    def get_labeled_index(self, labeled_classes, labeled_ratio):
        labeled_idxs = []
        unlabeled_idxs = []
        for idx, label in enumerate(self.targets):
            if label in labeled_classes and np.random.rand() < labeled_ratio:
                labeled_idxs.append(idx)
            else:
                unlabeled_idxs.append(idx)
        return labeled_idxs, unlabeled_idxs

    def shrink_data(self, idxs):
        targets = np.array(self.targets)
        self.targets = targets[idxs].tolist()
        self.data = self.data[idxs, ...]