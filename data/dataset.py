from typing import List
from os.path import join
import numpy as np
import pickle
import torchvision.transforms as tv
from torchvision.datasets import CIFAR10, CIFAR100
from data.randaugment import RandAugmentMC
from torch.utils.data import Dataset


class Transform:
    def __init__(self, dataset_name: str, is_train: bool):
        if "cifar" in dataset_name:
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
        elif "imagenet" in dataset_name:
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

        self.transform = dict_transform['train'] if is_train else dict_transform['test']

        self.simclr = tv.Compose([
            tv.RandomResizedCrop(size=32, scale=(0.2, 1.)),
            tv.RandomHorizontalFlip(),
            tv.RandomApply([
                tv.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            tv.RandomGrayscale(p=0.2),
            tv.ToTensor(),
            tv.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

        self.fixmatch_weak = tv.Compose([
            tv.RandomHorizontalFlip(),
            tv.RandomCrop(size=32,
                          padding=int(32 * 0.125),
                          padding_mode='reflect'),
            tv.ToTensor(),
            tv.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

        self.fixmatch_strong = tv.Compose([
            tv.RandomHorizontalFlip(),
            tv.RandomCrop(size=32,
                          padding=int(32 * 0.125),
                          padding_mode='reflect'),
            RandAugmentMC(n=2, m=10),
            tv.ToTensor(),
            tv.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

    def __call__(self, inp):
        out1, out2 = self.transform(inp), self.transform(inp)
        out_weak, out_aug = self.fixmatch_weak(inp), self.fixmatch_strong(inp)

        return out1, out2, out_weak, out_aug


def OpenWorldDataset(dataset_name: str,
                     label_num: int,
                     label_ratio: float,
                     data_dir: str,
                     is_train: bool,
                     is_label: bool,
                     seed: int,
                     unlabeled_idxs: None,
                     ):
    extra_args = dict(label_num=label_num, label_ratio=label_ratio, seed=seed)

    if dataset_name == "cifar10":
        dataset_class = Cifar10(
            data_dir=data_dir,
            is_label=is_label,
            transform=Transform(dataset_name=dataset_name, is_train=is_train),
            unlabeled_idxs=unlabeled_idxs,
            **extra_args
        )
    elif dataset_name == "cifar100":
        dataset_class = Cifar100(
            data_dir=data_dir,
            is_label=is_label,
            transform=Transform(dataset_name=dataset_name, is_train=is_train),
            unlabeled_idxs=unlabeled_idxs,
            **extra_args
        )
    elif dataset_name == "imagenet":
        dataset_class = Imagenet(
            data_dir=data_dir,
            anno_file='./imagenet/ImageNet100_label_{}_{:.2f}.txt'.format(label_num, label_ratio),
            transform=Transform(dataset_name=dataset_name, is_train=is_train)
        )

    else:
        raise ValueError("Unknown dataset: {}".format(dataset_name))

    return dataset_class


class Imagenet(Dataset):
    def __init__(self, data_dir: str, anno_file: str, loader, transform=None, target_transform=None):
        super().__init__()

        self.data_dir = data_dir
        self.anno_file = join(data_dir, anno_file)
        self.loader = loader
        self.transform = transform
        self.target_transform = target_transform

        self.read_file()

    def read_file(self):
        filenames, targets = [], []

        with open(self.anno_file, "r") as f:
            for line in f.readlines():
                line_split = line.strip("\n").split(" ")
                filenames.append(line_split[0])
                targets.append(int(line_split[1]))

        self.samples = filenames
        self.targets = targets

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path = join(self.data_dir, self.samples[index])
        target = self.targets[index]
        sample = self.loader(path)

        if self.transform is not None:
            sample = self.transform(sample)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, index


class Cifar100(CIFAR100):
    def __init__(self,
                 data_dir: str,
                 is_label: bool,
                 is_train: bool = True,
                 transform: tv.Compose = None,
                 label_num: int = 50,
                 label_ratio: float = 0.5,
                 seed: int = 0,
                 unlabeled_idxs: List = None,
                 ):
        super(Cifar100, self).__init__(data_dir, train=is_train, transform=transform, target_transform=None,
                                       download=True)

        downloaded_list = self.train_list

        self.data = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = join(data_dir, self.base_folder, file_name)
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
        np.random.seed(seed)

        if is_label:
            self.labeled_idxs, self.unlabeled_idxs = self.get_labeled_index(labeled_classes, label_ratio)
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


class Cifar10(CIFAR10):
    def __init__(self,
                 data_dir: str,
                 is_label: bool,
                 is_train: bool = True,
                 transform: tv.Compose = None,
                 label_num: int = 50,
                 label_ratio: float = 0.5,
                 seed: int = 0,
                 unlabeled_idxs: List = None,
                 ):
        super(Cifar10, self).__init__(data_dir, train=is_train, transform=transform, target_transform=None,
                                      download=True)

        downloaded_list = self.train_list

        self.data = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = join(data_dir, self.base_folder, file_name)
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
        np.random.seed(seed)

        if is_label:
            self.labeled_idxs, self.unlabeled_idxs = self.get_labeled_index(labeled_classes, label_ratio)
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
