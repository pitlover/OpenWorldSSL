import itertools
from typing import List, Tuple
from os.path import join
import numpy as np
import pickle
import torchvision.transforms as tv
import accimage
import bisect
import warnings

from torchvision import get_image_backend
from torchvision.datasets import CIFAR10, CIFAR100
from data.randaugment import RandAugmentMC
from torch.utils.data import Dataset
from PIL import Image
from torch.utils.data.sampler import Sampler


class Transform:
    def __init__(self, dataset_name: str, is_train: bool):
        if "cifar" in dataset_name:
            res = 32
            dict_transform = {
                'train': tv.Compose([
                    tv.RandomCrop(res, padding=4),
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
            res = 224
            dict_transform = {
                "train": tv.Compose([
                    tv.RandomResizedCrop(res, scale=(0.5, 1.0)),
                    tv.RandomHorizontalFlip(),
                    tv.ToTensor(),
                    tv.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]),
                "test": tv.Compose([
                    tv.Resize(256),
                    tv.CenterCrop(res),
                    tv.ToTensor(),
                    tv.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
            }

        self.transform = dict_transform['train'] if is_train else dict_transform['test']

        self.simclr = tv.Compose([
            tv.RandomResizedCrop(size=res, scale=(0.2, 1.)),
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
            tv.RandomResizedCrop(res, scale=(0.5, 1.0)),
            # tv.RandomCrop(size=res,
            #               padding=int(res * 0.125),
            #               padding_mode='reflect'),
            tv.ToTensor(),
            tv.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

        self.fixmatch_strong = tv.Compose([
            tv.RandomHorizontalFlip(),
            tv.RandomResizedCrop(res, scale=(0.5, 1.0)),
            # tv.RandomCrop(size=res,
            #               padding=int(res * 0.125),
            #               padding_mode='reflect'),
            RandAugmentMC(n=2, m=10),
            tv.ToTensor(),
            tv.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

    def __call__(self, inp):
        out1, out2 = self.transform(inp), self.transform(inp)
        out_aug_weak, out_aug_strong = self.fixmatch_weak(inp), self.fixmatch_strong(inp)

        return out1, out2, out_aug_weak, out_aug_strong


def OpenWorldDataset(dataset_name: str,
                     label_num: int,
                     label_ratio: float,
                     data_dir: str,
                     is_train: bool,
                     is_label: bool,
                     seed: int = 0,
                     unlabeled_idxs: List = None,
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
        label = "label" if is_label else "unlabel"
        dataset_class = Imagenet(
            data_dir=data_dir,
            anno_file='ImageNet100_{}_{}_{:.2f}.txt'.format(label, label_num, label_ratio),
            transform=Transform(dataset_name=dataset_name, is_train=is_train)
        )

    else:
        raise ValueError("Unknown dataset: {}".format(dataset_name))

    return dataset_class


class Imagenet(Dataset):
    def __init__(self, data_dir: str, anno_file: str, transform=None, target_transform=None):
        # super().__init__()
        self.data_dir = join(data_dir, "imagenet")
        self.anno_file = join(self.data_dir, anno_file)
        self.transform = transform
        self.target_transform = target_transform
        self.loader = self._get_loader

        self._read_file()
        self.unlabeled_idxs = None  # Just for unity with cifar

    def _pil_loader(self, path):
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def _accimage_loader(self, path):
        try:
            return self._accimage.Image(path)
        except IOError:
            # Potentially a decoding problem, fall back to PIL.Image
            return self._pil_loader(path)

    def _get_loader(self, path):
        from torchvision import get_image_backend
        if get_image_backend() == 'accimage':
            return self._accimage_loader(path)
        else:
            return self._pil_loader(path)

    def _read_file(self):
        filenames, targets = [], []

        with open(self.anno_file, "r") as f:
            for line in f.readlines():
                line_split = line.strip("\n").split(" ")
                filenames.append(line_split[0])
                targets.append(int(line_split[1]))

        self.samples = filenames
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index) -> Tuple:
        path = join(self.data_dir, self.samples[index])
        target = self.targets[index]
        sample = self.loader(path)

        if self.transform is not None:
            out1, out2, out_weak, out_aug = self.transform(sample)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return (out1, out2, out_weak, out_aug), target


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


class ConcatDataset(Dataset):
    """
    Original code is from orca.
    Dataset to concatenate multiple datasets.
    Purpose: useful to assemble different existing datasets, possibly
    large-scale datasets as the concatenation operation is done in an
    on-the-fly manner.
    Arguments:
        datasets (sequence): List of datasets to be concatenated
    """

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets):
        super(ConcatDataset, self).__init__()
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        self.datasets = list(datasets)
        self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]

        return self.datasets[dataset_idx][sample_idx]

    @property
    def cummulative_sizes(self):
        warnings.warn("cummulative_sizes attribute is renamed to "
                      "cumulative_sizes", DeprecationWarning, stacklevel=2)
        return self.cumulative_sizes


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)

    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices
    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, primary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.primary_batch_size = primary_batch_size
        self.secondary_batch_size = batch_size - primary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                   grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size
