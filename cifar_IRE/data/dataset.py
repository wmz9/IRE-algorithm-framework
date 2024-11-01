import torch
import torchvision
import torchvision.datasets 
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from data.autoaugment import CIFAR10Policy

# from utils.configurable import configurable
# from data.build import DATASET_REGISTRY

# @DATASET_REGISTRY.register()
# class CIFAR10_base:
#     @configurable
#     def __init__(self, datadir) -> None:
#         self.datadir = datadir

#         self.n_classes = 10
#         self.mean = np.array([125.3, 123.0, 113.9]) / 255.0
#         self.std = np.array([63.0, 62.1, 66.7]) / 255.0
    
#     @classmethod
#     def from_config(cls, args):
#         return {
#             "datadir": args.datadir,
#         }
    
#     def get_data(self):
#         train_data = torchvision.datasets.CIFAR10(root=self.datadir, train=True, transform=self._train_transform(), download=True)
#         val_data = torchvision.datasets.CIFAR10(root=self.datadir, train=False, transform=self._test_transform(), download=True)
#         return train_data, val_data

#     def _train_transform(self):
#         train_transform = torchvision.transforms.Compose([
#             torchvision.transforms.RandomCrop(size=(32, 32), padding=4),
#             torchvision.transforms.RandomHorizontalFlip(),
#             torchvision.transforms.ToTensor(),
#             torchvision.transforms.Normalize(self.mean, self.std),
#             # Cutout()
#         ])
#         return train_transform

#     def _test_transform(self):
#         test_transform = torchvision.transforms.Compose([
#             torchvision.transforms.ToTensor(),
#             torchvision.transforms.Normalize(self.mean, self.std)
#         ])
#         return test_transform

# @DATASET_REGISTRY.register()
# class CIFAR10_cutout(CIFAR10_base):
#     def _train_transform(self):
#         train_transform = torchvision.transforms.Compose([
#             torchvision.transforms.RandomCrop(size=(32, 32), padding=4),
#             torchvision.transforms.RandomHorizontalFlip(),
#             torchvision.transforms.ToTensor(),
#             torchvision.transforms.Normalize(self.mean, self.std),
#             Cutout(size=16, p=0.5),
#         ])
#         return train_transform



# @DATASET_REGISTRY.register()
class CIFAR100_base:
    # @configurable
    def __init__(self, batch_size, threads, autoaugment=False, cutout=False) -> None:
        self.datadir = './data'
        self.n_classes = 100
        self.mean = np.array([125.3, 123.0, 113.9]) / 255.0
        self.std = np.array([63.0, 62.1, 66.7]) / 255.0
    
        if cutout:
            train_transform = transforms.Compose([
                torchvision.transforms.RandomCrop(size=(32, 32), padding=4),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(self.mean, self.std),
                Cutout()
            ])
        elif autoaugment:
            train_transform = transforms.Compose([
                torchvision.transforms.RandomCrop(size=(32, 32), padding=4),
                torchvision.transforms.RandomHorizontalFlip(),
                CIFAR10Policy(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(self.mean, self.std),
            ])
        else:
            train_transform = transforms.Compose([
                torchvision.transforms.RandomCrop(size=(32, 32), padding=4),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(self.mean, self.std),
            ])

        test_transform = transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(self.mean, self.std)
        ])

        self.train_set = torchvision.datasets.CIFAR100(root=self.datadir, train=True, transform=train_transform, download=True)
        self.test_set = torchvision.datasets.CIFAR100(root=self.datadir, train=False, transform=test_transform, download=True)


        self.train = torch.utils.data.DataLoader(self.train_set, batch_size=batch_size, shuffle=True, num_workers=threads)
        self.estimate = torch.utils.data.DataLoader(self.train_set, batch_size=batch_size, shuffle=True, num_workers=4)
        self.test = torch.utils.data.DataLoader(self.test_set, batch_size=batch_size, shuffle=False, num_workers=threads)


class CIFAR10_base:
    def __init__(self, batch_size, threads, autoaugment=False,cutout=False):
        mean, std = self._get_statistics()

        if cutout:
            train_transform = transforms.Compose([
                torchvision.transforms.RandomCrop(size=(32, 32), padding=4),
                torchvision.transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
                Cutout()
            ])
        elif autoaugment:
            train_transform = transforms.Compose([
                torchvision.transforms.RandomCrop(size=(32, 32), padding=4),
                torchvision.transforms.RandomHorizontalFlip(),
                CIFAR10Policy(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(self.mean, self.std),
            ])
        else:
                  train_transform = transforms.Compose([
                torchvision.transforms.RandomCrop(size=(32, 32), padding=4),
                torchvision.transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        self.train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
        self.test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

        self.train = torch.utils.data.DataLoader(self.train_set, batch_size=batch_size, shuffle=True, num_workers=threads)
        self.estimate = torch.utils.data.DataLoader(self.train_set, batch_size=batch_size, shuffle=True, num_workers=4)
        self.test = torch.utils.data.DataLoader(self.test_set, batch_size=batch_size, shuffle=False, num_workers=threads)

        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    def _get_statistics(self):
        train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
        data = torch.cat([d[0] for d in DataLoader(train_set)])
        return data.mean(dim=[0, 2, 3]), data.std(dim=[0, 2, 3])

class CIFAR10_fast:
    def __init__(self, batch_size, threads):
        mean, std = self._get_statistics()

        train_transform = transforms.Compose([
            torchvision.transforms.RandomCrop(size=(32, 32), padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            # Cutout()
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
        test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

        self.train = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=threads)
        self.estimate = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=threads)
        self.test = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=threads)

        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    def _get_statistics(self):
        train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())

        data = torch.cat([d[0] for d in DataLoader(train_set)])
        return data.mean(dim=[0, 2, 3]), data.std(dim=[0, 2, 3])


# # @DATASET_REGISTRY.register()
# class CIFAR100_cutout(CIFAR100_base):
#     def _train_transform(self):
#         train_transform = torchvision.transforms.Compose([
#             torchvision.transforms.RandomCrop(size=(32, 32), padding=4),
#             torchvision.transforms.RandomHorizontalFlip(),
#             torchvision.transforms.ToTensor(),
#             torchvision.transforms.Normalize(self.mean, self.std),
#             Cutout(size=16, p=0.5),
#         ])
#         return train_transform


#@DATASET_REGISTRY.register()
class ImageNet_base:
    #@configurable
    def __init__(self, datadir, batch_size) -> None:
        self.datadir = datadir
        self.batch_size = batch_size
        self.n_classes = 1000
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        self.train_datasets = torchvision.datasets.ImageFolder(root=self.datadir + '/train', transform=self._train_transform())
        self.val_datasets = torchvision.datasets.ImageFolder(root=self.datadir + '/val', transform=self._test_transform())
        # self.train_datasets = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=self._train_transform())
        # self.val_datasets = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=self._test_transform())
        
        self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_datasets)
        self.val_sampler = torch.utils.data.distributed.DistributedSampler(self.val_datasets)

        self.train_dataloader = torch.utils.data.DataLoader(self.train_datasets,
		        batch_size=batch_size, sampler=self.train_sampler, pin_memory=True, num_workers=4)
        self.val_dataloader = torch.utils.data.DataLoader(self.val_datasets,
                batch_size=batch_size, sampler=self.val_sampler, pin_memory=True, num_workers=4)

    @classmethod
    def from_config(cls, args):
        return {
            "datadir": args.datadir,
        }
    
    def get_estimate_data(self, batch_size):
        train_dataloader = torch.utils.data.DataLoader(self.train_datasets, batch_size=batch_size, shuffle=True, num_workers=4)
        estimate_image, _ = next(iter(train_dataloader))
        return estimate_image

    def _train_transform(self):
        train_transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(224),
            torchvision.transforms.RandomHorizontalFlip(),
            #torchvision.transforms.RandAugment(2,15),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(self.mean, self.std),
        ])
        return train_transform

    def _test_transform(self):
        test_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(self.mean, self.std),
        ])
        return test_transform



class Cutout(object):
    def __init__(self, size=16, p=0.5):
        self.size = size
        self.p = p

    def __call__(self, image):
        if torch.rand([1]).item() > self.p:
            return image
        
        h, w = image.size(1), image.size(2)
        mask = np.ones((h,w), np.float32)

        x = np.random.randint(w)
        y = np.random.randint(h)

        y1 = np.clip(y - self.size // 2, 0, h)
        y2 = np.clip(y + self.size // 2, 0, h)
        x1 = np.clip(x - self.size // 2, 0, w)
        x2 = np.clip(x + self.size // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(image)
        image = image * mask
        return image