import torch
import torchvision.transforms
from PIL import Image
from torch.utils.data import Dataset
import os


def get_transforms(img_size):
    return torchvision.transforms.Compose([
                torchvision.transforms.Resize(img_size, torchvision.transforms.functional.InterpolationMode.BICUBIC),
                torchvision.transforms.CenterCrop(img_size),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor()
            ])

def is_img(I):
    extension = I.split('.')[-1]
    return (extension.lower() == 'png') or (extension.lower() == 'jpg') or (extension.lower() == 'jpeg') or \
           (extension.lower() == 'bmp') or (extension.lower() == 'webp')


class unlabeled_data(Dataset):
    def __init__(self, data_dir, image_size, T=None):
        super(unlabeled_data, self).__init__()

        folders = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
        self.samples = []

        for f in folders:
            self.samples += [os.path.join(f, I) for I in os.listdir(f) if is_img(I)]

        if T is None:
            self.T = get_transforms(image_size)
        else:
            self.T = T

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        I = Image.open(self.samples[item]).convert("RGB")
        x0 = self.T(I)
        return x0


class labeled_data(Dataset):
    def __init__(self, data_dir, image_size, T=None):
        super(labeled_data, self).__init__()

        folders = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
        self.samples = []

        for f in folders:
            self.samples += [os.path.join(f, I) for I in os.listdir(f) if is_img(I)]

        if T is None:
            self.T = torchvision.transforms.Compose([
                torchvision.transforms.Resize(image_size,
                                              interpolation=torchvision.transforms.functional.InterpolationMode.BICUBIC),
                torchvision.transforms.CenterCrop(image_size),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor()
            ])
        else:
            self.T = T

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        I = Image.open(self.samples[item]).convert("RGB")
        x0 = self.T(I)
        return x0
