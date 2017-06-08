import os
from numpy import random
from PIL import Image
from torch import stack
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


class BSD100Dataset(Dataset):
    def __init__(self, args, transform=None):
        self.transform = transform
        self.images = []
        self.args = args

        for root, _, filenames in os.walk(args.train_path):
            for fname in filenames:
                if "HR" in fname:
                    self.images.append(root + "/" + fname)

    def __getitem__(self, index):
        path = self.images[index]
        img = Image.open(path)

        if self.transform != None:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.images)


class MultipleRandomCrops():
    def __init__(self, size, num_crops):
        if isinstance(size, tuple):
            self.crop_size = size
        else:
            self.crop_size = (size, size)

        self.num_crops = num_crops

    def __call__(self, img):
        w, h = img.size
        tw, th = self.crop_size
        crops = []

        if tw > w or th > h:
            return ValueError("Crop size too large")
        elif tw == w and th == h:
            return [img] * self.num_crops

        for i in range(self.num_crops):
            x = random.randint(0, w - tw)
            y = random.randint(0, h - th)
            crops.append(img.crop((x, y, x + tw, y + th)))

        return crops


class MultipleImagesToTensor():
    def __init__(self):
        pass

    def __call__(self, imgs):
        to_tensor = ToTensor()
        imgs = [to_tensor(img) for img in imgs]
        return stack(imgs)
