import os
from numpy import random
from PIL import Image
from torch import stack
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, ToPILImage, Scale


class BSD100Dataset(Dataset):
    def __init__(self, args, transform=None):
        self.transform = transform
        self.args = args
        self.images = []

        for root, _, filenames in os.walk(args.data_path):
            for fname in filenames:
                if "HR" in fname:
                    self.images.append(root + "/" + fname)

    def __getitem__(self, index):
        path = self.images[index]
        img = Image.open(path)
        (w, h) = img.size

        if w < self.args.min_size or h < self.args.min_size:
            img = self.scale(img)

        if self.transform != None:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.images)


class ImagenetDataset(Dataset):
    def __init__(self, args, transform):
        self.transform = transform
        self.scale = Scale(args.min_size)
        self.args = args
        images = []

        for dname in os.listdir(args.data_path):
            d = os.path.join(args.data_path, dname)

            if os.path.isdir(d):
                for root, _, fnames in os.walk(d):
                    for fname in fnames:
                        images.append(d + "/" + fname)

	random.shuffle(images)
	self.images = images[:args.num_examples]

    def __getitem__(self, index):
        path = self.images[index]
        img = Image.open(path)
        (w, h) = img.size

        if w < self.args.min_size or h < self.args.min_size:
            img = self.scale(img)

        if self.transform != None:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.images)


def build_evaluation_dataset(args):
    bsd100_hr_imgs, bsd100_lr_imgs = [], []
    urban100_hr_imgs, urban100_lr_imgs = [], []
    set5_hr_imgs, set5_lr_imgs = [], []
    set14_hr_imgs, set14_lr_imgs = [], []

    for dname in os.listdir(args.test_path):
        d = os.path.join(args.test_path, dname, "image_SRF_4")

        if os.path.isdir(d):
            if "BSD100" in d:
                for root, _, filenames in os.walk(d):
                    for fname in filenames:
                        if "HR" in fname:
                            hr_img, lr_img = open_image(root, fname, args.upscale_factor)
       	       	       	    bsd100_hr_imgs.append(hr_img)
                       	    bsd100_lr_imgs.append(lr_img)
            elif "Urban100" in d:
                for root, _, filenames in os.walk(d):
                    for fname in filenames:
                        if "HR" in fname:
                            hr_img, lr_img = open_image(root, fname, args.upscale_factor)
       	       	       	    urban100_hr_imgs.append(hr_img)
                       	    urban100_lr_imgs.append(lr_img)
            elif "Set5" in d:
                for root, _, filenames in os.walk(d):
                    for fname in filenames:
                        if "HR" in fname:
                            hr_img, lr_img = open_image(root, fname, args.upscale_factor)
                            set5_hr_imgs.append(hr_img)
                            set5_lr_imgs.append(lr_img)
            elif "Set14" in d:
                for root, _, filenames in os.walk(d):
                    for fname in filenames:
                        if "HR" in fname:
                            hr_img, lr_img = open_image(root, fname, args.upscale_factor)
       	       	       	    set14_hr_imgs.append(hr_img)
                       	    set14_lr_imgs.append(lr_img)

    return {"BSD100": (bsd100_hr_imgs, bsd100_lr_imgs),
            "Urban100": (urban100_hr_imgs, urban100_lr_imgs),
            "Set5": (set5_hr_imgs, set5_lr_imgs),
            "Set14": (set14_hr_imgs, set14_lr_imgs)}

def open_image(root, fname, upscale_factor):
    to_tensor = ToTensor()

    img = Image.open(root + "/" + fname)

    if img.mode != "RGB":
        img = img.convert("RGB")

    w, h = img.size
    dw, dh = w % upscale_factor, h % upscale_factor
    hr_img = img.resize((w - dw, h - dh), Image.BICUBIC)
    lr_img = img.resize((w / upscale_factor, h / upscale_factor), Image.BICUBIC)

    return to_tensor(hr_img), to_tensor(lr_img)


def custom_collate(batch):
    to_pil = ToPILImage()
    to_tensor = ToTensor()

    hr_imgs, lr_imgs = [], []
    num_images, num_channels, dim, _ = batch[0].size()

    for i in range(len(batch)):
        lr_batch = [to_tensor(to_pil(image).resize((dim/4, dim/4), Image.BICUBIC)) for image in batch[i]]
        lr_imgs.append(stack(lr_batch))
        hr_imgs.append(batch[i])

    hr_imgs = stack(hr_imgs).view(len(batch) * num_images, num_channels, dim, dim)
    lr_imgs = stack(lr_imgs).view(len(batch) * num_images, num_channels, dim/4, dim/4)

    return (hr_imgs, lr_imgs)


class CheckImageIsRGB():
    def __init__(self):
        pass

    def __call__(self, img):
        if img.mode != "RGB":
            img = img.convert("RGB")

        return img


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
            raise ValueError("Crop size too large")
        elif tw == w and th == h:
            return [img] * self.num_crops

        for i in range(self.num_crops):
            x = random.randint(0, w - tw)
            y = random.randint(0, h - th)
            crops.append(img.crop((x, y, x + tw, y + th)))

        return crops


class MultipleImagesToTensor():
    def __init__(self):
        self.to_tensor = ToTensor()

    def __call__(self, imgs):
        imgs = [self.to_tensor(img) for img in imgs]
        return stack(imgs)
