import os
from numpy import random
from PIL import Image
from torch import stack
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, ToPILImage, Scale, Normalize


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

        if img.mode != "RGB":
            img = img.convert("RGB")

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

        if w < 128 or h < 128:
            img = self.scale(img)

        if self.transform != None:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.images)


def build_ycbcr_evaluation_dataset(args):
    bsd100_hr_imgs, bsd100_lr_y_imgs, bsd100_lr_cbcr_imgs = [], [], []
    urban100_hr_imgs, urban100_lr_y_imgs, urban100_lr_cbcr_imgs = [], [], []
    set5_hr_imgs, set5_lr_y_imgs, set5_lr_cbcr_imgs = [], [], []
    set14_hr_imgs, set14_lr_y_imgs, set14_lr_cbcr_imgs = [], [], []

    for dname in os.listdir(args.test_path):
        d = os.path.join(args.test_path, dname, "image_SRF_4")

        if os.path.isdir(d):
            if "BSD100" in d:
                for root, _, filenames in os.walk(d):
                    for fname in filenames:
                        if "HR" in fname:
                            bsd100_hr_imgs.append(open_ycbcr_image(root, fname))
                        elif "LR" in fname:
                            y_img, cb_img, cr_img = open_ycbcr_image(root, fname, get_cbcr=True)
                            bsd100_lr_y_imgs.append(y_img)
                            bsd100_lr_cbcr_imgs.append((cb_img, cr_img))
            elif "Urban100" in d:
                for root, _, filenames in os.walk(d):
                    for fname in filenames:
                        if "HR" in fname:
                            urban100_hr_imgs.append(open_ycbcr_image(root, fname))
                        elif "LR" in fname:
                            y_img, cb_img, cr_img = open_ycbcr_image(root, fname, get_cbcr=True)
                            urban100_lr_y_imgs.append(y_img)
                            urban100_lr_cbcr_imgs.append((cb_img, cr_img))
            elif "Set5" in d:
                for root, _, filenames in os.walk(d):
                    for fname in filenames:
                        if "HR" in fname:
                            set5_hr_imgs.append(open_ycbcr_image(root, fname))
                        elif "LR" in fname:
                            y_img, cb_img, cr_img = open_ycbcr_image(root, fname, get_cbcr=True)
                            set5_lr_y_imgs.append(y_img)
                            set5_lr_cbcr_imgs.append((cb_img, cr_img))
            elif "Set14" in d:
                for root, _, filenames in os.walk(d):
                    for fname in filenames:
                        if "HR" in fname:
                            set14_hr_imgs.append(open_ycbcr_image(root, fname))
                        elif "LR" in fname:
                            y_img, cb_img, cr_img = open_ycbcr_image(root, fname, get_cbcr=True)
                            set14_lr_y_imgs.append(y_img)
                            set14_lr_cbcr_imgs.append((cb_img, cr_img))

    return {"BSD100": (bsd100_hr_imgs, bsd100_lr_y_imgs, bsd100_lr_cbcr_imgs),
            "Urban100": (urban100_hr_imgs, urban100_lr_y_imgs, urban100_lr_cbcr_imgs),
            "Set5": (set5_hr_imgs, set5_lr_y_imgs, set5_lr_cbcr_imgs),
            "Set14": (set14_hr_imgs, set14_lr_y_imgs, set14_lr_cbcr_imgs)}

def build_rgb_evaluation_dataset(args):
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
                            bsd100_hr_imgs.append(open_rgb_image(root, fname))
                        elif "LR" in fname:
                            bsd100_lr_imgs.append(open_rgb_image(root, fname))
            elif "Urban100" in d:
                for root, _, filenames in os.walk(d):
                    for fname in filenames:
                        if "HR" in fname:
                            urban100_hr_imgs.append(open_rgb_image(root, fname))
                        elif "LR" in fname:
                            urban100_lr_imgs.append(open_rgb_image(root, fname))
            elif "Set5" in d:
                for root, _, filenames in os.walk(d):
                    for fname in filenames:
                        if "HR" in fname:
                            set5_hr_imgs.append(open_rgb_image(root, fname))
                        elif "LR" in fname:
                            set5_lr_imgs.append(open_rgb_image(root, fname))
            elif "Set14" in d:
                for root, _, filenames in os.walk(d):
                    for fname in filenames:
                        if "HR" in fname:
                            set14_hr_imgs.append(open_rgb_image(root, fname))
                        elif "LR" in fname:
                            set14_lr_imgs.append(open_rgb_image(root, fname))

    return {"BSD100": (bsd100_hr_imgs, bsd100_lr_imgs),
            "Urban100": (urban100_hr_imgs, urban100_lr_imgs),
            "Set5": (set5_hr_imgs, set5_lr_imgs),
            "Set14": (set14_hr_imgs, set14_lr_imgs)}

def open_rgb_image(root, fname):
    to_tensor = ToTensor()
    normalize = Normalize((.5, .5, .5), (.5, .5, .5))

    img = Image.open(root + "/" + fname)

    if img.mode != "RGB":
        img = img.convert("RGB")

    return normalize(to_tensor(img))

def open_ycbcr_image(root, fname, get_cbcr=False):
    to_tensor = ToTensor()
    normalize = Normalize((.5, .5, .5), (.5, .5, .5))

    img = Image.open(root + "/" + fname)
    img = img.convert("YCbCr")
    img = to_tensor(img)
    y, cb, cr = img[0, :, :].unsqueeze(0), img[1, :, :].unsqueeze(0), img[2, :, :].unsqueeze(0)

    if get_cbcr:
        return (y, cb, cr)

    return normalize(y)


class ExtractYChannel():
    def __init__(self):
        self.to_tensor = ToTensor()
        self.to_pil = ToPILImage()

    def __call__(self, img):
        img = img.convert("YCbCr")
        img = self.to_tensor(img)
        img = img[0, :, :].unsqueeze(0)
        return self.to_pil(img)


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
