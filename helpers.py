from PIL import Image
from skimage import measure
from numpy import sqrt, log10
from torch import nn
from torch.autograd import Variable
from torchvision import transforms


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.fill_(0)
    elif isinstance(m, nn.Linear):
        fan_out, fan_in = m.weight.size(0), m.weight.size(1)
        variance = sqrt(2.0 / (fan_in + fan_out))
        m.weight.data.normal_(0.0, variance)
        m.bias.data.fill_(0)


def compute_statistics(sr_img, hr_img, r=1):
    to_pil = transforms.ToPILImage()
    to_tensor = transforms.ToTensor()

    cropped_sr_img = center_crop(to_pil(sr_img.data[0].clamp(min=0, max=1)))
    cropped_hr_img = center_crop(to_pil(hr_img))
    sr_y, _, _ = cropped_sr_img.convert("YCbCr").split()
    hr_y, _, _ = cropped_hr_img.convert("YCbCr").split()
    sr_y = to_tensor(sr_y).numpy()
    hr_y = to_tensor(hr_y).numpy()

    psnr = measure.compare_psnr(hr_y, sr_y, data_range=r)
    ssim = measure.compare_ssim(hr_y[0], sr_y[0], data_range=r)

    return (psnr, ssim)


def compute_rgb_psnr(sr_img, hr_img, r=1):
    mse_loss = nn.MSELoss()
    to_pil = transforms.ToPILImage()
    to_tensor = transforms.ToTensor()

    cropped_sr_img = center_crop(to_pil(sr_img.data[0].clamp(min=0, max=1)))
    cropped_hr_img = center_crop(to_pil(hr_img))
    cropped_sr_img = Variable(to_tensor(cropped_sr_img))
    cropped_hr_img = Variable(to_tensor(cropped_hr_img))

    mse = mse_loss(cropped_sr_img, cropped_hr_img).data[0]

    return 10 * log10((r**2) / mse)


def center_crop(img, border=8):
    (w, h) = img.size
    n_w, n_h = w - border, h - border
    l, r = (w - n_w) / 2, (w + n_w) / 2
    t, b = (h - n_h) / 2, (h + n_h) / 2
    return img.crop((l, t, r, b))


def save_sr_results(args, dataset_name, sr_imgs):
    to_pil = transforms.ToPILImage()

    for i in range(len(sr_imgs)):
        img = to_pil(sr_imgs[i].clamp(min=0, max=1))
        img.save("%s/%s/sr_img_%03d.png" % (args.out_folder, dataset_name, i + 1))
