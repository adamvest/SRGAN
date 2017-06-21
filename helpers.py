from PIL import Image
from skimage import measure
from numpy import sqrt
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


def compute_statistics(sr_img, hr_img, r=2):
    to_pil = transforms.ToPILImage()
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize((.5, .5, .5), (.5, .5, .5))

    cropped_sr_img = center_crop(to_pil(unnormalize(sr_img.data[0])))
    cropped_hr_img = center_crop(to_pil(hr_img))
    sr_y, _, _ = cropped_sr_img.convert("YCbCr").split()
    hr_y, _, _ = cropped_hr_img.convert("YCbCr").split()
    sr_y = normalize(to_tensor(sr_y)).numpy()
    hr_y = normalize(to_tensor(hr_y)).numpy()

    psnr = measure.compare_psnr(hr_y, sr_y, data_range=r)
    ssim = measure.compare_ssim(hr_y[0], sr_y[0], data_range=r)

    return (psnr, ssim)


def unnormalize(img):
    return img.mul_(.5).add_(.5)


def center_crop(img, border=8):
    (w, h) = img.size
    n_w, n_h = w - border, h - border
    l, r = (w - n_w) / 2, (w + n_w) / 2
    t, b = (h - n_h) / 2, (h + n_h) / 2
    return img.crop((l, t, r, b))


def save_images(model, args):
    to_pil = transforms.ToPILImage()
    to_tensor = transforms.ToTensor()

    hr_img = Image.open("./Set5/image_SRF_4/img_003_SRF_4_HR.png")
    lr_img = Variable(to_tensor(Image.open("./Set5/image_SRF_4/img_003_SRF_4_LR.png")), volatile=True)

    if args.use_cuda:
        lr_img = lr_img.cuda()

    sr_img = model(lr_img.unsqueeze(0))

    if args.use_cuda:
        sr_img = sr_img.data[0].cpu()
        lr_img = lr_img.data.cpu()

    sr_img = to_pil(unnormalize(sr_img.clamp(min=-1, max=1)))
    lr_img = to_pil(lr_img)

    hr_img.save("%s/hr_img.png" % args.out_folder)
    lr_img.save("%s/lr_img.png" % args.out_folder)
    sr_img.save("%s/sr_img.png" % args.out_folder)


def save_sr_results(args, dataset_name, sr_imgs):
    to_pil = transforms.ToPILImage()

    for i in range(len(sr_imgs)):
        img = to_pil(unnormalize(sr_imgs[i].clamp(min=-1, max=1)))
        img.save("%s/%s/sr_img_%03d.png" % (args.out_folder, dataset_name, i + 1))
