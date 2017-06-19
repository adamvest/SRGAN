from PIL import Image
from skimage.measure import compare_psnr, compare_ssim
from numpy import sqrt, log10
from torch import is_tensor, stack, nn
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

    cropped_sr_img = center_crop(to_pil(unnormalize(sr_img.data[0]).clamp_(min = 0, max = 1)))
    cropped_hr_img = center_crop(to_pil(unnormalize(hr_img.data[0])))

    if convert_to_ycbcr:
        cropped_sr_img, _, _ = cropped_sr_img.convert("YCbCr").split()
        cropped_hr_img, _, _ = cropped_hr_img.convert("YCbCr").split()

    hr_img = Variable(normalize(to_tensor(cropped_hr_img)))
    sr_img = Variable(normalize(to_tensor(cropped_sr_img)))

    psnr = compare_psnr(hr_img, sr_img, data_range=r)
    ssim = compare_ssim(hr_img, sr_img, data_range=r)

    return (psnr, ssim)


def unnormalize(img):
    return img.mul_(.5).add_(.5)


def center_crop(img, border=8):
    (w, h) = img.size
    n_w, n_h = w - border, h - border
    l, r = (w - n_w) / 2, (w + n_w) / 2
    t, b = (h - n_h) / 2, (h + n_h) / 2
    return img.crop((l, t, r, b))


def reconstruct_rgb_image(y, cb, cr, resize=False, scale=4):
    if resize:
        (w, h) = cb.size
        cb = cb.resize((scale * w, scale * h), Image.BICUBIC)
        cr = cr.resize((scale * w, scale * h), Image.BICUBIC)

    img = Image.merge("YCbCr", (y, cb, cr))
    return img.convert("RGB")


def save_images(model, args):
    to_pil = transforms.ToPILImage()
    to_tensor = transforms.ToTensor()

    hr_img = Image.open("./Set5/image_SRF_4/img_003_SRF_4_HR.png")
    lr_img = Image.open("./Set5/image_SRF_4/img_003_SRF_4_LR.png")

    if not args.use_rgb:
        lr_img = lr_img.convert("YCbCr")
        (lr_img, lr_cb, lr_cr) = lr_img.split()

    lr_img = Variable(to_tensor(lr_img), volatile=True)

    if args.use_cuda:
        lr_img = lr_img.cuda()

    sr_img = model(lr_img.unsqueeze(0))

    if args.use_cuda:
        sr_img = to_pil(sr_img.data[0].cpu())
        lr_img = to_pil(lr_img.data.cpu())

    if not args.use_rgb:
        lr_img = reconstruct_rgb_image(lr_img, lr_cb, lr_cr)
        sr_img = reconstruct_rgb_image(sr_img, lr_cb, lr_cr, resize=True)

    hr_img.save("%s/hr_img.png" % args.out_folder)
    lr_img.save("%s/lr_img.png" % args.out_folder)
    sr_img.save("%s/sr_img.png" % args.out_folder)


def save_sr_results(args, dataset_name, sr_imgs, sr_cbcr_imgs=None):
    to_pil = transforms.ToPILImage()

    for i in range(len(sr_imgs)):
        img = to_pil(sr_imgs[i])

        if sr_cbcr_imgs != None:
            img = reconstruct_rgb_image(img, sr_cbcr_imgs[i][0], sr_cbcr_imgs[i][1])

        img.save("%s/%s/sr_img_%03d.png" % (args.out_folder, dataset_name, i + 1))
