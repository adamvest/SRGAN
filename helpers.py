from PIL import Image
from numpy import sqrt, log10
from torch import is_tensor, stack, nn
from torch.autograd import Variable
from torchvision import transforms


def custom_collate(batch):
    to_pil = transforms.ToPILImage()
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize((.5, .5, .5), (.5, .5, .5))

    hr_imgs, lr_imgs = [], []
    num_images, num_channels, dim, _ = batch[0].size()

    for i in range(len(batch)):
        lr_batch = [to_tensor(to_pil(image).resize((dim/4, dim/4), Image.BICUBIC)) for image in batch[i]]
        lr_imgs.append(stack(lr_batch))
        hr_imgs.append(normalize(batch[i]))

    hr_imgs = stack(hr_imgs).view(len(batch) * num_images, num_channels, dim, dim)
    lr_imgs = stack(lr_imgs).view(len(batch) * num_images, num_channels, dim/4, dim/4)

    return (hr_imgs, lr_imgs)


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


def evaluate_psnr(sr_img, hr_img, convert_to_ycbcr=False, r=2):
    to_pil = transforms.ToPILImage()
    to_tensor = transforms.ToTensor()

    cropped_sr_img = center_crop(to_pil(sr_img.data[0]))
    cropped_hr_img = center_crop(to_pil(hr_img.data[0]))

    if convert_to_ycbcr:
        cropped_sr_img = cropped_sr_img.convert("YCbCr")
        cropped_sr_img, _, _ = cropped_sr_img.split()
        cropped_hr_img = cropped_hr_img.convert("YCbCr")
        cropped_hr_img, _, _ = cropped_hr_img.split()

    mse_loss = nn.MSELoss()
    mse = mse_loss(Variable(to_tensor(cropped_sr_img)), Variable(to_tensor(cropped_hr_img)))

    return 10 * log10((r**2) / mse.data[0])


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
        lr_img = Image.merge("YCbCr", (lr_img, lr_cb, lr_cr))
        lr_img = lr_img.convert("RGB") #convert for saving

        (w, h) = lr_cb.size
        sr_cb = lr_cb.resize((4 * w, 4 * h), Image.BICUBIC)
        sr_cr = lr_cr.resize((4 * w, 4 * h), Image.BICUBIC)
        sr_img = Image.merge("YCbCr", (sr_img, sr_cb, sr_cr))
        sr_img = sr_img.convert("RGB") #convert for saving

    hr_img.save("%s/hr_img.png" % args.out_folder)
    lr_img.save("%s/lr_img.png" % args.out_folder)
    sr_img.save("%s/sr_img.png" % args.out_folder)


def save_sr_results(args, dataset_name, sr_imgs, sr_cbcr_imgs=None):
    to_pil = transforms.ToPILImage()
    count = 0

    for i in range(len(sr_imgs)):
        count += 1
        print "before:", sr_imgs[i].size()
        img = to_pil(sr_imgs[i])
        print "after:", img.size

        if sr_cbcr_imgs != None:
            print "y size:", img.size
       	    print "cb size:", sr_cbcr_imgs[i][0].size
       	    print "cr size:", sr_cbcr_imgs[i][1].size

            img = Image.merge("YCbCr", (img, sr_cbcr_imgs[i][0], sr_cbcr_imgs[i][1]))
            img = img.convert("RGB")

        img.save("%s/%s/sr_img_%03d.png" % (args.out_folder, dataset_name, count))
