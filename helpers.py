from PIL import Image
from numpy import sqrt, log10
from torch import is_tensor, stack, nn
from torch.autograd import Variable
from torchvision import transforms


def custom_collate(batch):
    to_pil = transforms.ToPILImage()
    to_tensor = transforms.ToTensor()

    hr_imgs, lr_imgs = [], []
    num_images, dim = batch[0].size(0), batch[0].size(2)

    for i in range(len(batch)):
        hr_imgs.append(batch[i])
        lr_batch = [to_tensor(to_pil(image).resize((dim/4, dim/4), Image.BICUBIC)) for image in batch[i]]
        lr_imgs.append(stack(lr_batch))

    hr_imgs = stack(hr_imgs).view(len(batch) * num_images, 3, dim, dim)
    lr_imgs = stack(lr_imgs).view(len(batch) * num_images, 3, dim/4, dim/4)

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


def evaluate_psnr(sr_img, hr_img, r=1):
    mse_loss = nn.MSELoss()
    mse = mse_loss(sr_img, hr_img)
    return 10 * log10((r**2) / mse.data[0])


def save_images(model, args):
    to_pil = transforms.ToPILImage()
    to_tensor = transforms.ToTensor()

    hr_img = Image.open("./Set5/image_SRF_4/img_003_SRF_4_HR.png")
    lr_img = Variable(to_tensor(Image.open("./Set5/image_SRF_4/img_003_SRF_4_LR.png")))

    if args.use_cuda:
        lr_img = lr_img.cuda()

    sr_img = model(lr_img.unsqueeze(0))
    sr_img = to_pil(sr_img.data[0].cpu())
    lr_img = to_pil(lr_img.data.cpu())

    hr_img.save("%s/hr_img.png" % args.out_folder)
    lr_img.save("%s/lr_img.png" % args.out_folder)
    sr_img.save("%s/sr_img.png" % args.out_folder)


def save_sr_results(args, dataset_name, sr_imgs):
    to_pil = transforms.ToPILImage()
    count = 0

    for sr_img in sr_imgs:
        count += 1
        img = to_pil(sr_img)
        img.save("%s/%s/sr_img_%03d.png" % (args.out_folder, dataset_name, count))
