#cluster stuff, remove this
import sys
sys.path.append("/home/adamvest/lib/python")

import options, data, models, helpers
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms

args = options.SRResNetTestOptions().parse()
srresnet = models.SRResNet(args)

if args.use_cuda:
    cuda.manual_seed(args.seed)
    srresnet.to_cuda()

if args.use_rgb:
    datasets = data.build_rgb_evaluation_dataset(args)

    for dataset_name, (hr_imgs, lr_imgs) in datasets.iteritems():
        total_psnr = 0.0
        sr_imgs = []

        for i in range(len(lr_imgs)):
            lr_img = Variable(lr_imgs[i].unsqueeze(0), volatile=True)
            hr_img = Variable(hr_imgs[i].unsqueeze(0), volatile=True)
            sr_img = srresnet.super_resolve(lr_img)
            sr_imgs.append(sr_img.data[0])
            total_psnr += helpers.evaluate_psnr(sr_img, hr_img, convert_to_ycbcr=True)
            del sr_img

        helpers.save_sr_results(args, dataset_name, sr_imgs)
        total_psnr /= len(lr_imgs)
        print "Dataset " + dataset_name + " PSNR: " + str(total_psnr)
else:
    to_pil = transforms.ToPILImage()

    datasets = data.build_ycbcr_evaluation_dataset(args)

    for dataset_name, (hr_y_imgs, lr_y_imgs, lr_cbcr_imgs) in datasets.iteritems():
        total_psnr = 0.0
        sr_y_imgs, sr_cbcr_imgs = [], []

        for i in range(len(lr_y_imgs)):
            lr_y_img = Variable(lr_y_imgs[i].unsqueeze(0), volatile=True)
            hr_y_img = Variable(hr_y_imgs[i].unsqueeze(0), volatile=True)
            sr_y_img = srresnet.super_resolve(lr_y_img)
            sr_y_imgs.append(sr_y_img.data[0])
            total_psnr += helpers.evaluate_psnr(sr_y_img, hr_y_img)
            del sr_y_img

        for (cb_img, cr_img) in lr_cbcr_imgs:
            cb_img, cr_img = to_pil(cb_img), to_pil(cr_img)
            w, h = cb_img.size
            sr_cb_img = cb_img.resize((4 * w, 4 * h), Image.BICUBIC)
            sr_cr_img = cr_img.resize((4 * w, 4 * h), Image.BICUBIC)
            sr_cbcr_imgs.append((sr_cb_img, sr_cr_img))

        helpers.save_sr_results(args, dataset_name, sr_y_imgs, sr_cbcr_imgs)
        total_psnr /= len(lr_y_imgs)
        print "Dataset " + dataset_name + " PSNR: " + str(total_psnr)
