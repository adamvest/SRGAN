#cluster stuff, remove this
import sys
sys.path.append("/home/adamvest/lib/python")

import options, data, models, helpers
from torch import cuda
from torch.autograd import Variable

args = options.SRResNetTestOptions().parse()
srresnet = models.SRResNet(args)

if args.use_cuda:
    cuda.manual_seed(args.seed)
    srresnet.to_cuda()

datasets = data.build_evaluation_dataset(args)

for dataset_name, (hr_imgs, lr_imgs) in datasets.iteritems():
    total_psnr, total_ssim = 0.0, 0.0
    sr_imgs = []

    for i in range(len(lr_imgs)):
        lr_img = Variable(lr_imgs[i].unsqueeze(0), volatile=True)
        sr_img = srresnet.super_resolve(lr_img)
        sr_imgs.append(sr_img.data[0])
        psnr, ssim = helpers.compute_statistics(sr_img, hr_imgs[i])
        total_psnr += psnr
        total_ssim += ssim
        del sr_img

    helpers.save_sr_results(args, dataset_name, sr_imgs)
    print "Dataset " + dataset_name + " PSNR: " + str(total_psnr / len(lr_imgs)) + \
            " SSIM: " + str(total_ssim / len(lr_imgs))
