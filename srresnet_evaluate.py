#cluster stuff, remove this
import sys
sys.path.append("/home/adamvest/lib/python")

import options, data, models, helpers
from torch.autograd import Variable


args = options.SRResNetTestOptions().parse()
datasets = data.build_evaluation_dataset(args)
srresnet = models.SRResNet(args)

for dataset_name, (hr_imgs, lr_imgs) in datasets.iteritems():
    total_psnr = 0.0
    sr_imgs = []

    for i in range(len(lr_imgs)):
        lr_img = Variable(lr_imgs[i].unsqueeze(0), volatile=True)
        hr_img = Variable(hr_imgs[i].unsqueeze(0), volatile=True)
        sr_img = srresnet.super_resolve(lr_img)
        sr_imgs.append(sr_img.data[0])
        total_psnr += helpers.evaluate_psnr(sr_img, hr_img)
	del lr_img, hr_img, sr_img

    helpers.save_sr_results(args, dataset_name, sr_imgs)
    total_psnr /= len(lr_imgs)
    print "Dataset " + dataset_name + " PSNR: " + str(total_psnr)
