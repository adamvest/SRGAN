import options, data, models, helpers
from torch.autograd import Variable


args = options.SRResNetTestOptions().parse()
datasets = data.build_evaluation_dataset(args)
srresnet = models.SRResNet(args)

for dataset_name, (hr_imgs, lr_imgs) in datasets.iteritems():
    assert len(hr_imgs) == len(lr_imgs)

    total_psnr = 0.0

    for i in range(len(lr_imgs)):
        lr_img = Variable(lr_imgs[i])
        hr_img = Variable(hr_imgs[i])
        sr_img = srresnet.super_resolve(lr_img)
        total_psnr += helpers.evaluate_psnr(sr_img, hr_img)

    total_psnr /= len(lr_imgs)

    print "Dataset " + dataset_name + " PSNR: " + str(total_psnr)
