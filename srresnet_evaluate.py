import options, data, models, helpers
from torch.autograd import Variable
import resource


args = options.SRResNetTestOptions().parse()
datasets = data.build_evaluation_dataset(args)
srresnet = models.SRResNet(args)
sr_imgs = []

for dataset_name, (hr_imgs, lr_imgs) in datasets.iteritems():
    assert len(hr_imgs) == len(lr_imgs)

    total_psnr = 0.0

    for i in range(len(lr_imgs)):
        lr_img = Variable(lr_imgs[i].unsqueeze(0), volatile=True)
        hr_img = Variable(hr_imgs[i].unsqueeze(0), volatile=True)
        sr_img = srresnet.super_resolve(lr_img)
        print 'Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        sr_imgs.append(sr_img.data[0])
        total_psnr += helpers.evaluate_psnr(sr_img, hr_img)

    total_psnr /= len(lr_imgs)
    helpers.save_sr_results(args, dataset_name, sr_imgs)
    sr_imgs = []

    print "Dataset " + dataset_name + " PSNR: " + str(total_psnr)
