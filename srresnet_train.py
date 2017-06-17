#cluster stuff, remove this
import sys
sys.path.append("/home/adamvest/lib/python")

import time
import options, data, helpers, models
from numpy import ceil
from torch import cuda
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import transforms


args = options.SRResNetTrainOptions().parse()

transform_list = []
if args.use_rgb:
    transform_list += [data.CheckImageIsRGB()]
else:
    transform_list += [data.ExtractYChannel()]

transform_list += [data.MultipleRandomCrops(args.crop_size, args.num_crops), data.MultipleImagesToTensor()]
transform = transforms.Compose(transform_list)

if args.dataset == "ImageNet":
    dataset = data.ImagenetDataset(args, transform=transform)
elif args.dataset == "BSD100":
    dataset = data.BSD100Dataset(args, transform=transform)
else:
    raise ValueError("Dataset not yet implemented")

data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        collate_fn=helpers.custom_collate, pin_memory=args.use_cuda)

iter_per_epoch = len(data_loader)
num_epochs = int(ceil(args.num_iter / iter_per_epoch))
num_iter = 0

srresnet = models.SRResNet(args)

if args.use_cuda:
    cuda.manual_seed(args.seed)
    srresnet.to_cuda()

for epoch in range(1, num_epochs + 1):
    epoch_start_time = time.time()

    for batch_num, (hr_imgs, lr_imgs) in enumerate(data_loader):
        num_iter += 1
        hr_imgs = Variable(hr_imgs)
        lr_imgs = Variable(lr_imgs)

        srresnet.train_on_batch(epoch, num_epochs, batch_num, iter_per_epoch, hr_imgs, lr_imgs)

        if num_iter >= args.num_iter:
            break

    srresnet.save_model()
    helpers.save_images(srresnet.get_model(), args)
    minutes = int(time.time() - epoch_start_time) / 60
    print "\nEpoch time: " + str(minutes) + " minutes\n"

    if num_iter >= args.num_iter:
        break

print "\nSpecified number of update iterations reached, training has ended!\n"
