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

transform = transforms.Compose([
                transforms.Scale(args.load_size),
                data.MultipleRandomCrops(args.crop_size, args.num_crops),
                data.MultipleImagesToTensor()
            ])

if args.dataset == "ImageNet":
    dataset = data.ImagenetDataset(args, transform=transform)
elif args.dataset == "BSD100":
    dataset = data.BSD100Dataset(args, transform=transform)
else:
    raise ValueError("Dataset not yet implemented")

data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        collate_fn=helpers.custom_collate, pin_memory=args.use_cuda)

num_batches = len(data_loader)
num_epochs = int(ceil(args.num_iter / num_batches))
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

        srresnet.train_on_batch(epoch, num_epochs, batch_num, num_batches, hr_imgs, lr_imgs)

        if num_iter >= args.num_iter:
            break

    srresnet.save_model()
    helpers.save_images(srresnet.model, args)
    minutes = int(time.time() - epoch_start_time) / 60
    print "\nEpoch time: " + str(minutes) + " minutes\n"

    if num_iter >= args.num_iter:
        break

print "\nSpecified number of update iterations reached, training has ended!\n"

