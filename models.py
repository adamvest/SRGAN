import helpers
import torch
import torch.nn as nn
import torchvision.models as mod


class SRGAN():
    def __init__(self, args):
        self.args = args

        self.generator = SRGAN_Generator(args)
        if args.gen == "":
            self.generator.apply(helpers.weights_init)
        else:
            self.generator.load_state_dict(torch.load(args.gen, map_location=lambda storage, loc: storage))

        if args.mode == "train":
            self.discriminator = SRGAN_Discriminator(args)
            if args.disc == "":
                self.discriminator.apply(helpers.weights_init)
            else:
                self.discriminator.load_state_dict(torch.load(args.disc, map_location=lambda storage, loc: storage))

            self.adversarial_loss = nn.BCELoss()

            if args.use_mse:
                self.content_loss = nn.MSELoss()
            else:
                self.content_loss = Vgg54Loss()

            self.gen_opt = torch.optim.Adam(self.generator.parameters(), lr=args.lr)
            self.disc_opt = torch.optim.Adam(self.discriminator.parameters(), lr=args.lr)
            self.labels = torch.autograd.Variable(torch.FloatTensor(args.batch_size), requires_grad=False)
        else:
            self.generator.eval()

    def anneal_lr(self, val=10):
        self.args.lr /= val
        self.gen_opt = torch.optim.Adam(self.generator.parameters(), lr=self.args.lr)
        self.disc_opt = torch.optim.Adam(self.discriminator.parameters(), lr=self.args.lr)

    def train_on_batch(self, epoch, num_epochs, batch_num, num_batches, hr_imgs, lr_imgs):
        if self.args.mode == "train":
            #train discriminator
            self.discriminator.zero_grad()
            self.labels.data.resize_(hr_imgs.size(0)).fill_(1)
            output = self.discriminator(hr_imgs)
            loss_d1 = self.adversarial_loss(output, self.labels)
            loss_d1.backward()

            self.labels.data.resize_(lr_imgs.size(0)).fill_(0)
            sr_imgs = self.generator(lr_imgs)
            output = self.discriminator(sr_imgs.detach())
            loss_d2 = self.adversarial_loss(output, self.labels)
            loss_d2.backward()
            loss_d = loss_d1 + loss_d2
            self.disc_opt.step()

            #train generator
            self.generator.zero_grad()
            self.labels.data.fill_(1)
            output = self.discriminator(sr_imgs)
            loss_g = self.content_loss(sr_imgs, hr_imgs) + \
                            self.args.adv_weight * self.adversarial_loss(output, self.labels)
            loss_g.backward()
            self.gen_opt.step()

            print("[%d/%d][%d/%d] Loss_Gen: %.4f Loss_Disc: %.4f"
                    % (epoch, num_epochs, batch_num + 1, num_batches, loss_g.data[0], loss_d.data[0]))
        else:
            raise ValueError("SRGAN not declared in train mode")

    def super_resolve(self, lr_img):
        if self.args.mode == "test":
            return self.generator(lr_img)
        else:
            raise ValueError("SRGAN not declared in test mode")

    def get_generator(self):
        return self.generator

    def save_models(self):
        torch.save(self.generator.state_dict(), "%s/generator_weights.pth" % self.args.out_folder)
        torch.save(self.discriminator.state_dict(), "%s/discriminator_weights.pth" % self.args.out_folder)

    def to_cuda(self):
        self.generator.cuda()

        if self.args.mode == "train":
            self.discriminator.cuda()
            self.labels.cuda()
            self.adversarial_loss.cuda()
            self.content_loss.cuda()


class SRResNet():
    def __init__(self, args):
        self.args = args

        self.model = SRGAN_Generator(args)
        if args.model == "":
            self.model.apply(helpers.weights_init)
        else:
            self.model.load_state_dict(torch.load(args.model, map_location=lambda storage, loc: storage))

        if args.mode == "train":
            if args.use_mse:
                self.content_loss = nn.MSELoss()
            else:
                self.content_loss = Vgg54Loss()
                self.tv_loss = TotalVariationLoss()

            self.opt = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        else:
            self.model.eval()

    def train_on_batch(self, epoch, num_epochs, batch_num, num_batches, hr_imgs, lr_imgs):
        if self.args.mode == "train":
            if self.args.use_cuda:
                hr_imgs = hr_imgs.cuda()
                lr_imgs = lr_imgs.cuda()

            self.model.zero_grad()
            sr_imgs = self.model(lr_imgs)

            if self.args.use_mse:
                loss = self.content_loss(sr_imgs, hr_imgs)
            else:
                loss = self.content_loss(sr_imgs, hr_imgs) + self.args.tv_weight * self.tv_loss(sr_imgs)

            loss.backward()
            self.opt.step()

            print("[%d/%d][%d/%d] Loss: %.4f"
                    % (epoch, num_epochs, batch_num + 1, num_batches, loss.data[0]))
        else:
            raise ValueError("SRResNet not declared in train mode")

    def super_resolve(self, lr_img):
        if self.args.mode == "test":
            if self.args.use_cuda:
                lr_img = lr_img.cuda()
                sr_img = self.model(lr_img)
                return sr_img.cpu()
            else:
                return self.model(lr_img)
        else:
            raise ValueError("SRResNet not declared in test mode")

    def get_model(self):
        return self.model

    def save_model(self):
        torch.save(self.model.state_dict(), "%s/srresnet_weights.pth" % self.args. out_folder)

    def to_cuda(self):
        self.model.cuda()

        if self.args.mode == "train":
            self.content_loss.cuda()

            if not self.args.use_mse:
                self.tv_loss.cuda()


class SRGAN_Generator(nn.Module):
    def __init__(self, args):
        super(SRGAN_Generator, self).__init__()

        dim = 3 if args.use_rgb else 1

        sequence = [nn.Conv2d(dim, 64, kernel_size=9, padding=4), nn.PReLU()]
        sequence += [GeneratorResidualSubnet()]
        sequence += [GeneratorPixelShuffleBlock(), GeneratorPixelShuffleBlock()]
        sequence += [nn.Conv2d(64, dim, kernel_size=9, padding=4)]

        if args.use_tanh:
            sequence += [nn.Tanh()]

        self.generator = nn.Sequential(*sequence)

    def forward(self, x):
        return self.generator(x)

class GeneratorPixelShuffleBlock(nn.Module):
    def __init__(self, num_filters=64):
        super(GeneratorPixelShuffleBlock, self).__init__()

        self.shuffle_block = nn.Sequential(
                nn.Conv2d(num_filters, num_filters * 4, kernel_size=3, padding=1),
                nn.PixelShuffle(2),
                nn.PReLU()
            )

    def forward(self, x):
        return self.shuffle_block(x)

class GeneratorResidualSubnet(nn.Module):
    def __init__(self, num_blocks=16, num_filters=64):
        super(GeneratorResidualSubnet, self).__init__()

        sequence = []

        for i in range(num_blocks):
            sequence += [GeneratorBlock(num_filters)]

        sequence += [nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1), nn.BatchNorm2d(num_filters)]

        self.subnet = nn.Sequential(*sequence)

    def forward(self, x):
        return x + self.subnet(x)

class GeneratorBlock(nn.Module):
    def __init__(self, num_filters):
        super(GeneratorBlock, self).__init__()

        self.block = nn.Sequential(
                nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
                nn.BatchNorm2d(num_filters),
                nn.PReLU(),
                nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
                nn.BatchNorm2d(num_filters)
            )

    def forward(self, x):
        return x + self.block(x)


class SRGAN_Discriminator(nn.Module):
    def __init__(self, args, num_filters=64):
        super(SRGAN_Discriminator, self).__init__()

        dim = 3 if args.use_rgb else 1

        conv_sequence = [nn.Conv2d(dim, 64, kernel_size=3, padding=1), nn.LeakyReLU(.2, inplace=True),
                            DiscriminatorBlock(num_filters=num_filters, strided=True)]

        for i in range(3):
            num_filters *= 2
            conv_sequence += [DiscriminatorBlock(num_filters=num_filters, strided=False),
                                DiscriminatorBlock(num_filters=num_filters, strided=True)]

        in_dim = ((args.crop_size / 16) ** 2) * num_filters * 2
        linear_sequence = [nn.Linear(in_dim, 1024), nn.LeakyReLU(.2, inplace=True),
                            nn.Linear(1024, 1), nn.Sigmoid()]

        self.conv_subnet = nn.Sequential(*conv_sequence)
        self.linear_subnet = nn.Sequential(*linear_sequence)

    def forward(self, x):
        y = self.conv_subnet(x)
        y = y.view(y.size(0), -1)
        return self.linear_subnet(y)

class DiscriminatorBlock(nn.Module):
    def __init__(self, num_filters, strided):
        super(DiscriminatorBlock, self).__init__()

        if strided:
            self.block = nn.Sequential(
                    nn.Conv2d(num_filters, num_filters * 2, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(num_filters * 2),
                    nn.LeakyReLU(.2, inplace=True)
                )
        else:
            self.block = nn.Sequential(
                    nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
                    nn.BatchNorm2d(num_filters),
                    nn.LeakyReLU(.2, inplace=True)
                )

    def forward(self, x):
        return self.block(x)


class Vgg54Loss(nn.Module):
    def __init__(self, rescaling_factor=12.75):
        super(Vgg54Loss, self).__init__()
        self.vgg = Vgg54()
        self.rescaling_factor = rescaling_factor

    def modified_euclidean_distance(self, x):
        (num_images, _, h, w) = x.size()
        return torch.sum(torch.pow(x, 2)).mul_(1.0 / (num_images * w * h * self.rescaling_factor))

    def __call__(self, sr_imgs, hr_imgs):
        sr_feature_maps = self.vgg(sr_imgs)
        hr_feature_maps = self.vgg(hr_imgs).detach()
        return self.modified_euclidean_distance(sr_feature_maps - hr_feature_maps)

class Vgg54(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg54, self).__init__()
        pretrained_model = mod.vgg19(pretrained=True)
        self.modified_pretrained = nn.Sequential(*list(pretrained_model.features.children())[:-1])

        for (_, layer) in self.modified_pretrained._modules.items():
            layer.requires_grad = requires_grad

    def forward(self, x):
        return self.modified_pretrained(x)


class Vgg22Loss(nn.Module):
    def __init__(self, rescaling_factor=12.75):
        super(Vgg22Loss, self).__init__()
        self.vgg = Vgg22()
        self.rescaling_factor = rescaling_factor

    def modified_euclidean_distance(self, x):
        (num_images, _, h, w) = x.size()
        return torch.sum(torch.pow(x, 2)).mul_(1.0 / (num_images * w * h * self.rescaling_factor))

    def __call__(self, sr_imgs, hr_imgs):
        sr_feature_maps = self.vgg(sr_imgs)
        hr_feature_maps = self.vgg(hr_imgs).detach()
        return self.modified_euclidean_distance(sr_feature_maps - hr_feature_maps)

class Vgg22(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg22, self).__init__()
        pretrained_model = mod.vgg19(pretrained=True)
        self.modified_pretrained = nn.Sequential(*list(pretrained_model.features.children())[:9])

        for (_, layer) in self.modified_pretrained._modules.items():
            layer.requires_grad = requires_grad

    def forward(self, x):
        return self.modified_pretrained(x)


class TotalVariationLoss(nn.Module):
    def __init__(self):
        super(TotalVariationLoss, self).__init__()

    def __call__(self, imgs):
        pixel_diff1 = imgs[:, :, 1:, :] - imgs[:, :, :-1, :]
        pixel_diff2 = imgs[:, :, :, 1:] - imgs[:, :, :, :-1]
        return torch.sum(torch.abs(pixel_diff1)) + torch.sum(torch.abs(pixel_diff2))
