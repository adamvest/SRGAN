import argparse


class SRGANTrainOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument('--train_path', default="./BSD100_SR/image_SRF_4/", help="path to training data")
        self.parser.add_argument('--use_cuda', type=int, default=0, help="use GPU to train model")
        self.parser.add_argument('--batch_size', type=int, default=1, help="training batch size")
        self.parser.add_argument('--use_mse', type=int, default=1, help="use MSE content loss (set to 0 for Vgg54)")
        self.parser.add_argument('--adv_weight', type=float, default=1e-3, help="weight for adversarial loss")
        self.parser.add_argument('--lr', type=float, default=1e-4, help="initial learning rate for Adam optimizer")
        self.parser.add_argument('--total_iter', type=float, default=2e5, help="number of update iterations to train for")
        self.parser.add_argument('--iter_to_lr_decay', type=float, default=1e5, help="number of update iterations before decaying the lr")
        self.parser.add_argument('--load_size', type=int, default=256, help="scale images to this size before cropping")
        self.parser.add_argument('--crop_size', type=int, default=96, help="randomly crop images to this size")
        self.parser.add_argument('--num_crops', type=int, default=16, help="number of random crops to take from each training example")
        self.parser.add_argument('--gen', required=True, help="path to generator weights (use SRResNet weights unless continuing training)")
        self.parser.add_argument('--disc', default="", help="path to discriminator weights")
        self.parser.add_argument('--out_folder', default="./srgan_out", help="directory to store model weights and images")
        self.parser.add_argument('--num_workers', type=int, default=1, help="number of workers used to load training data")
        self.parser.add_argument('--mode', default="train", help="indicates training mode")

    def parse(self):
        return self.parser.parse_args()


class SRGANTestOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument('--lr_image', requried=True, help="path to image to super-resolve")
        self.parser.add_argument('--use_cuda', type=int, default=0, help="use GPU to perform super-resolution")
        self.parser.add_argument('--gen', requried=True, help="path to generator weights")
        self.parser.add_argument('--out_folder', default="./srgan_out", help="directory to store super-resolved image")
        self.parser.add_argument('--mode', default="test", help="indicates testing mode")

    def parse(self):
        return self.parser.parse_args()


class SRResNetTrainOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument('--train_path', default="./BSD100_SR/image_SRF_4/", help="path to training data")
        self.parser.add_argument('--use_cuda', type=int, default=0, help="use GPU to train model")
        self.parser.add_argument('--seed', type=int, default=21, help="random seed")
        self.parser.add_argument('--batch_size', type=int, default=1, help="training batch size")
        self.parser.add_argument('--use_mse', type=int, default=1, help="use MSE content loss (set to 0 for Vgg22)")
        self.parser.add_argument('--tv_weight', type=float, default=2e-8, help="weight for total variation loss when using Vgg loss")
        self.parser.add_argument('--lr', type=float, default=1e-4, help="learning rate for Adam optimizer")
        self.parser.add_argument('--num_iter', type=float, default=1e6, help="number of update iterations to train for")
        self.parser.add_argument('--load_size', type=int, default=256, help="scale images to this size before cropping")
        self.parser.add_argument('--crop_size', type=int, default=96, help="randomly crop images to this size")
        self.parser.add_argument('--num_crops', type=int, default=16, help="number of random crops to take from each training example")
        self.parser.add_argument('--model', default="", help="path to model weights")
        self.parser.add_argument('--out_folder', default="./srresnet_out", help="directory to store super-resolved image")
        self.parser.add_argument('--num_workers', type=int, default=1, help="number of workers used to load training data")
        self.parser.add_argument('--mode', default="train", help="indicates training mode")

    def parse(self):
        return self.parser.parse_args()


class SRResNetTestOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument('--lr_image', requried=True, help="path to image to super-resolve")
        self.parser.add_argument('--use_cuda', type=int, default=0, help="use GPU to perform super-resolution")
        self.parser.add_argument('--gen', requried=True, help="path to generator weights")
        self.parser.add_argument('--out_folder', default="./srresnet_out", help="directory to store super-resolved image")
        self.parser.add_argument('--mode', default="test", help="indicates testing mode")

    def parse(self):
        return self.parser.parse_args()
