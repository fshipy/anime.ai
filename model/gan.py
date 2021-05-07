import torch
import torch.nn as nn

# reference: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

# TODO: try with StyleGAN
# reference: https://towardsdatascience.com/generating-anime-characters-with-stylegan2-6f8ae59e237b

class Generator(nn.Module):
    def __init__(self, in_channel=100, feature_map=64, out_channel=3):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(in_channel, feature_map * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_map * 8),
            nn.ReLU(True),
            # state size. (feature_map*8) x 4 x 4
            nn.ConvTranspose2d(feature_map * 8, feature_map * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map * 16),
            nn.ReLU(True),
            # state size. (feature_map*16) x 8 x 8
            nn.ConvTranspose2d( feature_map * 16, feature_map * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map * 8),
            nn.ReLU(True),
            # state size. (feature_map*8) x 16 x 16
            nn.ConvTranspose2d( feature_map * 8, feature_map * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map * 4),
            nn.ReLU(True),
            # state size. (feature_map*4) x 32 x 32
            nn.ConvTranspose2d( feature_map * 4, feature_map * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map * 2),
            nn.ReLU(True),
            # state size. (feature_map*2) x 64 x 64

            nn.ConvTranspose2d( feature_map * 2, feature_map, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map),
            nn.ReLU(True),
            # state size. (feature_map) x 128 x 128
            nn.ConvTranspose2d( feature_map, out_channel, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (out_channel) x 256 x 256
        )

    def forward(self, input):
        return self.main(input)


# custom weights initialization called on generator model
def weights_init(model):
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)