import torch
from torch import nn
import torchvision.models as models
import random


class Masked_Residual_Aggregation(nn.Module):
    def __init__(self, in_channels, num_parts):
        super(Masked_Residual_Aggregation, self).__init__()
        self.in_channels = in_channels
        self.num_parts = num_parts

        # Convolutional 1x1 to get a m Layers(attantion layers)
        self.mask_conv = nn.Conv2d(in_channels, num_parts, kernel_size=1)

        # Convolutional 1x1 to obtain the final h(x) (H x W x (CxM) --> H x W x (C))
        self.final_mask_conv = nn.Conv2d(in_channels * num_parts, in_channels, kernel_size=1)

    def forward(self, x):
        masks = torch.sigmoid(self.mask_conv(x)) # --> H x W x m

        # Expand masks(N_layer) to corresponding size of f(x)
        masks = masks.unsqueeze(2)   # ---> H x W x 1 x m

        # B x C x H x W
        x_exp = x.unsqueeze(1)
        attention_layers = x_exp * masks
        h = attention_layers + x_exp    # --> H x W x C x M

        h = torch.cat([h[:, i, :, :, :] for i in range(self.num_parts)], dim=1)

        h_1 = self.final_mask_conv(h)

        h_output = torch.fft.fft2(h_1, norm='ortho')

        return h_output


class Phase_Based_Augmentation(nn.Module):
    def __init__(self, gamma=0.1):
        super().__init__()
        self.gamma = gamma

    def forward(self, h_freq):
        batch_size, channels, H, W = h_freq.shape

        amplitude = torch.abs(h_freq)
        phase_origin = torch.angle(h_freq)

        h_ran = h_freq[random.randint(0, batch_size-1)] # [channels, H, W]

        phase_random = torch.angle(h_ran)

        phase_new = self.gamma * phase_random.unsqueeze(0) + (1 - self.gamma) * phase_origin

        h_freq_new = amplitude * torch.exp(1j * phase_new)

        return h_freq_new


class Hybrid_Module(nn.Module):
    # in_channels = кол-во каналов из batch из CNN(HxWxC)
    # num_parts кол-во масок в Masked Residual Aggregation, гипер параматр
    # gamma - для фазовой аугментации
    def __init__(self, in_channels, num_parts, gamma):
        super().__init__()
        self.masked_module = Masked_Residual_Aggregation(in_channels, num_parts)
        self.phase_augm = Phase_Based_Augmentation(gamma)

    def forward(self, x):
        x = self.masked_module(x)

        x = self.phase_augm(x)

        x = torch.fft.ifft2(x, norm='ortho').real

        return x


class APFA(nn.Module):
    def __init__(self, in_channels=2048, num_parts=4, gamma=0.1, num_classes=1000):
        super().__init__()

        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.resnet_encoder = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )

        self.hyb_module = Hybrid_Module(in_channels, num_parts, gamma)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(in_channels, num_classes)
        )

    def forward(self, x):
        x = self.resnet_encoder(x)  # [B, 2048, H, W]

        x = self.hyb_module(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.classifier(x)

        return x
