import math
import torch
from torch import nn
from torchvision import models
from ..model.apha import APFA


def spatial_pyramid_pool(previous_conv, num_sample, previous_conv_size, out_pool_size):
    '''
    previous_conv: a tensor vector of previous convolution layer
    num_sample: an int number of image in the batch
    previous_conv_size: an int vector [height, width] of the matrix features size of previous convolution layer
    out_pool_size: a int vector of expected output size of max pooling layer

    returns: a tensor vector with shape [1 x n] is the concentration of multi-level pooling
    '''
    for i in range(len(out_pool_size)):
        h_wid = int(math.ceil(previous_conv_size[0] / out_pool_size[i]))
        w_wid = int(math.ceil(previous_conv_size[1] / out_pool_size[i]))
        h_pad = (h_wid*out_pool_size[i] - previous_conv_size[0] + 1) // 2
        w_pad = (w_wid*out_pool_size[i] - previous_conv_size[1] + 1) // 2

        print(h_wid, w_wid, h_pad, w_pad)
        maxpool = nn.MaxPool2d((h_wid, w_wid), stride=(h_wid, w_wid), padding=(h_pad, w_pad))
        x = maxpool(previous_conv)
        if(i == 0):
            spp = x.view(num_sample,-1)
            #print("spp size:",spp.size())
        else:
            # print("size:",spp.size())
            spp = torch.cat((spp,x.view(num_sample,-1)), 1)
    return spp


# TODO реализовать это с ALEXNet или с ResNet
# чтобы работало на птицах(CUB200_2011, AWA2)
# https://arxiv.org/pdf/1406.4729v4
# в качестве backbone можно использовать пока ALEXNet чисто для простоты
# возможно это даст чуть более лучшие результаты, если я просто возьму заранее готовые веса.
# или придется обучать все сначала


class Model_with_pyramid(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.backbone = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)

        print(self.backbone.children())
    def forward(self, x):
        pass


if __name__ == "__main__":
    # example
    a = torch.rand(2, 3, 7, 10)
    print(spatial_pyramid_pool(a, 2, [7, 10], [1]).shape)

    b = torch.rand(1, 2)
    c = torch.rand(1, 4)

    print(torch.cat((b, c), dim=1))

    model = Model_with_pyramid()
