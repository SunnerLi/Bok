from tool.customVGG import CustomVGG
from BagOK.vgg import vgg16_bn
from BagOK.bagOfKernel import init
# from BagOK.summary import summary
from BagOK.networkSummary import __summary
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)
        self.max1  = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.max2  = nn.MaxPool2d(2, 2)

    def forward(self, x):
        return self.max2(self.conv2(self.max1(self.conv1(x))))

if __name__ == '__main__':
    # net = CustomVGG('./vgg_conv.pth').cuda()
    # net = Net().cuda()
    # net_sum = __summary(net, [3, 224, 224])
    # init(net, [3, 224, 224], net_sum)
    net = vgg16_bn(pretrained = True).cuda()
    net_sum = __summary(net, [3, 244, 244], verbose = True)
    init(net, [3, 224, 224], net_sum)