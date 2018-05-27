from tool.customVGG import CustomVGG
from BagOK.bagOfKernel import init
# from BagOK.summary import summary
from BagOK.networkSummary import __summary

if __name__ == '__main__':
    net = CustomVGG('./vgg_conv.pth').cuda()
    net_sum = __summary(net, [3, 224, 224])
    init(net, [3, 224, 224], net_sum)