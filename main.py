from tool.customVGG import CustomVGG
from BagOK.bagOfKernel import init
from BagOK.summary import summary

if __name__ == '__main__':
    net = CustomVGG('./vgg_conv.pth').cuda()
    net_sum = summary(net, [3, 224, 224])
    print(type(net_sum))
    init(net, net_sum)