from customVGG import CustomVGG
from BagOK.bagOfKernel import init, __readInfoJSON, __writeInfoJSON, __summary
from BagOK.summary import summary

if __name__ == '__main__':
    net = CustomVGG('./vgg_conv.pth').cuda()
    # net_sum = summary(net, [3, 224, 224])
    sum = __summary(net, [3, 224, 224], verbose = True)
    __writeInfoJSON(sum, 'vgg')