from BagOK.bagOfKernel import init, __readInfoJSON, __writeInfoJSON
from BagOK.networkSummary import __summary
from BagOK.summary import summary
from customVGG import CustomVGG
from BagOK.utils import INFO
import argparse

"""
    此份文件定義了所有pre-trained model的操作
"""

SKIP = False

def parse():
    global SKIP
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip', default = False, type = bool, help = 'If skipping the exist model')
    args = parser.parse_args()
    SKIP = args.skip

def recordVGG(info):
    global SKIP
    import torchvision.models.vgg as vggGen
    
    if not (SKIP and 'vgg11' in info['name_list']):
        INFO("proceeding for VGG11...")
        net = vggGen.vgg11(pretrained = True).cuda()
        sum = __summary(net, [3, 224, 224], verbose = True)
        __writeInfoJSON(sum, 'vgg11')
    else:
        INFO("Skip VGG11")

    if not (SKIP and 'vgg13' in info['name_list']):
        INFO("proceeding for VGG13...")
        net = vggGen.vgg13(pretrained = True).cuda()
        sum = __summary(net, [3, 224, 224], verbose = True)
        __writeInfoJSON(sum, 'vgg13')
    else:
        INFO("Skip VGG13")

    if not (SKIP and 'vgg16' in info['name_list']):
        INFO("proceeding for VGG16...")
        net = vggGen.vgg16(pretrained = True).cuda()
        sum = __summary(net, [3, 224, 224], verbose = True)
        __writeInfoJSON(sum, 'vgg16')
    else:
        INFO("Skip VGG16")

    if not (SKIP and 'vgg19' in info['name_list']):
        INFO("proceeding for VGG19...")
        net = vggGen.vgg19(pretrained = True).cuda()
        sum = __summary(net, [3, 224, 224], verbose = True)
        __writeInfoJSON(sum, 'vgg19')
    else:
        INFO("Skip VGG19")

    if not (SKIP and 'vgg11_bn' in info['name_list']):
        INFO("proceeding for VGG11_bn...")
        net = vggGen.vgg11_bn(pretrained = True).cuda()
        sum = __summary(net, [3, 224, 224], verbose = True)
        __writeInfoJSON(sum, 'vgg11_bn')
    else:
        INFO("Skip VGG11_bn")

    if not (SKIP and 'vgg13_bn' in info['name_list']):
        INFO("proceeding for VGG13_bn...")
        net = vggGen.vgg13_bn(pretrained = True).cuda()
        sum = __summary(net, [3, 224, 224], verbose = True)
        __writeInfoJSON(sum, 'vgg13_bn')
    else:
        INFO("Skip VGG13_bn")

    if not (SKIP and 'vgg16_bn' in info['name_list']):
        INFO("proceeding for VGG16_bn...")
        net = vggGen.vgg16_bn(pretrained = True).cuda()
        sum = __summary(net, [3, 224, 224], verbose = True)
        __writeInfoJSON(sum, 'vgg16_bn')
    else:
        INFO("Skip VGG16_bn")

    if not (SKIP and 'vgg19_bn' in info['name_list']):
        INFO("proceeding for VGG19_bn...")
        net = vggGen.vgg19_bn(pretrained = True).cuda()
        sum = __summary(net, [3, 224, 224], verbose = True)
        __writeInfoJSON(sum, 'vgg19_bn')
    else:
        INFO("Skip VGG19_bn")

def recordResNet(info):
    global SKIP
    import torchvision.models.resnet as resGen

    if not (SKIP and 'resnet18' in info['name_list']):
        INFO("proceeding for ResNet18")
        net = resGen.resnet18(pretrained = True).cuda()
        sum = __summary(net, [3, 224, 224], verbose = True)
        __writeInfoJSON(sum, 'resnet18')
    else:
        INFO("Skip ResNet18")

    if not (SKIP and 'resnet34' in info['name_list']):
        INFO("proceeding for ResNet34")
        net = resGen.resnet34(pretrained = True).cuda()
        sum = __summary(net, [3, 224, 224], verbose = True)
        __writeInfoJSON(sum, 'resnet34')
    else:
        INFO("Skip ResNet34")

    if not (SKIP and 'resnet50' in info['name_list']):
        INFO("proceeding for ResNet50")
        net = resGen.resnet50(pretrained = True).cuda()
        sum = __summary(net, [3, 224, 224], verbose = True)
        __writeInfoJSON(sum, 'resnet50')
    else:
        INFO("Skip ResNet50")

    if not (SKIP and 'resnet101' in info['name_list']):
        INFO("proceeding for ResNet101")
        net = resGen.resnet101(pretrained = True).cuda()
        sum = __summary(net, [3, 224, 224], verbose = True)
        __writeInfoJSON(sum, 'resnet101')
    else:
        INFO("Skip ResNet101")

    if not (SKIP and 'resnet152' in info['name_list']):
        INFO("proceeding for ResNet152")
        net = resGen.resnet152(pretrained = True).cuda()
        sum = __summary(net, [3, 224, 224], verbose = True)
        __writeInfoJSON(sum, 'resnet152')
    else:
        INFO("Skip ResNet152")

def recordSqueeze(info):
    global SKIP
    import torchvision.models.squeezenet as sqGen

    if not (SKIP and 'squeezenet1_0' in info['name_list']):
        INFO("proceeding for squeezenet1_0")
        net = sqGen.squeezenet1_0(pretrained = True).cuda()
        sum = __summary(net, [3, 224, 224], verbose = True)
        __writeInfoJSON(sum, 'squeezenet1_0')
    else:
        INFO("Skip squeezenet1_0")

    if not (SKIP and 'squeezenet1_1' in info['name_list']):
        INFO("proceeding for squeezenet1_1")
        net = sqGen.squeezenet1_1(pretrained = True).cuda()
        sum = __summary(net, [3, 224, 224], verbose = True)
        __writeInfoJSON(sum, 'squeezenet1_1')
    else:
        INFO("Skip squeezenet1_1")

def recordDense(info):
    global SKIP
    import torchvision.models.densenet as denGen

    if not (SKIP and 'densenet121' in info['name_list']):
        INFO("proceeding for DenseNet121")
        net = denGen.densenet121(pretrained = True).cuda()
        sum = __summary(net, [3, 224, 224], verbose = True)
        __writeInfoJSON(sum, 'densenet121')
    else:
        INFO("Skip DenseNet121")

    if not (SKIP and 'densenet161' in info['name_list']):
        INFO("proceeding for DenseNet161")
        net = denGen.densenet161(pretrained = True).cuda()
        sum = __summary(net, [3, 224, 224], verbose = True)
        __writeInfoJSON(sum, 'densenet161')
    else:
        INFO("Skip DenseNet161")

    if not (SKIP and 'densenet169' in info['name_list']):
        INFO("proceeding for DenseNet169")
        net = denGen.densenet169(pretrained = True).cuda()
        sum = __summary(net, [3, 224, 224], verbose = True)
        __writeInfoJSON(sum, 'densenet169')
    else:
        INFO("Skip DenseNet169")

    if not (SKIP and 'densenet201' in info['name_list']):
        INFO("proceeding for DenseNet201")
        net = denGen.densenet201(pretrained = True).cuda()
        sum = __summary(net, [3, 224, 224], verbose = True)
        __writeInfoJSON(sum, 'densenet201')
    else:
        INFO("Skip DenseNet201")

def recordAlex(info):
    import torchvision.models.alexnet as alexGen

    if not (SKIP and 'alexnet' in info['name_list']):
        INFO("proceeding for AlexNet")
        net = alexGen(pretrained = True).cuda()
        sum = __summary(net, [3, 224, 224], verbose = True)
        __writeInfoJSON(sum, 'alexnet')
    else:
        INFO("Skip AlexNet")

if __name__ == '__main__':
    parse()
    info = __readInfoJSON()
    recordVGG(info)
    recordResNet(info)
    recordSqueeze(info)
    recordDense(info)
    recordAlex(info)