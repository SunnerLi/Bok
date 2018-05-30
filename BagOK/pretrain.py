import torchvision.models as models

def load(name):
    """
        根據代號名稱，從torchvision中載入相對應的pretrained model
    """
    net = None
    if name == 'vgg11':
        net = models.vgg.vgg11(pretrained = True)
    elif name == 'vgg13':
        net = models.vgg.vgg13(pretrained = True)
    elif name == 'vgg16':
        net = models.vgg.vgg16(pretrained = True)
    elif name == 'vgg19':
        net = models.vgg.vgg19(pretrained = True)
    elif name == 'vgg11_bn':
        net = models.vgg.vgg11_bn(pretrained = True)
    elif name == 'vgg13_bn':
        net = models.vgg.vgg13_bn(pretrained = True)        
    elif name == 'vgg16_bn':
        net = models.vgg.vgg16_bn(pretrained = True)        
    elif name == 'vgg19_bn':
        net = models.vgg.vgg19_bn(pretrained = True)        
    elif name == 'resnet18':
        net = models.resnet.resnet18(pretrained = True)        
    elif name == 'resnet34':
        net = models.resnet.resnet34(pretrained = True)        
    elif name == 'resnet50':
        net = models.resnet.resnet50(pretrained = True)                
    elif name == 'resnet101':
        net = models.resnet.resnet101(pretrained = True)                
    elif name == 'resnet152':
        net = models.resnet.resnet152(pretrained = True)                
    elif name == 'squeezenet1_0':
        net = models.squeezenet.squeezenet1_0(pretrained = True)
    elif name == 'squeezenet1_1':
        net = models.squeezenet.squeezenet1_1(pretrained = True)
    elif name == 'densenet121':
        net = models.densenet.densenet121(pretrained = True)        
    elif name == 'densenet161':
        net = models.densenet.densenet161(pretrained = True)        
    elif name == 'densenet169':
        net = models.densenet.densenet169(pretrained = True)                
    elif name == 'densenet201':
        net = models.densenet.densenet201(pretrained = True)  
    elif name == 'alexnet':
        net = models.alexnet(pretrained = True)        
    else:
        raise Exception('The model with specific symbol %s have not been implemented...' % name)
    return net