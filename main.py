from tool.customVGG import CustomVGG
import BagOK.bagOfKernel as bok
import torch.nn as nn
import time

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)
        # self.relu1 = nn.ReLU()
        # self.max1  = nn.MaxPool2d(2, 2)
        # self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        # self.relu2 = nn.ReLU()
        # self.max2  = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(3, 4, 3, 1, 1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(4, 8, 3, 1, 1)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        # return self.max2(self.relu2(self.conv2(self.max1(self.relu1(self.conv1(x))))))
        return self.relu2(self.conv2(self.relu1(self.conv1(x))))

if __name__ == '__main__':
    net = Net()
    start = time.time()
    bok.init(net, [3, 224, 224], net_type = 'vgg16', verbose = True)