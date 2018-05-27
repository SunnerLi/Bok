from BagOK.bagOfKernel import __summary
import torch.nn as nn
import torch

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 4, 3, 1, 1)
        self.max1  = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(4, 8, 3, 1, 1)
        self.max2  = nn.MaxPool2d(2, 2)

    def forward(self, x):
        return self.max2(self.conv2(self.max1(self.conv1(x))))

net1 = Net().cuda()
net2 = Net().cuda()
__summary(net1, [3, 28, 28], verbose = True)
# print(net2.state_dict())
net1.load_state_dict(net2.state_dict())