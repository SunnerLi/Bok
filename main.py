import BagOK.bagOfKernel as bok
import torch.nn as nn
import time

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

    def forward(self, x):
        return self.main(x)

if __name__ == '__main__':
    net = Net()
    start = time.time()
    bok.init(net, [3, 224, 224], net_type = None, verbose = True)