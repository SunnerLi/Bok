# from BagOK.bagOfKernel import __summary
import torch.nn as nn
import torch
import copy

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 4, 3, 1, 1)
        self.max1  = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(4, 8, 3, 1, 1)
        self.max2  = nn.MaxPool2d(2, 2)

    def forward(self, x):
        return self.max2(self.conv2(self.max1(self.conv1(x))))

def load_state_dict(source, target):
    result = copy.deepcopy(target)
    for source_p, target_p in zip(source.parameters(), target.parameters()):
        if source_p.size() == target_p.size():
            target_p.requires_grad = False
            _ = source_p[:]
            target_p[:] = _
        else:
            target_p.requires_grad = False
            if len(source_p.size()) == 4:
                output_channel = min(source_p.size(0), target_p.size(0))
                input_channel = min(source_p.size(1), target_p.size(1))
                _ = source_p[:output_channel, :input_channel, :, :]
                target_p[:output_channel, :input_channel, :, :] = _.data
            elif len(source_p.size()) == 2:
                output_channel = min(source_p.size(0), target_p.size(0))
                input_channel = min(source_p.size(1), target_p.size(1))
                _ = source_p[:output_channel, :input_channel]
                target_p[:output_channel, :input_channel] = _.data
            elif len(source_p.size()) == 1:
                output_channel = min(source_p.size(0), target_p.size(0))
                _ = source_p[:output_channel]
                target_p[:output_channel] = _.data
                    
    
    result.load_state_dict(target.state_dict())

    for param in result.parameters():
        param.requires_grad = True
    del target
    target = result
    for param in target.parameters():
        print(param.requires_grad)

    return result

def show(net):
    for param in net.parameters():
        print(param)

# net1 = Net().cuda()
# net2 = Net().cuda()
# __summary(net1, [3, 28, 28], verbose = True)
# # print(net2.state_dict())
# net1.load_state_dict(net2.state_dict())
net1 = nn.Conv2d(2, 5, 3, 1, 1)
net2 = nn.Conv2d(1, 4, 3, 1, 1)
# net1.load_state_dict(net2.state_dict())
show(net1)
print('---------------------')
show(net2)
for param in net2.parameters():
    print(param.requires_grad)
net2 = load_state_dict(source = net1, target = net2)
show(net2)
for param in net2.parameters():
    print(param.requires_grad)

# net1.state_dict().update(net2.state_dict())
# net1.load_state_dict(net2.state_dict())