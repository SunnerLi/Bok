import torch.nn as nn
import copy

class BokException(Exception):
    """
        The exception which is used in Bok repo uniquely
    """
    def __init__(self, remind):
        print("[ Bok ] Error >> ", remind)

def INFO(string):
    print(" {:>10} ".format("[ Bok ]  "), string)

def load_state_dict(source, target):
    """
        將source模型裡面的所有參數，複製到target模型裏面，
        如果channel不一樣，則只會部份引入
        * 注意：這個函式的回傳值才是初始化結果，務必使用另一個物件來承接！！！！

        Arg:    source      - pretrained-model的nn.Module物件
                target      - 你想要賦予的model之nn.Module物件
        Ret:    被初始化的nn.Module物件
    """
    # -------------------------------------------------------------------
    #                          賦予參數流程
    # 由於直接賦予nn.Parameter數值，將會導致再次創建新的Variable，
    # 因此這個函式採取的作法是，先初始化一個有新Variable的模型，
    # 然後在用另外一個複製的模型來承接參數
    # 還有另一點需注意！由於不支援直接對有梯度的Variable賦值，
    # 因此需要先取消梯度功能，然後在回復它
    # -------------------------------------------------------------------
    # (1) 複製結果模型
    result = copy.deepcopy(target)

    # (2) 複製參數，但是會長出新的Variable
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
                    
    # (3) 把參數放入真正的結果模型中
    result.load_state_dict(target.state_dict())

    # (4) 恢復梯度儲存功能
    for param in result.parameters():
        param.requires_grad = True
    return result

if __name__ == '__main__':
    net1 = nn.Conv2d(2, 5, 3, 1, 1)
    net2 = nn.Conv2d(1, 4, 3, 1, 1)
    for param in net2.parameters():
        print(param)
    load_state_dict(source = net1, target = net2)
    for param in net2.parameters():
        print(param)
