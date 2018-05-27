from collections import OrderedDict
from torch.autograd import Variable
import torch.nn as nn
import torch
import json
import os

"""
    這份script定義了所有BoK的操作，
    使用者僅需要呼叫init即可，
    其他函式是提供開發者更新預訓練網路資訊時才需要呼叫
"""

def init(net, summary, net_type = None):
    """
        對於給定的網路進行初始化
        
        Arg:    net     - 想要初始化的網路，為一個nn.Module物件
                summary - 透過pytorch-summary回傳的統計資訊
                type    - 為想要初始化的網路依據名子
    """
    # -----------------------------------------
    #
    # target_summary = None
    # if type is not None:                          // 如果type有指定的話，則過程比較簡單
    #     target_summary = 那個type他的layer順序
    #     if summary == target_summary:
    #         直接把預訓練模型load進來
    # else:                                         // 如果沒指定的話，則要搜索過所有可能的model
    #     diff = len(summary)
    #     for pretrain_model_name in pretiain_model_list:
    #         candidate_summary = pretrain_model_name的layer順序
    #         計算candidate_summary和summary的差異性
    #         if 差異性 < diff:
    #             diff = 差異性
    #             target_summary = candidate_summary
    # load_appropriate(net, target_summary)
    # -----------------------------------------
    target_summary = None
    info = __readInfoJSON()
    if net_type is not None:
        target_summary = info['model_list'][net_type]
        if diff(summary, target_summary) == 0:
            # __directLoad(net, net_type)
            pass
    else:
        name_2_diff = OrderedDict()
        small_diff = 100
        small_model_name = None
        for model_name in info['name_list']:
            candidate_summary = info['model_list'][model_name]
            diff_value = diff(summary, candidate_summary)
            name_2_diff[model_name] = diff_value
            if diff_value < small_diff:
                diff_value = small_diff
                small_model_name = model_name
        __appropriateLoad(net, summary, name_2_diff, info)

def __getLayerInfo(info, net_name, index):
    summary = info['model_list'][net_name]
    for idx, layer in enumerate(summary):
        if idx == index:
            return name, summary[layer]

def __appropriateLoad(net, summary, name_2_diff, info):
    """
        層層尋找誤差最小的來填充
    """
    # 初始化最小的分數
    small_score = None
    for key in info['model_list']:
        small_score = name_2_diff[key]

    # 依分數大到小排出名子序列
    filter_dict = OrderedDict(name_2_diff)
    model_score_list = []
    for i in range(len(info['model_list'])):
        big_score = None
        big_model = None
        for model_name in filter_dict:
            if big_score is None:
                big_score = name_2_diff[model_name]
            else:
                if name_2_diff[model_name] < big_score:
                    big_score = name_2_diff[model_name]
                    big_model = model_name
        model_sort_list.append(big_model)
        filter_dict.pop(big_model)

    # 一層層搜尋最有可能的pre-trained model
    ancenter = []
    for i, layer_name in enumerate(summary):
        layer_type = layer_name.split('-')[0]
        source = 'random'
        for net in model_score_list:
            layer_name, layer_summary = __getLayerInfo(info, net, i)
            if layer_name.split('-')[0] == layer_type and \
                layer_summary['weight_param'] == summary[layer_name]['weight_param']:
                source = net
                break
        ancenter.append(net)        

    # -----------------------------------------
    #       <<  根據祖先list填充權重值 >>
    # 
    # param_list = []
    # for net_name in set(ancenter):
    #     def register_scratch_hook(module):
    #         layer_counter = 0
    #         def scratch_hook(module, input, output):
    #             if ancenter[layer_counter] == net_name:
    #                 param.add(module.state_dict(), layer_counter)
    #             layer_counter += 1

    #     pretrained_model = 把model載入
    #     掛上scratch_hook
    #     pretrained_model.forward()

    # def register_assign_hook(module):
    #     layer_counter = 0
    #     def assign_hook(module, input, output):
    #         module.load_state_dict(param_list[layer_counter])
    #         layer_counter += 1
    # 掛上assign_hook
    # net.forward()
    #
    # -----------------------------------------
    pass

def diff(source, target):
    """
        比較兩個summary差異多大
        順序不一樣或大小不一樣都扣分，滿分為0分
    """
    score = 0
    source_order = source.keys()
    target_order = target.keys()
    for i in range(len(source_order)):
        source_layer_name = source_order[i]
        target_layer_name = target_order[i]
        if source_layer_name != target_layer_name:
            score -= 1
        else:
            if source[source_layer_name]['weight_param'] != target[target_layer_name]['weight_param']:
                score -= 1
    return diff 

def __readInfoJSON(name = './bag.json'):
    """
        << 使用者切勿直接使用此函式 >>

        將JSON檔中的summary資訊讀進來
        讀入的JSON物件格式如下:
        info = {
            'name_list': 為一個list，存放的所有model的名子
            'model_list': {
                <key>: <summary>
                代表每個key是一個model的名子，呼叫info[model_list][<name>]即可提取該網路的summary物件
            }
        }
        使用前需要先透過info[name_list]獲取所有紀錄過的model，在個別進行操作
    """
    if not os.path.exists(name):
        info = dict()
    else:
        info = json.load(open(name, 'r'))
    return info

def __writeInfoJSON(summary, net_name, file_name = './bag.json'):
    """
        << 使用者切勿直接使用此函式 >>

        將網路的summary資訊寫入JSON檔中

        Arg:    summary     - 網路的summary資訊
                net_name    - 要指定寫入info檔中網路的名子
    """
    info = __readInfoJSON(file_name)
    if 'name_list' not in info:
        info['name_list'] = [net_name]
    else:
        info['name_list'].append(net_name)
    if 'model_list' not in info:
        info['model_list'] = {}
    info['model_list'][net_name] = summary
    json.dump(info, open(file_name, 'w'))

def __summary(model, input_size, verbose = False):
    """
        << 使用者建議勿使用此函式 >>

        獲取這個網路的統計資訊
        此函式使用上要小心size_list的格式
        此函式是由pytorch-summary修改而來，原址如下：
        https://github.com/sksq96/pytorch-summary/pull/8/commits/0e6ca4af8bdae390445f1fb25fcf186980e173e7#diff-a494f73aef695b955ce1bc55166b432a


        回傳一個orderdict物件，格式如下：
        sum_dict = [layer]{
            'input_shape': 輸入tensor的大小，為一個list物件
            'output_shape': 輸出tensor的大小，為一個list物件

            # 以下配對是這個layer中有參數才有的：
            'trainable': 參數是否可訓練，為一個bool
            'weight_param': weight的大小，為一個list物件
            'bias_param': bias的大小，為一個list物件
        }

        Arg:    model       - 網路，為一個nn.Module物件
                input_size  - 一個含有輸入大小的list，為list of list，每一個element代表一個輸入的大小（不含batch)
    """
    def register_hook(module):
        apple = "!"
        def hook(module, input, output):
            class_name = str(module.__class__).split('.')[-1].split("'")[0]
            module_idx = len(summary)

            m_key = '%s-%i' % (class_name, module_idx+1)
            summary[m_key] = OrderedDict()
            summary[m_key]['input_shape'] = list(input[0].size())
            summary[m_key]['input_shape'][0] = -1
            if isinstance(output, (list,tuple)):
                summary[m_key]['output_shape'] = [[-1] + list(o.size())[1:] for o in output]
            else:
                summary[m_key]['output_shape'] = list(output.size())
                summary[m_key]['output_shape'][0] = -1

            print(module.state_dict())
            print(apple)
            params = 0
            if hasattr(module, 'weight'):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]['trainable'] = module.weight.requires_grad
                summary[m_key]['weight_param'] = list(module.weight.size())
            if hasattr(module, 'bias') and hasattr(module.bias, 'size'):
                params +=  torch.prod(torch.LongTensor(list(module.bias.size())))
                summary[m_key]['bias_param'] = list(module.bias.size())
            summary[m_key]['nb_params'] = params
                
        if (not isinstance(module, nn.Sequential) and 
           not isinstance(module, nn.ModuleList) and 
           not (module == model)):
            hooks.append(module.register_forward_hook(hook))

    # Form the input list
    input_size = list(input_size)                
    if torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor
    x = Variable(torch.rand(1,*input_size)).type(dtype)
            
    # create properties
    summary = OrderedDict()
    hooks = []
    # register hook
    model.apply(register_hook)
    # make a forward pass
    model(x)
    # remove these hooks
    for h in hooks:
        h.remove()
    # show infomation
    if verbose:
        __showSum(input_size, summary)
    return summary

def __showSum(input_size, summary):
    print('-----------------------------------------------------------------------------------------------------------------------')
    line_new = '{:>20}  {:>25} {:>15} {:>25} {:>25}'.format('Layer (type)', 'Output Shape', 'Param #', 'Weight Param', 'Bias Param')
    print(line_new)
    print('=======================================================================================================================')
    line_new = '{:>20}  {:>25} {:>15}'.format('Input Tensor', str([-1,]+input_size), 'None')
    print(line_new)
    total_params = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = '{:>20}  {:>25} {:>15}'.format(layer, str(summary[layer]['output_shape']), summary[layer]['nb_params'])
        total_params += summary[layer]['nb_params']
        
        if 'trainable' in summary[layer]:
            if summary[layer]['trainable'] == True:
                trainable_params += summary[layer]['nb_params']
        if 'weight_param' in summary[layer]:
            line_new += '{:>25} {:>25}'.format(str(summary[layer]['weight_param']), str(summary[layer]['bias_param']))

        print(line_new)
    print('=======================================================================================================================')
    print('Total params: ' + str(total_params))
    print('Trainable params: ' + str(trainable_params))
    print('Non-trainable params: ' + str(total_params - trainable_params))
    print('-----------------------------------------------------------------------------------------------------------------------')