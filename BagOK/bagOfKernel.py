from .networkSummary import __summary
from collections import OrderedDict
from torch.autograd import Variable
from .pretrain import load
# import networkSummary.__summary
import torch.nn as nn
import torch
import json
import os

"""
    這份script定義了所有BoK的操作，
    使用者僅需要呼叫init即可，
    其他函式是提供開發者更新預訓練網路資訊時才需要呼叫
"""

layer_count = 0

def init(net, input_size, summary, net_type = None):
    """
        對於給定的網路進行初始化
        
        Arg:    net         - 想要初始化的網路，為一個nn.Module物件
                input_size  - 
                summary     - 透過pytorch-summary回傳的統計資訊
                type        - 為想要初始化的網路依據名子
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
            print('Direct assign!')
            net.load_state_dict(load(net_type).state_dict())
            return
    else:
        name_2_diff = OrderedDict()
        small_diff = 100
        for model_name in info['name_list']:
            candidate_summary = info['model_list'][model_name]
            diff_value = diff(summary, candidate_summary)
            name_2_diff[model_name] = diff_value
            if diff_value < small_diff:
                diff_value = small_diff
        __appropriateLoad(net, input_size, summary, name_2_diff, info)

def __getLayerInfo(info, net_name, index):
    summary = info['model_list'][net_name]['net']
    for idx, layer in enumerate(summary):
        if idx == index:
            return layer, summary[layer]

def __appropriateLoad(net, net_input_size, summary, name_2_diff, info):
    """
        層層尋找誤差最小的來填充
    """
    # 依分數大到小排出名子序列
    filter_dict = OrderedDict(name_2_diff)
    model_sort_list = []
    for i in range(len(info['model_list'])):
        big_score = None
        big_model = None
        for model_name in filter_dict:
            if big_score is None:
                big_score = name_2_diff[model_name]
                big_model = model_name
            else:
                if name_2_diff[model_name] < big_score:
                    big_score = name_2_diff[model_name]
                    big_model = model_name
        model_sort_list.append(big_model)
        filter_dict.pop(big_model)

    # 一層層搜尋最有可能的pre-trained model
    ancenter = []
    for i, layer_name in enumerate(summary['layer_list']):
        layer_type = layer_name.split('-')[0]
        source = 'random'
        for net_name in model_sort_list:
            layer_name, layer_summary = __getLayerInfo(info, net_name, i)
            if layer_name.split('-')[0] == layer_type:
                if 'weight_param' in layer_summary:
                    if layer_summary['weight_param'] == summary['net'][layer_name]['weight_param']:
                        source = net_name
                        break
        ancenter.append(source)

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
    param_list = [None] * len(ancenter)
    for net_name in set(ancenter):
        if net_name != 'random':
            def register_scratch_hook(module):
                global layer_count
                layer_count = 0
                def scratch_hook(module, input, output):
                    global layer_count
                    if layer_count < len(ancenter):
                        if ancenter[layer_count] == net_name:
                            param_list[layer_count] = module.state_dict()
                    layer_count += 1
                if (not isinstance(module, nn.Sequential) and not isinstance(module, nn.ModuleList)):
                    hook_list.append(module.register_forward_hook(scratch_hook))
            pretrained_model = load(net_name)
            hook_list = []
            pretrained_model.apply(register_scratch_hook)

            # Form the input list
            input_size = info['model_list'][net_name]['input_size']
            if torch.cuda.is_available():
                dtype = torch.cuda.FloatTensor
                pretrained_model = pretrained_model.cuda()
            else:
                dtype = torch.FloatTensor
                pretrained_model = pretrained_model.cpu()
            x = Variable(torch.rand(1,*input_size)).type(dtype)
            pretrained_model(x)

            # remove these hooks
            for h in hook_list:
                h.remove()
    
    def register_assign_hook(module):
        global layer_count
        layer_count = 0
        def assign_hook(module, input, output):
            global layer_count
            if layer_count < len(ancenter):
                if param_list[layer_count] is not None:
                    module.load_state_dict(param_list[layer_count])
            layer_count += 1
        if (not isinstance(module, nn.Sequential) and not isinstance(module, nn.ModuleList)):
            hook_list.append(module.register_forward_hook(assign_hook))
    
    hook_list = []
    net.apply(register_assign_hook)

    if torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
        net = net.cuda()
    else:
        dtype = torch.FloatTensor
        net = net.cpu()
    x = Variable(torch.rand(1,*net_input_size)).type(dtype)
    net(x)

    # remove these hooks
    for h in hook_list:
        h.remove()

def diff(source, target):
    """
        比較兩個summary差異多大
        順序不一樣或大小不一樣都扣分，滿分為0分
    """
    score = 0
    source_order = source['layer_list']
    target_order = target['layer_list']
    for i in range(len(source_order)):
        source_layer_name = source_order[i]
        target_layer_name = target_order[i]
        if source_layer_name != target_layer_name:
            score -= 1
        else:
            if 'weight_param' in source['net'][source_layer_name] and \
                source['net'][source_layer_name]['weight_param'] != target['net'][target_layer_name]['weight_param']:
                score -= 1
    return score 

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