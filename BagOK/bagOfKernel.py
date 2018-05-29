from .utils import INFO, load_state_dict
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

__INFO_NAME = './bag.json'
layer_count = 0
VERBOSE = False

def init(net, net_input_size, summary, net_type = None, verbose = False):
    """
        對於給定的網路進行初始化
        初始化的方式有兩種，一種是hard initialization，另一種是soft initialization
        
        1. Hard initialization非常嚴謹的比較兩個網路的layer順序，
        只容許channel不同，大小和順序都要相同，否則就會對於相似度進行扣分，
        在初始化時，只會針對完全相同的layer index的module做初始化

        2. Soft initialization則剔除所有沒有參數的layer，
        僅針對有參數的layer進行順序比較，
        這種作法可以增加來源網路不被設成random的機率。

        * 但這樣的作法可能導致很多網路都有可能會是候選網路
          這種網路的作法並不考慮任何沒有參數的操作，
          因此包括Tensor相加、各種非線性或是奇特操作，
          此比較方式都將忽略這些步驟的差異。
          對於初始化，重要的是將特徵直接賦予，
          所以會對VGG這種沒有太多無參數操作的模型加分
        
        Arg:    net         - 想要初始化的網路，為一個nn.Module物件
                input_size  - 
                summary     - 透過pytorch-summary回傳的統計資訊
                type        - 為想要初始化的網路依據名子
    """
    # Check if the info file is exist
    if not os.path.exists(__INFO_NAME):
        raise Exception('You should download the info JSON file!')

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
            print('Direct assign!')
            # net.load_state_dict(load(net_type).state_dict())
            net = load_state_dict(load(net_type), net)                          # 可能會沒有賦值到？？？
            return
    else:
        """
            尋找最有可能的組合來填充

            1. 先計算所有模型的weight_layer_list
            2. 得到每一個候選網路的weight_layer_list，就是剔除那些沒有權重的layer
            3. 針對每一個候選的網路，都去計算相似度
            4. 針對每一層，根據相似度尋找可能的來源網路
            5. 掛hook去把這些可能網路的參數抓出來
            6. 掛hook去把可能的參數初始到目標網路上
        """
        # -------------------------------------------
        # (1) 計算簡化版的layer_list
        #     包括以名子為基礎的weight_layer_list，
        #     以及以index為基礎的weight_index_list。
        #     對info和本身的summary都要做
        # -------------------------------------------
        for model_name in info['name_list']:
            result_weight_layer_list = list(info['model_list'][model_name]['layer_list'])
            result_weight_index_list = list(range(len(result_weight_layer_list)))
            for layer_name in info['model_list'][model_name]['net']:
                if 'weight_param' not in info['model_list'][model_name]['net'][layer_name]:
                    result_weight_layer_list.remove(layer_name)
                    result_weight_index_list.remove(int(layer_name.split('-')[-1]))
            info['model_list'][model_name]['weight_layer_list'] = result_weight_layer_list
            info['model_list'][model_name]['weight_index_list'] = result_weight_index_list
        result_weight_layer_list = list(summary['layer_list'])
        result_weight_index_list = list(range(len(result_weight_layer_list)))
        for layer_name in summary['net']:
            if 'weight_param' not in summary['net'][layer_name]:
                result_weight_layer_list.remove(layer_name)
                result_weight_index_list.remove(layer_name.split('-')[-1])
        summary['weight_layer_list'] = result_weight_layer_list
        summary['weight_index_list'] = result_weight_index_list

        # -------------------------------------------
        # (2) 針對每一個候選網路計算相似度 (未完成)
        #     包括soft跟hard的情況 
        # -------------------------------------------
        name_2_hard_diff = OrderedDict()
        name_2_soft_diff = OrderedDict()
        small_diff = 100
        for model_name in info['name_list']:            
            candidate_summary = info['model_list'][model_name]
            name_2_hard_diff[model_name] = __summary_diff(
                source = summary, 
                target = candidate_summary, 
                source_layer_list = summary['layer_list'], 
                target_layer_list = candidate_summary['layer_list']
            )
            name_2_soft_diff[model_name] = __summary_diff(
                source = summary, 
                target = candidate_summary, 
                source_layer_list = summary['weight_layer_list'], 
                target_layer_list = candidate_summary['weight_layer_list']
            )
            
        # -------------------------------------------
        # (3) 依分數大到小排出名子序列
        # -------------------------------------------
        model_hard_sort_list = getSortList(info, name_2_hard_diff)
        model_soft_sort_list = getSortList(info, name_2_hard_diff)

        # -------------------------------------------
        # (4) 一層層搜尋最有可能的pre-trained model
        # -------------------------------------------
        ancenter = []
        for i, layer_name in enumerate(summary['layer_list']):
            layer_type = layer_name.split('-')[0]
            source = 'random'
            for net_name in model_sort_list:
                if i < len(info['model_list'][net_name]['layer_list']):
                    layer_name, layer_summary = __getLayerInfo(info, net_name, i)
                    if layer_name.split('-')[0] == layer_type:
                        if 'weight_param' in layer_summary:
                            if layer_summary['weight_param'] == summary['net'][layer_name]['weight_param']:
                                source = net_name
                                break
            ancenter.append(source)
        
        # -------------------------------------------
        # (Opt) 印出每一層的資訊，以及預訓練的模型來源
        # -------------------------------------------
        INFO('------------------------------------------------------------')
        category_line = '{:>25} {:>25}'.format('Layer Type', 'From')
        INFO(category_line)
        INFO('============================================================')
        for layer, anc in zip(summary['layer_list'], ancenter):
            layer_info = '{:>25} {:>25}'.format(layer, anc)
            INFO(layer_info)
        INFO('------------------------------------------------------------')

        # -------------------------------------------
        # (5) 根據ancenter list擷取每一層的weight        
        # -------------------------------------------
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

        # -------------------------------------------
        # (6) 用hook把weight初始化上去
        # -------------------------------------------
        def register_assign_hook(module):
            global layer_count
            layer_count = 0
            def assign_hook(module, input, output):
                global layer_count
                if layer_count < len(ancenter):
                    if param_list[layer_count] is not None:
                        # module.load_state_dict(param_list[layer_count])
                        module = load_state_dict(source = param_list[layer_count], target = module)
                layer_count += 1
            if (not isinstance(module, nn.Sequential) and not isinstance(module, nn.ModuleList)):
                hook_list.append(module.register_forward_hook(assign_hook))
        hook_list = []
        net.apply(register_assign_hook)

        # Form the input list
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

def getSortList(info, name_2_diff):
    """
        根據name_2_diff的資訊，
        回傳一個相似程度由高分至低分的list

        Arg:    info        - info物件
                name_2_diff - OrderDict物件，順序為 <model_name> : <diff_value>
        Ret:    List，裏面每個element是model名稱
    """
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
    model_sort_list = list(reversed(model_sort_list))
    return model_sort_list

def getAncestor(info, summary, layer_list, model_sort_list):
    """
        根據 '層串列' 和 '模型相似度串列' ，組合出 '祖先串列' 和 '曾名子來源串列'
    """
    ancestor = []
    for i, layer_name in enumerate(summary['layer_list']):
        layer_type = layer_name.split('-')[0]
        source = 'random'
        for net_name in model_sort_list:
            if i < len(info['model_list'][net_name]['layer_list']):
                layer_name, layer_summary = __getLayerInfo(info, net_name, i)
                if layer_name.split('-')[0] == layer_type:
                    if 'weight_param' in layer_summary:
                        if layer_summary['weight_param'] == summary['net'][layer_name]['weight_param']:
                            source = net_name
                            break
        ancenter.append(source)

def __getLayerInfo(info, net_name, index):
    layer_name = info['model_list'][net_name]['layer_list'][index]
    return layer_name, info['model_list'][net_name]['net'][layer_name]    

def __size_diff(size1, size2):
    """
        比較兩個size是否大小一致，
        如果hw一致但channel不一致，仍判定為一樣
    """
    if len(size1) != len(size2):
        raise Exception('The size is not the same')             # !!!!!
    if len(size1) == 4 or len(size1) == 3:
        if size1[-1] == size2[-1] and size1[-2] == size2[-2]:
            return True
        return False
    else:
        raise Exception('The rank of size is only %d' % (len(size1)))

def __summary_diff(source, target, source_layer_list, target_layer_list):
    """
        比較兩個summary差異多大
        順序不一樣或大小不一樣都扣分，滿分為0分
        為了soft or hard初始化的彈性，layer的順序需要另外給予

        source端為想要初始化的模型
        target端為已經訓練過的完美模型
    """
    score = 0
    layer_num = min(len(source_layer_list), len(target_layer_list))
    for i in range(layer_num):
        source_layer_name = source_layer_list[i]
        target_layer_name = target_layer_list[i]
        if source_layer_name != target_layer_name:
            score -= 1
        else:
            if 'weight_param' in source['net'][source_layer_name] and not \
                __size_diff(source['net'][source_layer_name]['weight_param'], target['net'][target_layer_name]['weight_param']):
                score -= 1
    return score 

def __readInfoJSON(file_name = './bag.json'):
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
    global __INFO_NAME
    if __INFO_NAME != file_name:
        __INFO_NAME = file_name
    if not os.path.exists(__INFO_NAME):
        info = dict()
    else:
        info = json.load(open(__INFO_NAME, 'r'))
    return info

def __writeInfoJSON(summary, net_name, file_name = './bag.json'):
    """
        << 使用者切勿直接使用此函式 >>

        將網路的summary資訊寫入JSON檔中

        Arg:    summary     - 網路的summary資訊
                net_name    - 要指定寫入info檔中網路的名子
    """
    global __INFO_NAME
    if __INFO_NAME != file_name:
        __INFO_NAME = file_name
    info = __readInfoJSON(__INFO_NAME)
    if 'name_list' not in info:
        info['name_list'] = [net_name]
    else:
        info['name_list'].append(net_name)
    if 'model_list' not in info:
        info['model_list'] = {}
    info['model_list'][net_name] = summary
    json.dump(info, open(__INFO_NAME, 'w'))