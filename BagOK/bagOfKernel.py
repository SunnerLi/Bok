from .utils import INFO, load_state_dict, BokException
from .infoIO import __readInfoJSON, __writeInfoJSON
from .networkSummary import __summary
from torch.autograd import Variable
from collections import OrderedDict
from .pretrain import load
import torch.nn as nn
import torch
import os

"""
    這份script定義了所有BoK的操作，
    使用者僅需要呼叫init即可，
    其他函式是提供開發者更新預訓練網路資訊時才需要呼叫
"""

# Constant difinition
__INFO_NAME = './bag.json'
layer_count = 0
VERBOSE = False

def init(net, net_input_size, net_type = None, verbose = False, file_name = './bag.json'):
    """
        對於給定的網路進行初始化
        初始化的方式有兩種，一種是hard initialization，另一種是soft initialization
        
        1. Hard initialization非常嚴謹的比較兩個網路的layer順序，
        只容許channel不同，大小和順序都要相同，否則就會對於相似度進行扣分，
        在初始化時，只會針對完全相同的layer index的module做初始化

        2. Soft initialization則剔除所有沒有參數的layer，
        僅針對有參數的layer進行順序比較，
        這種作法可以增加來源網路不被設成random的機率。
        
        Arg:    net             - 想要初始化的網路，為一個nn.Module物件
                net_input_size  - list物件，紀錄了網路輸入tensor的大小
                net_type        - 為想要初始化的網路依據名子
                verbose         - bool，想不想要顯示log
                file_name       - info檔案的路徑，建議直接使用預設
    """
    global __INFO_NAME

    # Obtain the summary object
    net = net.cuda() if torch.cuda.is_available() else net
    summary = __summary(net, net_input_size, verbose = verbose)

    # Check if the info file is exist
    __INFO_NAME = file_name
    if not os.path.exists(__INFO_NAME):
        raise Exception('You should download the info JSON file!')

    # -----------------------------------------
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
    info = __readInfoJSON(__INFO_NAME)
    if net_type is not None:
        target_summary = info['model_list'][net_type]
        if __summary_diff(summary, target_summary, summary['layer_list'], target_summary['layer_list']) == 0:
            net = load_state_dict(load(net_type), net)
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
                    result_weight_index_list.remove(int(layer_name.split('-')[-1]) - 1)
            info['model_list'][model_name]['weight_layer_list'] = result_weight_layer_list
            info['model_list'][model_name]['weight_index_list'] = result_weight_index_list
        result_weight_layer_list = list(summary['layer_list'])
        result_weight_index_list = list(range(len(result_weight_layer_list)))
        for layer_name in summary['net']:
            if 'weight_param' not in summary['net'][layer_name]:
                result_weight_layer_list.remove(layer_name)
                result_weight_index_list.remove(int(layer_name.split('-')[-1]) - 1)
        summary['weight_layer_list'] = result_weight_layer_list
        summary['weight_index_list'] = result_weight_index_list

        # -------------------------------------------
        # (2) 針對每一個候選網路計算相似度 
        #     包括soft跟hard的情況 
        # -------------------------------------------
        name_2_hard_diff = OrderedDict()
        name_2_soft_diff = OrderedDict()
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
        #     包括soft跟hard的情況 
        # -------------------------------------------
        model_hard_sort_list = __getSortList(info, name_2_hard_diff)
        model_soft_sort_list = __getSortList(info, name_2_soft_diff)

        # -------------------------------------------
        # (4) 一層層搜尋最有可能的pre-trained model
        #     融合soft和hard的概念成一個單一結果
        # -------------------------------------------
        net_ancestor_list, layer_ancestor_list = __getAncestor(info, summary, 
            layer_list_key  = 'layer_list', 
            model_sort_list = model_hard_sort_list, 
            net_ancestor    = None, 
            layer_ancestor  = None
        )
        net_ancestor_list, layer_ancestor_list = __getAncestor(info, summary, 
            layer_list_key  = 'weight_layer_list', 
            model_sort_list = model_soft_sort_list, 
            net_ancestor    = net_ancestor_list, 
            layer_ancestor  = layer_ancestor_list
        )
        
        # -----------------------------------------------------------------------
        # (Opt) 印出每一層的資訊，以及預訓練的模型來源
        #       最後確保net_ancestor_list和layer_ancestor_list長度相等
        # -----------------------------------------------------------------------
        if verbose:
            INFO('-----------------------------------------------------------------------------------------------------------------------')
            category_line = '{:>25} {:>25} {:>25}'.format('Layer Type', 'From(Network)', 'From(Layer)')
            INFO(category_line)
            INFO('=======================================================================================================================')
            for layer, net_anc, layer_anc in zip(summary['layer_list'], net_ancestor_list, layer_ancestor_list):
                layer_info = '{:>25} {:>25} {:>25}'.format(layer, net_anc, layer_anc)
                INFO(layer_info)
            INFO('-----------------------------------------------------------------------------------------------------------------------')
        if len(net_ancestor_list) != len(layer_ancestor_list):
            raise BokException('The length of both ancestor list is not the same!')

        # -------------------------------------------
        # (5) 根據ancenter list擷取每一層的weight        
        # -------------------------------------------
        param_list = [None] * len(net_ancestor_list)
        for net_name in set(net_ancestor_list):
            if net_name != 'random':
                if verbose:                
                    INFO('Extract the parameters in %s' % (net_name))                
                def register_scratch_hook(module):
                    global layer_count
                    layer_count = 0
                    def scratch_hook(module, input, output):
                        global layer_count
                        layer_name = '%s-%i' % (str(module.__class__).split('.')[-1].split("'")[0], layer_count + 1)
                        for i in range(len(net_ancestor_list)):
                            if net_ancestor_list[i] == net_name and layer_ancestor_list[i] == layer_name:
                                param_list[i] = module
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
                if layer_count < len(net_ancestor_list):
                    if param_list[layer_count] is not None:
                        module = load_state_dict(source = param_list[layer_count], target = module)
                layer_count += 1
            if (not isinstance(module, nn.Sequential) and not isinstance(module, nn.ModuleList)):
                hook_list.append(module.register_forward_hook(assign_hook))
        if verbose:                
            INFO('Assign the parameter...')                                    
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
        if verbose:        
            INFO('Finish Bok initialization!')

def __getSortList(info, name_2_diff):
    """
        根據name_2_diff的資訊，
        回傳一個相似程度由高分至低分的list
        舉個例子，如果name_2_diff長這樣：{'vgg11': -5 'vgg19': -2}
        則回傳的結果長這樣:['vgg19', 'vgg11']

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

def __getAncestor(info, summary, layer_list_key, model_sort_list, net_ancestor = None, layer_ancestor = None):
    """
        根據模型的層層資訊和相似度資訊 ，組合出 '祖先串列' 和 '層名子來源串列'，
        所謂的祖先串列，紀錄的每一層的pretrained-model是誰，
        層名子來源串列，則紀錄對應到pretrained-model是從哪一層來提取

        Arg:    info            - info物件
                summary         - 要被初始化的模型，它的summary資訊
                layer_list_key  - layer_list的key值，可指定為 'layer_list' 或 'weight_layer_list'
                model_sort_list - list物件，根據相似度紀錄了模型的名稱，例如['vgg16', 'vgg19', ...]
                net_ancestor    - list物件，紀錄了每一層的祖先來自哪個pretrained-model
                layer_ancestor  - list物件，紀錄了每一層的祖先來自pretrained-model的哪一層
        Ret:    修正過的net_ancestor和layer_ancestor
    """
    if net_ancestor is None or layer_ancestor is None:
        net_ancestor_result = []
        layer_ancestor_result = []
    else:
        net_ancestor_result = list(net_ancestor)
        layer_ancestor_result = list(layer_ancestor)
    for i, layer_name in enumerate(summary[layer_list_key]):
        layer_type = layer_name.split('-')[0]
        net_source = 'random'
        layer_source = 'random'
        if net_ancestor is None or net_ancestor_result[i] == 'random':
            for net_name in model_sort_list:
                if i < len(info['model_list'][net_name][layer_list_key]):
                    layer_name, layer_summary = __getLayerInfo(info, net_name, i, list_key = layer_list_key)
                    if layer_name.split('-')[0] == layer_type:
                        if 'weight_param' in layer_summary:
                            if __size_diff(layer_summary['weight_param'], summary['net'][layer_name]['weight_param']):
                                net_source = net_name
                                layer_source = layer_name
                                break
            if net_ancestor is None:
                net_ancestor_result.append(net_source)
                layer_ancestor_result.append(layer_source)
            else:
                idx = int(layer_name.split('-')[-1]) - 1
                net_ancestor_result[idx] = net_source
                layer_ancestor_result[idx] = layer_source
    return net_ancestor_result, layer_ancestor_result

def __getLayerInfo(info, net_name, index, list_key = 'layer_list'):
    """
        根據給定的網路名稱和第幾層的index，回傳那一層的相關資訊

        Arg:    info        - info物件
                net_name    - 想要窺看的網路名稱
                index       - 想要得知該網路的第幾層
                list_key    - 推算的依據，可指定為 'layer_list' 或 'weight_layer_list'
        Ret:    那一層的名子，跟它的summary資訊
    """
    layer_name = info['model_list'][net_name][list_key][index]
    return layer_name, info['model_list'][net_name]['net'][layer_name]    

def __size_diff(size1, size2):
    """
        比較兩個size是否大小一致，
        如果HW一致但channel不一致，仍判定為一樣

        Arg:    size1   - 一個list，紀錄的大小資訊，格式需為[CHW]
                size2   - 一個list，紀錄的大小資訊，格式需為[CHW]
        Ret:    大小是否一樣
    """
    if len(size1) != len(size2):
        raise BokException('The size is not the same')
    if len(size1) == 4 or len(size1) == 3:
        if size1[-1] == size2[-1] and size1[-2] == size2[-2]:
            return True
        return False
    else:
        raise BokException('The rank of size is only %d' % (len(size1)))

def __summary_diff(source, target, source_layer_list, target_layer_list):
    """
        比較兩個summary差異多大
        順序不一樣或大小不一樣都扣分，滿分為0分
        為了soft or hard初始化的彈性，layer的順序需要另外給予
        source端為想要初始化的模型
        target端為已經訓練過的完美模型

        Arg:    source              - 想要初始化的模型，它的summary資訊
                target              - 已經訓練過的完美模型，它的summary資訊
                source_layer_list   - 想要初始化的模型，它的每層資訊，可傳入info['model_list'][<model_name>]['weight_layer_list']
                target_layer_list   - 已經訓練過的完美模型，它的每層資訊，可傳入info['model_list'][<model_name>]['weight_layer_list']
        Ret:    兩個模型的相似度得分
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