from torch.autograd import Variable
from collections import OrderedDict
import torch.nn as nn
import torch

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
            module_idx = len(__sum)

            m_key = '%s-%i' % (class_name, module_idx+1)
            __sum[m_key] = OrderedDict()
            __sum[m_key]['input_shape'] = list(input[0].size())
            __sum[m_key]['input_shape'][0] = -1
            if isinstance(output, (list,tuple)):
                __sum[m_key]['output_shape'] = [[-1] + list(o.size())[1:] for o in output]
            else:
                __sum[m_key]['output_shape'] = list(output.size())
                __sum[m_key]['output_shape'][0] = -1

            params = 0
            if hasattr(module, 'weight'):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                __sum[m_key]['trainable'] = module.weight.requires_grad
                __sum[m_key]['weight_param'] = list(module.weight.size())
            if hasattr(module, 'bias') and hasattr(module.bias, 'size'):
                params +=  torch.prod(torch.LongTensor(list(module.bias.size())))
                __sum[m_key]['bias_param'] = list(module.bias.size())
            __sum[m_key]['nb_params'] = params
                
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
    __sum = OrderedDict()
    hooks = []
    # register hook
    model.apply(register_hook)
    # make a forward pass
    model(x)
    # remove these hooks
    for h in hooks:
        h.remove()
    summary = {}
    summary['input_size'] = input_size
    summary['layer_list'] = list(__sum.keys())
    summary['net'] = __sum
    # show infomation
    if verbose:
        __showSum(input_size, __sum)
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
            line_new += '{:>25} '.format(str(summary[layer]['weight_param']))
        if 'bias_param' in summary[layer]:
            line_new += '{:>25} '.format(str(summary[layer]['bias_param']))

        print(line_new)
    print('=======================================================================================================================')
    print('Total params: ' + str(total_params))
    print('Trainable params: ' + str(trainable_params))
    print('Non-trainable params: ' + str(total_params - trainable_params))
    print('-----------------------------------------------------------------------------------------------------------------------')