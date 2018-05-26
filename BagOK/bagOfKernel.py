from summary import summary

"""
    這份script定義了所有BoK的操作，
    使用者僅需要呼叫init即可，
    其他函式是提供開發者更新預訓練網路資訊時才需要呼叫
"""

def init(net, summary, type = None):
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
   
    pass

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
    pass

def __writeInfoJSON(summary, net_name, file_name = './bag.json'):
    """
        << 使用者切勿直接使用此函式 >>

        將網路的summary資訊寫入JSON檔中

        Arg:    summary     - 網路的summary資訊
                net_name    - 要指定寫入info檔中網路的名子
    """
    pass

def __summary(net, size_list):
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

        Arg:    net         - 網路，為一個nn.Module物件
                size_list   - 一個含有輸入大小的list，為list of list，每一個element代表一個輸入的大小（不含batch)
    """
    pass