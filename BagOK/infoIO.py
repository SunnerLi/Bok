import json
import os

"""
    這份script說明的info的操作，以及info和summary之間的關係

    ======================================================================================
    1.  從檔案中讀入的物件，為一個dict，名為info，
        他的結構如下：
        info = {
            'name_list' : 一個list物件，每個element是模型的名子，譬如['vgg11', 'vgg16', ...]
            'model_list': 一個dict物件
        }


    2.  info['model_list']是一個dict物件，他的結構長的像下面這樣：
        info = {
            'name_list' : <略>
            'model_list': {
                <key>: <summary>
            }
        }
        每一個key是model的名子，這個名子可以在info['name_list']中取得。
        舉個例子，呼叫info['model_list']['vgg16']就可以獲取VGG-16的summary資訊


    3.  summary又是一個很大的dict物件，他是由pytorch-summary套件延伸定義的，他長這樣：
        summary = {
            'layer_list'        : 一個list物件，每個element是這個模型每層的名子，
                                  譬如['Conv2d-1', 'ReLU-2', 'BatchNorm-3', ...]
            'input_size'        : 一個list物件，存放了輸入tensor的大小，大小格式為CHW，譬如[3, 224, 224]
            'net'               : 一個很大的dict物件...
            'weight_layer_list' : 一個list物件，此物件不會被儲存，但會在init()函式中被計算創建，
                                  定義和'layer_list'相同，只是僅紀錄有權重的層的名子，
                                  譬如['Conv2d-1', 'BatchNorm-3', ...]
            'weight_index_list' : 一個list物件，此物件不會被儲存，但會在init()函式中被計算創建，
                                  定義和'layer_list'相同，只是僅紀錄有權重的層的index順序，
                                  譬如[1, 3, ...]
        }

    
    4. 在上述的結構中，net仍未被說明，他的結構跟pytorch-summary的定義幾乎一致，只是加了些內容：
       net = {
           <key>: <detail_dict>
       }
       每一個key是層的名子，你只能透過info['model_list'][<model_name>]['net'].keys()取得，
       而detail_dict則是一個dict物件，他的定義如下：
       net = {
           <key>: {
               'nb_params'   : 一個int，這層的權重總數
               'trainable'   : 一個bool，這層的參數是否可被訓練
               'input_shape' : 一個list，紀錄了輸入tensor的大小
               'output_shape': 一個list，紀錄了輸出tensor的大小
               'weight_param': 一個list，為本套件新增的資訊，紀錄了weight parameter的大小
               'bias_param'  : 一個list，為本套件新增的資訊，紀錄了bias parameter的大小
           }
       }
"""


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
    if not os.path.exists(file_name):
        info = dict()
    else:
        info = json.load(open(file_name, 'r'))
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