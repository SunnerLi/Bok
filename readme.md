## Bok: Bag-of-kernel Initialization Approach to Accelerate Natural Vision Task Training

[![Packagist](https://img.shields.io/badge/Pytorch-0.4.0-red.svg)]()
[![Packagist](https://img.shields.io/badge/Python-3.5.2-blue.svg)]()

![](https://i.imgur.com/J8runwS.png)

Motivation
---
In the usual training process, the random initialization is widely to be used. In some cases, we might use pre-trained model and fine-tune toward the parameters. However, the structure of model should be the absolute same as original, thus load whole pre-trained model is lack of flexibility.    

Abstraction
---
We purposed a method to deploy the look-up table idea into the initialization step of neural network. Before the model is initialized, the function will try to compare the network structure with several pre-trained model, and determine the parameters from the specific layer and specific pre-trained model. Moreover, the program also consider the case with different channel numbers.     
<br/>    
This program has **soft** and **hard** comparison approach. In the hard comparison approach, the layer would be refer while the layer order of two structure should be the same. On the contrary, the layer which doesn't have parameters will be ignore to be considered in soft comparison approach, and this method can increase the probability that the layer can be initialized by any pre-trained model.    
<br/>    
In this repository, 20 well-known models is arranged by ourself, and we record them as an information file bag.json. Before you want to training for visual task, you just need to initialize the parameters from the table.    

Usage
---
First, you should download the whole containing in `BagOK` folder, and it look like this:
```
$ ls
BagOK   my_program.py
```

Next, the step is very simple! Just import the library and use `init` method:
```python
from MyModel import myVGG
import BagOK.bagOfKernel as bok

net = myVGG()                                                   # Initialize your network
bok.init(net, [3, 224, 224], net_type = None, verbose = True)   # Feed and done!
```

Or you can just simply assign the kind of model:
```python
net = myVGG()
init(net, [3, 224, 224], net_type = 'vgg-16')   # Use the parameters of VGG to initialize
```

Models
---
The pretrained-models we consider including these following:
01. VGG11
02. VGG13
03. VGG16
04. VGG19
05. VGG11_BN
06. VGG13_BN
07. VGG16_BN
08. VGG19_BN
09. ResNet101
10. ResNet152
11. ResNet18
12. ResNet34
13. ResNet50
14. AlexNet
15. SqueezeNet1_0
16. SqueezeNet1_1
17. DenseNet121
18. DenseNet161
19. DenseNet169
20. DenseNet201 