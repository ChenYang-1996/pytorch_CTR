pytorch_CTR, all models implemented by pytorch
===
DIN, DIEN, CAN will be added soon!
| 模型 | 创新点 |
| ------ | ------ |
| FM | 将二阶交叉特征考虑进来，提高模型的表达能力；引入隐向量，缓解了数据稀疏带来的参数难训练问题；模型复杂度为线性 |
| FFM | FFM在FM的基础上进一步改进，在模型中引入类别（field）的概念，将同一个field的特征单独进行one-hot |
| DCN | ![image](https://user-images.githubusercontent.com/53995142/157873085-81734661-4446-4eea-890a-ec727db512fd.png) |
| W&D | ![image](https://user-images.githubusercontent.com/53995142/157873175-99764297-86ac-4e29-9558-4adf3e9bfa31.png) |
| DeepFM |![image](https://user-images.githubusercontent.com/53995142/157873272-8f20c240-5f4a-477c-99e8-e3441030c6d3.png) |
| DCAP(CIKM2021) | ![image](https://user-images.githubusercontent.com/53995142/157872865-2a897619-2893-4a8c-8590-51b35f1d21a8.png) |

1th step: pip install -r requirements.txt
--
2th step: put the dataset to the /torchfm/dataset
--
3th step: Create a new folder to save the model，please input the command: mkdir chkpt
--
4th step: if you want train your model, please run the example/main.py 
--
5th step: If you want see the loss and acc, please input the command: mkdir logs, and then tensorboard --logdir=logs
--
worth that: I only use the ml-1m dataset for training and testing

m1-1m download link: https://pan.baidu.com/s/12_LdFDwAp7qpbN9Pl4odHw?pwd=1234 提取码：1234 

m1-20m download link:：https://pan.baidu.com/s/1FvCB0oZs0kF-vksNzFiczQ?pwd=1234 提取码：1234 

Criteo download link：https://pan.baidu.com/s/1DCzyUTqNUZuplcIrIeC71g?pwd=1234 提取码：1234 

fixed by https://github.com/zachstarkk/DCAP and https://github.com/rixwew/pytorch-fm

