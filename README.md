run  main.py demo :

```
python main.py 
```



参数说明：
--backbone：0 是‘embeddingNet’  ，1 是 'resnet18'，2 是‘cnn9’ 设置backbone,默认为 0 

```
python main.py --backbone 1 
```
--ues4imgextend: 1 代表 使用4img 的方法，0代表使用2img 方法  默认为0

```
python main.py --backbone 1 --ues4imgextend 1
```
--backbone_out : int 类型，控制backbone网络输出的特征向量的维度。默认为4

```
python main.py --backbone 1 --ues4imgextend 1 --backbone_out 4
```
--use_marginal 是否使用marginal loss。 这个只能在输出两个img的时候使用。默认为false

--debug_mode 这个是调试代码时候用的。debug 模式可以加快数据集的读入速度。

--activation_funciton 选择激活函数，PReLU,ReLU,LeakyReLU,RReLU,PReLU 默认 PReLU

--add_batchnorm  选择是否添加batchnorm 默认为 1 不添加选0

--optimizer 选择优化器 ，Adam，SGD，AdamW，sgd-nestov 

--function 可以选择 不同的计算距离的函数。DotProductSimilarity BiLinearSimilarity PearsonCorrelation O_2 CosineSimilarity



其他可调参数

--lr  ： learning rate 默认为1e-4

--batch-size  ： batch默认为128

--epochs ： 默认为4



目前效果最好的方法：
AUC ： 0.825         EER： 0.246

```
python main.py --backbone 2 --use_marginal True --activation_funciton RReLU 
```



10.26

1. 使用带预训练的resnet18  来训练73mnist              √
2. 设置fine-turning resnet 18 即冻结大部分网络参数。  调试得到一个效果比较好的模型作为teacher-model。

尝试了两个方法：

resnet18 主干网络固定，训练分类头。

resnet18 主干网络lr 1e-4  分类头lr 1e-3 

**效果都不太好。**

3. 选embeddingNet or cnn9 作为student model 进行知识蒸馏训练。





训练 Iris :

```
 python main.py --train_data_root CASIA-Iris-Lamp/train.txt --test_data_root CASIA-Iris-Lamp/test.txt --batch-size 16
```



训练100类的iris

```
python .\main.py --train_data_root CASIA-Iris-Lamp/train.txt --test_data_root CASIA-Iris-Lamp/test.txt --epochs 100 --backbone 1 --batch-size 16 --n_classes 100
```



训练triplet ：

--loss_fn：可选 OnlineTripletLoss，TripletLoss

--select： 可选RandomNegativeTripletSelector HardestNegativeTripletSelector SemihardNegativeTripletSelector AllTripletSelector。这些都是在loss_fn 为onlinetirpletloss的时候才有效

--backbone_out：控制输出的feature的维度。

--backbone：0 是‘embeddingNet’  ，1 是 'resnet18'，2 是‘cnn9’ 设置backbone,默认为 0 

for example:

```
python triplet.py --train_data_root CASIA-Iris-Lamp/train.txt --test_data_root CASIA-Iris-Lamp/test.txt --epochs 100 --backbone 1 --n_classes 100  --loss_fn OnlineTripletLoss --select RandomNegativeTripletSelector 
```



