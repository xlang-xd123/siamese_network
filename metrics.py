import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (roc_auc_score)
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import torch.nn as nn
from torch.nn import Parameter
import math
class Metric:
    def __init__(self):
        pass

    def __call__(self, outputs, target, loss):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def value(self):
        raise NotImplementedError

    def name(self):
        raise NotImplementedError


class AccumulatedAccuracyMetric(Metric):
    """
    Works with classification model
    """

    def __init__(self):
        self.correct = 0
        self.total = 0

    def __call__(self, outputs, target, loss):
        pred = outputs[0].data.max(1, keepdim=True)[1]
        self.correct += pred.eq(target[0].data.view_as(pred)).cpu().sum()
        self.total += target[0].size(0)
        return self.value()

    def reset(self):
        self.correct = 0
        self.total = 0

    def value(self):
        return 100 * float(self.correct) / self.total

    def name(self):
        return 'Accuracy'


class AverageNonzeroTripletsMetric(Metric):
    '''
    Counts average number of nonzero triplets found in minibatches
    '''

    def __init__(self):
        self.values = []

    def __call__(self, outputs, target, loss):
        self.values.append(loss[1])
        return self.value()

    def reset(self):
        self.values = []

    def value(self):
        return np.mean(self.values)

    def name(self):
        return 'Average nonzero triplets'



class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin

            cos(theta + m)
        """
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features)).cuda()
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.criterion = torch.nn.CrossEntropyLoss()
    def forward(self, inputs, labels):
        outputs = []
        # print('labels = ' )
        label = labels[0]
        # print(label)
        losses = 0
        # print(label,labels)
        # print(len(inputs))
        for input in inputs:
            # print(input)
        # --------------------------- cos(theta) & phi(theta) ---------------------------
            cosine = F.linear(F.normalize(input), F.normalize(self.weight))
            # input 和 weight 计算余弦相似度

            sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
            # 转为正弦

            phi = cosine * self.cos_m - sine * self.sin_m
            # phi = cos （角度 + m）

            if self.easy_margin:
                phi = torch.where(cosine > 0, phi, cosine)
            else:
                phi = torch.where(cosine > self.th, phi, cosine - self.mm)



            # --------------------------- convert label to one-hot ---------------------------
            # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
            one_hot = torch.zeros(cosine.size(), device='cuda')
            one_hot.scatter_(1, label.view(-1, 1).long(), 1)


            # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------


            output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
            output *= self.s
            # output = (output,)
            outputs.append(output)
            # print(output)
        # print(output)
            losses += self.criterion(output, label)


        # print(inputs)
        # print(outputs)
        # return  outputs
        return losses
# class EERmetric(Metric):
#     """
#     Works with classification model
#     """
#
#     def __init__(self):
#         self.correct = 0
#         self.total = 0
#
#     def __call__(self, outputs, target, loss):
#         pred = outputs[0].data.max(1, keepdim=True)[1]
#         self.correct += pred.eq(target[0].data.view_as(pred)).cpu().sum()
#         self.total += target[0].size(0)
#         return self.value()
#
#     def reset(self):
#         self.correct = 0
#         self.total = 0
#
#     def value(self):
#         return 100 * float(self.correct) / self.total
#
#     def name(self):
#         return 'EER'
class AUC_metric(Metric):
    """
    Works with classification model
    """

    def __init__(self):
        self.target = torch.tensor([])
        self.outputs = torch.tensor([])

    def __call__(self, outputs, target, loss):
        if type(outputs[0]) == tuple:
            pred1 = F.cosine_similarity(outputs[0][0], outputs[0][1])
            pred2 = F.cosine_similarity(outputs[1][0], outputs[1][1])
            pred = 0.5* pred2 + 0.5 * pred1
        else:
            pred = F.cosine_similarity(outputs[0], outputs[1])
        target = target[0]
        self.target = torch.cat([self.target,target.cpu().detach()],axis = 0)
        self.outputs = torch.cat([self.outputs,pred.cpu().detach()],axis = 0)
        # fpr, tpr, thresholds = roc_curve(y_test, scores)
        # pred = outputs[0].data.max(1, keepdim=True)[1]
        # self.correct += pred.eq(target[0].data.view_as(pred)).cpu().sum()
        # self.total += target[0].size(0)
        return self.value()

    def reset(self):
        self.target = torch.tensor([])
        self.outputs = torch.tensor([])

    def value(self):

        return roc_auc_score(self.target, self.outputs)

    # def value2(self):
    #     fpr, tpr, thresholds = roc_curve(self.outputs, self.target)
    #

    def name(self):
        return 'AUC'

class EER_metric(Metric):
    """
    Works with classification model
    """

    def __init__(self):
        self.target = torch.tensor([])
        self.outputs = torch.tensor([])

    def __call__(self, outputs, target, loss):
        if type(outputs[0]) == tuple:
            pred1 = F.cosine_similarity(outputs[0][0], outputs[0][1])
            pred2 = F.cosine_similarity(outputs[1][0], outputs[1][1])
            pred = 0.5 * pred2 + 0.5 * pred1
        else:
            pred = F.cosine_similarity(outputs[0], outputs[1])
        target = target[0]
        self.target = torch.cat([self.target, target.cpu().detach()], axis=0)
        self.outputs = torch.cat([self.outputs, pred.cpu().detach()], axis=0)
        # fpr, tpr, thresholds = roc_curve(y_test, scores)
        # pred = outputs[0].data.max(1, keepdim=True)[1]
        # self.correct += pred.eq(target[0].data.view_as(pred)).cpu().sum()
        # self.total += target[0].size(0)
        return self.value()

    def reset(self):
        self.target = torch.tensor([])
        self.outputs = torch.tensor([])

    def value(self):
        fpr, tpr, thresholds = roc_curve( self.target, self.outputs)
        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

        return eer

    # def value2(self):
    #     fpr, tpr, thresholds = roc_curve(self.outputs, self.target)
    #

    def name(self):
        return 'EER'

    # def name2(self):
    #     return 'EER'