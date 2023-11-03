import torch
import torch.nn as nn
import torch.nn.functional as F
cls_loss = torch.nn.NLLLoss()
import math

class CosineSimilarity(nn.Module):

    def forward(self, tensor_1, tensor_2):
        norm_tensor_1=tensor_1.norm(dim=-1, keepdim=True)
        norm_tensor_2=tensor_2.norm(dim=-1, keepdim=True)
        norm_tensor_1=norm_tensor_1.numpy()
        norm_tensor_2=norm_tensor_2.numpy()

        for  i,vec2 in enumerate(norm_tensor_1[0]) :
            for j,scalar in enumerate(vec2):
                if scalar==0:
                    norm_tensor_1[0][i][j]=1
        for i, vec2 in enumerate(norm_tensor_2[0]):
            for j, scalar in enumerate(vec2):
                if scalar == 0:
                    norm_tensor_2[0][i][j]=1
        norm_tensor_1=torch.tensor(norm_tensor_1)
        norm_tensor_2 = torch.tensor(norm_tensor_2)
        normalized_tensor_1 = tensor_1 / norm_tensor_1
        normalized_tensor_2 = tensor_2 / norm_tensor_2
        return (normalized_tensor_1*normalized_tensor_2).sum(dim=-1)

class DotProductSimilarity(nn.Module):
    def __init__(self,scale_output=False):
        super(DotProductSimilarity,self).__init__()
        self.scale_output=scale_output
    def forward(self,tensor_1,tensor_2):
        result=(tensor_1*tensor_2).sum(dim=-1)
        if(self.scale_output):
            result/=math.sqrt(tensor_1.size(-1))
        return  result

class BiLinearSimilarity(nn.Module):
    def __init__(self,tensor_1_dim = 4,tensor_2_dim = 4,activation=None):
        super(BiLinearSimilarity,self).__init__()
        self.weight_matrix=nn.Parameter(torch.Tensor(tensor_1_dim,tensor_2_dim))
        self.bias=nn.Parameter(torch.Tensor(1))
        self.activation=activation
        self.reset_parameters()
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight_matrix)
        self.bias.data.fill_(0)
    def forward(self, tensor_1,tensor_2):
        intermediate=torch.matmul(tensor_1,self.weight_matrix)
        result=(intermediate*tensor_2).sum(dim=-1)+self.bias
        if self.activation is not None:
            result=self.activation(result)
        return result


class PearsonCorrelation(nn.Module):
    def __init__(self):
        self.inti = 0
    def forward(self,tensor_1,tensor_2):
        x = tensor_1
        y = tensor_2

        vx = x - torch.mean(x)
        vy = y - torch.mean(y)

        cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
        return cost



# def distance():

class distance_o2(nn.Module):
    def __init__(self,fun_name):
        super(distance_o2,self).__init__()
        self.fun_name = fun_name
        if self.fun_name == 'PearsonCorrelation':
            self.function = PearsonCorrelation()
        elif self.fun_name == 'DotProductSimilarity':
            self.function = DotProductSimilarity()
        # elif self.fun_name == 'CosineSimilarity':
        elif self.fun_name == 'BiLinearSimilarity':
            self.function = BiLinearSimilarity()
        print(self.fun_name)

    def forword(self,output1,output2):

        if self.fun_name == "O_2":
            distances = (output2 - output1).pow(2).sum(1)
            return distances

        elif self.fun_name == 'CosineSimilarity':
            distances = torch.nn.CosineSimilarity(output1,output2)
            return distances
        else :
            return self.function(output1,output2)


class FocalLoss(nn.Module):

    def __init__(self, gamma=0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()
class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """

    def __init__(self, opt):
        super(ContrastiveLoss, self).__init__()
        self.margin = 1.
        self.eps = 1e-9
        self.distance = distance_o2(opt.distance)

    def forward(self, outputs1, outputs2, target, size_average=True):

        if type(outputs1) == tuple and type(outputs2) == tuple :
            output1 = outputs1[0]
            output2 = outputs1[1]

            distances = (output2 - output1).pow(2).sum(1)  # squared distances
            losses1 = 0.5 * (target.float() * distances +
                            (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))

            output3= outputs1[0]
            output4 = outputs1[1]
            distances = (output4 - output3).pow(2).sum(1)  # squared distances
            losses2 = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
            losses = 0.5*losses2 + 0.5*losses1
            # print('debug for loss function')

        else :
            output1 = outputs1
            output2 = outputs2
            distances = (output2 - output1).pow(2).sum(1)  # squared distances
            losses = 0.5 * (target.float() * distances +
                            (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))





        return losses.mean() if size_average else losses.sum()
class ContrastiveLoss_class_loss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """

    def __init__(self, margin):
        super(ContrastiveLoss_class_loss, self).__init__()
        self.margin = margin
        self.eps = 1e-9
        self.loss_func = nn.CrossEntropyLoss()
    def forward(self, output1, output2, target, size_average=True):
        distances = (output2 - output1).pow(2).sum(1)  # squared distances
        losses1 = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))


        pred = F.cosine_similarity(output1,output2)
        loss2 = self.loss_func(pred,target.float())
        losses = losses1 +0.05 * loss2
        return losses.mean() if size_average else losses.sum()

class myLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """

    def __init__(self, margin):
        super(myLoss, self).__init__()
        # self.margin = margin
        self.eps = 1e-9
        self.loss_func = nn.CrossEntropyLoss()
    def forward(self, output1, output2, target, size_average=True):
        # output1 =  F.normalize(output1, dim=1)
        # output2 = F.normalize(output2, dim=1)
        pred = F.cosine_similarity(output1,output2)
        loss = self.loss_func(pred,target.float())




        # distances = (output2 - output1).pow(2).sum(1)  # squared distances
        # losses = 0.5 * (target.float() * distances +
        #                 (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        return loss



class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()


class OnlineContrastiveLoss(nn.Module):
    """
    Online Contrastive loss
    Takes a batch of embeddings and corresponding labels.
    Pairs are generated using pair_selector object that take embeddings and targets and return indices of positive
    and negative pairs
    """

    def __init__(self, margin, pair_selector):
        super(OnlineContrastiveLoss, self).__init__()
        self.margin = margin
        self.pair_selector = pair_selector

    def forward(self, embeddings, target):
        positive_pairs, negative_pairs = self.pair_selector.get_pairs(embeddings, target)
        if embeddings.is_cuda:
            positive_pairs = positive_pairs.cuda()
            negative_pairs = negative_pairs.cuda()
        positive_loss = (embeddings[positive_pairs[:, 0]] - embeddings[positive_pairs[:, 1]]).pow(2).sum(1)
        negative_loss = F.relu(
            self.margin - (embeddings[negative_pairs[:, 0]] - embeddings[negative_pairs[:, 1]]).pow(2).sum(
                1).sqrt()).pow(2)
        loss = torch.cat([positive_loss, negative_loss], dim=0)
        return loss.mean()


class OnlineTripletLoss(nn.Module):
    """
    Online Triplets loss
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using triplet_selector object that take embeddings and targets and return indices of
    triplets
    """

    def __init__(self, margin, triplet_selector):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector

    def forward(self, embeddings, target):

        triplets = self.triplet_selector.get_triplets(embeddings, target)

        if embeddings.is_cuda:
            triplets = triplets.cuda()

        ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1)  # .pow(.5)
        an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(ap_distances - an_distances + self.margin)

        return losses.mean(), len(triplets)
