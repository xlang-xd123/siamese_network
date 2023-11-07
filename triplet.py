import argparse
from datasets import *
from networks import *
from torch.optim import lr_scheduler
import torch.optim as optim
from losses import *
from metrics import *
from trainer import fit
from torchvision.datasets import MNIST
from torchvision import transforms,models
mean, std = 0.1307, 0.3081
from utils import *

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_classes', type=int, default= 100, help='number of classes')
    parser.add_argument('--train_data_root', type=str, default='CASIA-Iris-Lamp/train.txt', help='train data path')
    parser.add_argument('--test_data_root', type=str, default='CASIA-Iris-Lamp/test.txt', help='test data path')
    parser.add_argument('--backbone', type= int , default= 0, help='set 0 is embeddingNet ,set 1 is resnet19,set 2 is cnn ')
    parser.add_argument('--use_cuda', type=bool, default=True, help='choose cuda or gpu')
    parser.add_argument('--loss_fn', type=str, default='TripletLoss', help='choose loss function')
    parser.add_argument('--select', type=str, default='RandomNegativeTripletSelector', help='choose loss function')

    parser.add_argument('--log_interval', type=int, default= 100, help='')
    parser.add_argument('--epochs', type=int, default=5, help='total training epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--backbone_out', type=int, default=4, help='dimension of the feature')
    parser.add_argument('--activation_funciton', type=str, default='PReLU', help='PReLU,ReLU,LeakyReLU,RReLU,PReLU')
    parser.add_argument('--add_batchnorm', type= int, default=1, help='')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam,SGD,sgd-nestov')
    parser.add_argument('--freeze_flag', type=int, default= 0 , help='freeze method ')
    parser.add_argument('--distance', type=str, default='cos', help='distance function ')

    return parser.parse_args()

def load_data(opt,trans = 1):
    mean, std = 0.1307, 0.3081
    if trans == 0:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((mean,), (std,))
        ])
    elif trans == 1:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            # transforms.Grayscale(3),
            transforms.ToTensor(),
            transforms.Normalize((mean,), (std,))
        ])

    train_dataset = GetData(opt.train_data_root,transform = transform, train=True)
    test_dataset = GetData(opt.test_data_root,transform = transform, train=False)
    kwargs = {'num_workers': 0, 'pin_memory': True} if opt.use_cuda else {}
    if opt.loss_fn == 'TripletLoss':
        triplet_train_dataset = TripletMNIST(train_dataset)
        triplet_test_dataset = TripletMNIST(test_dataset)
        triplet_train_loader = torch.utils.data.DataLoader(triplet_train_dataset, batch_size=opt.batch_size,
                                                           shuffle=True,
                                                           **kwargs)
        triplet_test_loader = torch.utils.data.DataLoader(triplet_test_dataset, batch_size=opt.batch_size, shuffle=True,
                                                          **kwargs)
    elif opt.loss_fn == 'OnlineTripletLoss' :
        # print('train_dataset.train_labels',train_dataset.train_labels)
        train_batch_sampler  = BalancedBatchSampler(train_dataset.train_labels, n_classes=opt.n_classes, n_samples=1)
        test_batch_sampler  = BalancedBatchSampler(test_dataset.test_labels, n_classes=opt.n_classes, n_samples=1)
        triplet_train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_batch_sampler, **kwargs)
        triplet_test_loader = torch.utils.data.DataLoader(test_dataset, batch_sampler=test_batch_sampler, **kwargs)

    # triplet_train_loader = torch.utils.data.DataLoader(triplet_train_dataset, batch_size=opt.batch_size, shuffle=True,
    #                                                    **kwargs)
    # triplet_test_loader = torch.utils.data.DataLoader(triplet_test_dataset, batch_size=opt.batch_size, shuffle=True,
    #                                                   **kwargs)
    return triplet_train_loader,triplet_test_loader


def load_model(opt):
    if opt.backbone == 0:
        print('load embeddingNet')
        backbone = EmbeddingNet(opt)
    elif opt.backbone == 1 :
        print('load resnet18')
        # backbone = resnet18(opt)
        backbone = models.resnet18(pretrained=True)
        backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,bias=False)

        num_ftrs = backbone.fc.in_features
        backbone.fc = nn.Linear(num_ftrs, opt.backbone_out)

    elif opt.backbone == 2 :
        print('load cnn9')
        backbone = Cnn9Net(opt)
    if opt.loss_fn == 'TripletLoss':

        model = TripletNet(backbone)
    elif opt.loss_fn == 'OnlineTripletLoss':
        model = backbone
    if opt.use_cuda:
        model.cuda()
    return model

def load_loss(opt):
    if opt.loss_fn == 'TripletLoss':
        return TripletLoss(1.)
    elif opt.loss_fn == 'OnlineTripletLoss' and opt.select == 'RandomNegativeTripletSelector':
        return OnlineTripletLoss(1.,RandomNegativeTripletSelector(1.))

    elif opt.loss_fn == 'OnlineTripletLoss' and opt.select == 'HardestNegativeTripletSelector':
        return OnlineTripletLoss(1.,HardestNegativeTripletSelector(1.))
    elif opt.loss_fn == 'OnlineTripletLoss' and opt.select == 'AllTripletSelector':
        return OnlineTripletLoss(1.,AllTripletSelector())
    elif opt.loss_fn == 'OnlineTripletLoss' and opt.select == 'SemihardNegativeTripletSelector':
        return OnlineTripletLoss(1.,SemihardNegativeTripletSelector(1.))
def main(opt):
    # 载入 train 和 test data,封装成dataloader
    triplet_train_loader,triplet_test_loader = load_data(opt)
    # 载入模型 (emmbedding net or resnet or cnn 9),损失函数，以及优化器
    model = load_model(opt)

    loss_fn = load_loss(opt)
    lr = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)




    fit(triplet_train_loader, triplet_test_loader, model, loss_fn, optimizer, scheduler, opt.epochs, opt.use_cuda, opt.log_interval)


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)











#
#
# from torchvision.datasets import MNIST
# from torchvision import transforms
#
# mean, std = 0.1307, 0.3081
#
# train_dataset = MNIST('../data/MNIST', train=True, download=True,
#                              transform=transforms.Compose([
#                                  transforms.ToTensor(),
#                                  transforms.Normalize((mean,), (std,))
#                              ]))
# test_dataset = MNIST('../data/MNIST', train=False, download=True,
#                             transform=transforms.Compose([
#                                 transforms.ToTensor(),
#                                 transforms.Normalize((mean,), (std,))
#                             ]))
# n_classes = 10
#
# import torch
# from torch.optim import lr_scheduler
# import torch.optim as optim
# from torch.autograd import Variable
#
# from trainer import fit
# import numpy as np
# cuda = torch.cuda.is_available()
#
# import matplotlib
# import matplotlib.pyplot as plt
#
# mnist_classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
# colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
#               '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
#               '#bcbd22', '#17becf']
#
# def plot_embeddings(embeddings, targets, xlim=None, ylim=None):
#     plt.figure(figsize=(10,10))
#     for i in range(10):
#         inds = np.where(targets==i)[0]
#         plt.scatter(embeddings[inds,0], embeddings[inds,1], alpha=0.5, color=colors[i])
#     if xlim:
#         plt.xlim(xlim[0], xlim[1])
#     if ylim:
#         plt.ylim(ylim[0], ylim[1])
#     plt.legend(mnist_classes)
#
# def extract_embeddings(dataloader, model):
#     with torch.no_grad():
#         model.eval()
#         embeddings = np.zeros((len(dataloader.dataset), 2))
#         labels = np.zeros(len(dataloader.dataset))
#         k = 0
#         for images, target in dataloader:
#             if cuda:
#                 images = images.cuda()
#             embeddings[k:k+len(images)] = model.get_embedding(images).data.cpu().numpy()
#             labels[k:k+len(images)] = target.numpy()
#             k += len(images)
#     return embeddings, labels
#
# from datasets import TripletMNIST
#
# triplet_train_dataset = TripletMNIST(train_dataset) # Returns triplets of images
# triplet_test_dataset = TripletMNIST(test_dataset)
# batch_size = 128
# kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
# triplet_train_loader = torch.utils.data.DataLoader(triplet_train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
# triplet_test_loader = torch.utils.data.DataLoader(triplet_test_dataset, batch_size=batch_size, shuffle=False, **kwargs)
#
# # Set up the network and training parameters
# from networks import EmbeddingNet_o, TripletNet
# from losses import TripletLoss
#
# margin = 1.
# embedding_net = EmbeddingNet_o()
# model = TripletNet(embedding_net)
# if cuda:
#     model.cuda()
# loss_fn = TripletLoss(margin)
# lr = 1e-3
# optimizer = optim.Adam(model.parameters(), lr=lr)
# scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
# n_epochs = 20
# log_interval = 100
#
# if __name__ =='__main__':
#
#     fit(triplet_train_loader, triplet_test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval)
#
