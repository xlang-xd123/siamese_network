# 1引入模块
import argparse
from datasets import GetData
from datasets import *
from networks import *
import torch
from torch.optim import lr_scheduler
import torch.optim as optim
from torch.autograd import Variable
from losses import myLoss,ContrastiveLoss,ContrastiveLoss_class_loss
from metrics import *
from trainer import fit
from torchvision.datasets import MNIST
from torchvision import transforms
mean, std = 0.1307, 0.3081
def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_classes', type=int, default= 10, help='number of classes')
    parser.add_argument('--train_data_root', type=str, default='mnist/train.txt', help='train data path')
    parser.add_argument('--test_data_root', type=str, default='mnist/test.txt', help='test data path')
    # parser.add_argument('--ues_4_img_extend', type=bool, default=False, help='set True to use 4 img')
    parser.add_argument('--ues4imgextend', type=int, default=0, help='set True to use 4 img')
    parser.add_argument('--backbone', type= int , default= 0, help='set 0 is embeddingNet ,set 1 is resnet19,set 2 is cnn ')
    parser.add_argument('--use_cuda', type=bool, default=True, help='choose cuda or gpu')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--loss_fn', type=str, default='ContrastiveLoss', help='choose loss function')
    parser.add_argument('--log_interval', type=int, default= 100, help='')
    parser.add_argument('--epochs', type=int, default=10, help='total training epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--debug_mode', type=bool, default=False, help='help debug')
    parser.add_argument('--backbone_out', type=int, default=4, help='dimension of the feature')
    parser.add_argument('--use_marginal', type=bool, default=False, help='')
    return parser.parse_args()
def load_data(opt):
    if opt.debug_mode:
        # 这一部分是debug的时候用到的。
        train_dataset = MNIST('../data/MNIST', train=True, download=True,
                                     transform=transforms.Compose([
                                         transforms.ToTensor(),
                                         transforms.Normalize((mean,), (std,))
                                     ]))
        test_dataset = MNIST('../data/MNIST', train=False, download=True,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((mean,), (std,))
                                    ]))
        opt.epochs = 1
        print('---- debug mode -----')
    else:
        # 正常情况下走这个分支。
        train_dataset = GetData(opt.train_data_root, train=True)
        test_dataset = GetData(opt.test_data_root, train=False)

    # 将数据集 转化成 2个img 的形式或者  4个img 的形式,然后装入dataloader
    if opt.ues4imgextend:
        print('load SiameseMNIST_4imgs')
        siamese_train_dataset = SiameseMNIST_4imgs(train_dataset)
        siamese_test_dataset = SiameseMNIST_4imgs(test_dataset)
    # elif opt.use_marginal:
    #     siamese_train_dataset = SiameseMNIST2(train_dataset)
    #     siamese_test_dataset = SiameseMNIST2(test_dataset)
    else:
        print('load SiameseMNIST')
        siamese_train_dataset = SiameseMNIST(train_dataset)
        siamese_test_dataset = SiameseMNIST(test_dataset)
    kwargs = {'num_workers': 0, 'pin_memory': True} if opt.use_cuda else {}

    siamese_train_loader = torch.utils.data.DataLoader(siamese_train_dataset, batch_size=opt.batch_size, shuffle=True,
                                                       **kwargs)
    siamese_test_loader = torch.utils.data.DataLoader(siamese_test_dataset, batch_size=opt.batch_size, shuffle=True,
                                                      **kwargs)
    return siamese_train_loader,siamese_test_loader
def load_model(opt):
    if opt.backbone == 0:
        print('load embeddingNet')
        backbone = EmbeddingNet(opt.backbone_out)
    elif opt.backbone == 1 :
        print('load resnet18')
        backbone = resnet18(opt.backbone_out)
    elif opt.backbone == 2 :
        print('load cnn9')
        backbone = Cnn9Net(opt.backbone_out)
        #todo
        pass

    if opt.ues4imgextend:
        model = SiameseNet_4img(backbone)
    else :
        model = SiameseNet(backbone)
    if opt.use_cuda:
        model.cuda()
    return model
def load_loss(opt):
    if opt.loss_fn == 'ContrastiveLoss':
        return ContrastiveLoss(1.)
    else:
        pass
#     如果还有其他的loss直接添加
def load_optimizer(opt,model):
    if opt.use_marginal and not opt.ues4imgextend:#如果使用marginal
        print('use marginal')
        metric_fc = ArcMarginProduct(opt.backbone_out,2)
        if opt.use_cuda:
            metric_fc.cuda()
        optimizer = optim.Adam([{'params': model.parameters()}, {'params': metric_fc.parameters()}], lr=opt.lr)
    else:
        metric_fc = None
        optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
    return optimizer,scheduler,metric_fc
def load_metrics(opt):
    # 如果需要增加其他的metrics ，只需要在metrics.py 增加。然后在这里写入。
    ret = []
    a = AUC_metric()
    b = EER_metric()

    ret.append(a)
    ret.append(b)

    return ret
def main(opt):
    # 载入 train 和 test data,封装成dataloader
    siamese_train_loader,siamese_test_loader = load_data(opt)
    # 载入模型 (emmbedding net or resnet or cnn 9),损失函数，以及优化器
    model = load_model(opt)
    # 载入loss 函数（）
    loss_fn = load_loss(opt)
    # 载入optimizer 和scheduler
    optimizer,scheduler,metric_fc = load_optimizer(opt,model)
    # 载入 eer ，auc 计算函数
    metrics = load_metrics(opt)
    # 开始训练。
    fit(siamese_train_loader, siamese_test_loader, model, loss_fn, optimizer, scheduler, opt.epochs, opt.use_cuda, opt.log_interval,
        metrics=metrics, metric_fc=metric_fc)

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)