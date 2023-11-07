import torch.nn as nn
import torch.nn.functional as F
def smart_activation(opt,feature = None):
    if opt.activation_funciton == 'PReLU':
        a =  nn.PReLU()
    elif opt.activation_funciton == 'ReLU':
        a = nn.ReLU()
    elif opt.activation_funciton == 'LeakyReLU':
        a = nn.LeakyReLU()
    elif opt.activation_funciton == 'RReLU':
        a = nn.RReLU()
    elif opt.activation_funciton == 'PReLU':
        a = nn.PReLU()
    else:
        print('not find the activation_funciton ,please check! (PReLU,ReLU,LeakyReLU,RReLU,PReLU)')
        return None
    if opt.add_batchnorm and feature:
        return nn.Sequential(a,nn.BatchNorm2d(feature))

    return  a ;

class Cnn9Net(nn.Module):
    def __init__(self,opt):
        super(Cnn9Net, self).__init__()

        self.convnet = nn.Sequential(nn.Conv2d(1, 32, 3,1,1), smart_activation(opt,32),
                                     nn.MaxPool2d(2, stride=1,padding=1),
                                     nn.Conv2d(32, 64, 5), smart_activation(opt,64),
                                     nn.MaxPool2d(2, stride=2,padding=1))
        self.convnet2 = nn.Sequential(nn.Conv2d(64, 64, 3,1,1), smart_activation(opt,64),
                                     nn.MaxPool2d(2, stride=1,padding=1),
                                     nn.Conv2d(64, 64, 1,1,1), smart_activation(opt,64),
                                     nn.MaxPool2d(2, stride=1,padding=1))
        self.convnet3= nn.Sequential(nn.Conv2d(64, 64, 3), smart_activation(opt,64),
                                     nn.MaxPool2d(2, stride=1,padding=1),
                                     nn.Conv2d(64, 64, 1), smart_activation(opt,64),
                                     nn.MaxPool2d(2, stride=1,padding=1))
        self.convnet4 = nn.Sequential(nn.Conv2d(64, 64, 3), smart_activation(opt,64),
                                     nn.MaxPool2d(2, stride=1,padding=1),
                                     nn.Conv2d(64, 64, 1), smart_activation(opt,64),
                                      nn.MaxPool2d(2, stride=1, padding=1))
        self.convnet5 = nn.Sequential(nn.Conv2d(64, 64, 5), smart_activation(opt),
                                     nn.AvgPool2d(13, stride=1,padding=0))

        self.fc = nn.Sequential(nn.Linear(64 , 64),
                                # nn.PReLU(),
                                # nn.Linear(256, 256),
                                smart_activation(opt),
                                nn.Linear(64, opt.backbone_out)
                                )
        # print('out = out = ' + str(out))

    def forward(self, x):
        # print(x.shape)
        output = self.convnet(x)
        # print(output.shape)
        output = self.convnet2(output)
        # print(output.shape)
        output = self.convnet3(output)
        # print(output.shape)
        output = self.convnet4(output)
        # print(output.shape)
        output = self.convnet5(output)
        # print(output.shape)

        output = output.view(output.size()[0], -1)
        # print(output.shape)
        output = self.fc(output)
        return output

class EmbeddingNet_o(nn.Module):
    def __init__(self):
        super(EmbeddingNet_o, self).__init__()
        self.convnet = nn.Sequential(nn.Conv2d(1, 32, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.Conv2d(32, 64, 5), nn.PReLU(),
                                     nn.MaxPool2d(2, stride=2))

        self.fc = nn.Sequential(nn.Linear(64 * 4 * 4, 256),
                                nn.PReLU(),
                                nn.Linear(256, 256),
                                nn.PReLU(),
                                nn.Linear(256, 2)
                                )

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)


class EmbeddingNet(nn.Module):
    def __init__(self,opt):
        super(EmbeddingNet, self).__init__()

        self.convnet = nn.Sequential(nn.Conv2d(1, 16, 5), smart_activation(opt,16),
                                     nn.MaxPool2d(2, stride=2),
                                     nn.Conv2d(16, 16, 5), smart_activation(opt,16),
                                     nn.MaxPool2d(2, stride=2))

        self.fc = nn.Sequential(nn.Linear(44944, 256),
                                smart_activation(opt),
                                nn.Linear(256, 256),
                                smart_activation(opt),
                                nn.Linear(256,opt.backbone_out)
                                )
        # print('out = out = ' + str(out))

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        # print(output.shape)
        output = self.fc(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)


class SiameseNet(nn.Module):
    def __init__(self, backbone):
        super(SiameseNet, self).__init__()
        self.backbone = backbone

        # self.nonlinear = nn.PReLU()
        # self.fc1 = nn.Linear(2, n_classes)

    def forward(self, x1, x2):
        output1 = self.backbone(x1)
        output2 = self.backbone(x2)
        # scores1 = F.log_softmax(self.fc1(output1), dim=-1)
        # scores2 = F.log_softmax(self.fc1(output2), dim=-1)
        return output1, output2
        # return output1, output2,scores1,scores2

    def get_embedding(self, x):
        return self.embedding_net(x)


class SiameseNet_4img(nn.Module):
    def __init__(self, embedding_net):
        super(SiameseNet_4img, self).__init__()
        self.embedding_net = embedding_net

        # self.nonlinear = nn.PReLU()
        # self.fc1 = nn.Linear(2, n_classes)

    def forward(self, x1, x2,x3 = None,x4 = None):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        if x3!= None and x4!= None:

            output3 = self.embedding_net(x3)
            output4 = self.embedding_net(x4)
            return  (output1, output3),(output2,output4)
        # scores1 = F.log_softmax(self.fc1(output1), dim=-1)
        # scores2 = F.log_softmax(self.fc1(output2), dim=-1)
        return output1, output2
        # return output1, output2,scores1,scores2

    def get_embedding(self, x):
        return self.embedding_net(x)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes,opt, stride=1,downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes, grayscale,opt):
        self.inplanes = 64
        self.opt = opt
        if grayscale:
            in_dim = 1
        else:
            in_dim = 3
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:  # 看上面的信息是否需要卷积修改，从而满足相加条件
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, self.opt,stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,self.opt))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # because MNIST is already 1x1 here:
        # disable avg pooling
        # x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        # probas = F.softmax(logits, dim=1)
        # return logits, probas
        return logits
def resnet18(opt):
    """Constructs a ResNet-18 model."""
    model = ResNet(block=BasicBlock,
                   layers=[2, 2, 2, 2],
                   num_classes=opt.backbone_out,
                   grayscale=True,opt = opt)
    return model


class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)