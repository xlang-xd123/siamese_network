import numpy as np
from PIL import Image
import warnings

warnings.filterwarnings("ignore")
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler

from torchvision.datasets import MNIST

# from torch.utils.data import Dataset
from PIL import Image
import os
from tqdm import tqdm
import torch
# import numpy as np
# from torchvision.datasets import MNIST
from torchvision import transforms




# a = np.array([[1,2],[3,4],[5,6]])
class GetData(Dataset):
    def __init__(self, root_dir, transform, train=True):
        self.train = train
        self.transform = transform
        item = open(root_dir, "r")
        self.item = item.readlines()
        #         print(self.item)
        gg = []
        ll = []
        pbar = tqdm(self.item)
        for i in pbar:
            pbar.set_description("Processing ")
            img, label = i[:-1].split(' ')
            # img = np.array(Image.open(img))
            gg.append(img)
            #             print('label' + label)
            ll.append(int(label))
            # print(img)
        #             print('still working '+ 'll')

        #             print(img.shape,label)
        #         if self.train == True:
        self.train_labels = torch.tensor(ll, )
        self.train_data = gg
        self.test_labels = torch.tensor(ll)
        self.test_data = gg

    #         print(self.item)
    #         # self
    #         self.root_dir = root_dir
    #         self.label_dir = label_dir
    #         self.path = os.path.join(self.root_dir, self.label_dir)
    #         self.img_path_list = os.lis

    def __getitem__(self, idx):
        #         if self.train :
        root, label = self.item[idx][:-1].split(' ')
        img = Image.open(root)
        if self.transform is not None:
            img = self.transform(img)
        return torch.tensor(img), int(label)

    def __len__(self):
        return len(self.item)


class SiameseMNIST(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, mnist_dataset):
        self.mnist_dataset = mnist_dataset

        self.train = self.mnist_dataset.train
        self.transform = self.mnist_dataset.transform

        if self.train:
            self.train_labels = self.mnist_dataset.train_labels
            self.train_data = self.mnist_dataset.train_data
            self.labels_set = set(self.train_labels.numpy())
            self.label_to_indices = {label: np.where(self.train_labels.numpy() == label)[0]
                                     for label in self.labels_set}
            # print(self.label_to_indices)
        else:
            # generate fixed pairs for testing
            self.test_labels = self.mnist_dataset.test_labels
            self.test_data = self.mnist_dataset.test_data
            self.labels_set = set(self.test_labels.numpy())
            self.label_to_indices = {label: np.where(self.test_labels.numpy() == label)[0]
                                     for label in self.labels_set}

            random_state = np.random.RandomState(29)

            positive_pairs = [[i,
                               random_state.choice(self.label_to_indices[self.test_labels[i].item()]),
                               1]
                              for i in range(0, len(self.test_data), 2)]

            negative_pairs = [[i,
                               random_state.choice(self.label_to_indices[
                                                       np.random.choice(
                                                           list(self.labels_set - set([self.test_labels[i].item()]))
                                                       )
                                                   ]),
                               0]
                              for i in range(1, len(self.test_data), 2)]
            self.test_pairs = positive_pairs + negative_pairs

    def __getitem__(self, index):
        if self.train:
            target = np.random.randint(0, 2)
            img1, label1 =self.load( self.train_data[index]), self.train_labels[index].item()
            if target == 1:
                siamese_index = index
                while siamese_index == index:
                    siamese_index = np.random.choice(self.label_to_indices[label1])
            else:
                siamese_label = np.random.choice(list(self.labels_set - set([label1])))
                siamese_index = np.random.choice(self.label_to_indices[siamese_label])
            img2 =self.load(self.train_data[siamese_index])
            label2 = self.train_labels[index].item()
        else:
            img1 =self.load( self.test_data[self.test_pairs[index][0]])
            img2 = self.load(self.test_data[self.test_pairs[index][1]])
            target = self.test_pairs[index][2]

        img1 = Image.fromarray(img1.numpy(), mode='L')
        img2 = Image.fromarray(img2.numpy(), mode='L')
        if self.transform is not None :
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        if self.train:

            return (img1, img2), target
        else:
            return (img1, img2), target

    def __len__(self):
        return len(self.mnist_dataset)
    def load(self,str):
        img = np.array(Image.open(str))

        img = torch.tensor(img)
        return img

class SiameseMNIST2(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, mnist_dataset):
        self.mnist_dataset = mnist_dataset

        self.train = self.mnist_dataset.train
        self.transform = self.mnist_dataset.transform

        if self.train:
            self.train_labels = self.mnist_dataset.train_labels
            self.train_data = self.mnist_dataset.train_data
            self.labels_set = set(self.train_labels.numpy())
            self.label_to_indices = {label: np.where(self.train_labels.numpy() == label)[0]
                                     for label in self.labels_set}
            # print(self.label_to_indices)
        else:
            # generate fixed pairs for testing
            self.test_labels = self.mnist_dataset.test_labels
            self.test_data = self.mnist_dataset.test_data
            self.labels_set = set(self.test_labels.numpy())
            self.label_to_indices = {label: np.where(self.test_labels.numpy() == label)[0]
                                     for label in self.labels_set}

            random_state = np.random.RandomState(29)

            positive_pairs = [[i,
                               random_state.choice(self.label_to_indices[self.test_labels[i].item()]),
                               1]
                              for i in range(0, len(self.test_data), 2)]

            negative_pairs = [[i,
                               random_state.choice(self.label_to_indices[
                                                       np.random.choice(
                                                           list(self.labels_set - set([self.test_labels[i].item()]))
                                                       )
                                                   ]),
                               0]
                              for i in range(1, len(self.test_data), 2)]
            self.test_pairs = positive_pairs + negative_pairs

    def __getitem__(self, index):
        if self.train:
            target = np.random.randint(0, 2)
            img1, label1 = self.train_data[index], self.train_labels[index].item()
            if target == 1:
                siamese_index = index
                while siamese_index == index:
                    siamese_index = np.random.choice(self.label_to_indices[label1])
            else:
                siamese_label = np.random.choice(list(self.labels_set - set([label1])))
                siamese_index = np.random.choice(self.label_to_indices[siamese_label])
            img2 = self.train_data[siamese_index]
            label2 = self.train_labels[index].item()
        else:
            img1 = self.test_data[self.test_pairs[index][0]]
            img2 = self.test_data[self.test_pairs[index][1]]
            target = self.test_pairs[index][2]

        img1 = Image.fromarray(img1.numpy(), mode='L')
        img2 = Image.fromarray(img2.numpy(), mode='L')
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        if self.train:

            return (img1, img2), (target,label1,label2)
        else:
            return (img1, img2), target

    def __len__(self):
        return len(self.mnist_dataset)

class SiameseMNIST_4imgs(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, mnist_dataset):
        self.mnist_dataset = mnist_dataset

        self.train = self.mnist_dataset.train
        self.transform = self.mnist_dataset.transform

        if self.train:
            self.train_labels = self.mnist_dataset.train_labels
            self.train_data = self.mnist_dataset.train_data
            self.labels_set = set(self.train_labels.numpy())
            self.label_to_indices = {label: np.where(self.train_labels.numpy() == label)[0]
                                     for label in self.labels_set}
            # print(self.label_to_indices)
        else:
            # generate fixed pairs for testing
            self.test_labels = self.mnist_dataset.test_labels
            self.test_data = self.mnist_dataset.test_data
            self.labels_set = set(self.test_labels.numpy())
            self.label_to_indices = {label: np.where(self.test_labels.numpy() == label)[0]
                                     for label in self.labels_set}

            self.random_state = np.random.RandomState(29)

            positive_pairs = [[i,
                               self.random_state.choice(self.label_to_indices[self.test_labels[i].item()]),
                               1]
                              for i in range(0, len(self.test_data), 2)]

            negative_pairs = [[i,
                               self.random_state.choice(self.label_to_indices[
                                                       np.random.choice(
                                                           list(self.labels_set - set([self.test_labels[i].item()]))
                                                       )
                                                   ]),
                               0]
                              for i in range(1, len(self.test_data), 2)]
            self.test_pairs = positive_pairs + negative_pairs

    def __getitem__(self, index):
        if self.train:
            target = np.random.randint(0, 2)
            img1, label1 = self.train_data[index], self.train_labels[index].item()
            if target == 1:
                siamese_index = index
                while siamese_index == index:
                    siamese_index = np.random.choice(self.label_to_indices[label1])
            else:
                siamese_label = np.random.choice(list(self.labels_set - set([label1])))
                siamese_index = np.random.choice(self.label_to_indices[siamese_label])
            img2 = self.train_data[siamese_index]
            label2 = self.train_labels[index].item()
        else:
            img1 = self.test_data[self.test_pairs[index][0]]
            img2 = self.test_data[self.test_pairs[index][1]]
            target = self.test_pairs[index][2]

        img1 = Image.fromarray(img1.numpy(), mode='L')
        img2 = Image.fromarray(img2.numpy(), mode='L')
        if not  self.train:
            # img3 = img1
            # img4 = img2
            # 增加 两个图片
            img3_label = np.random.choice(list(self.labels_set ))
            img3_index = np.random.choice(self.label_to_indices[img3_label])
            img3 = self.test_data[img3_index]

            img4_label = np.random.choice(list(self.labels_set ))
            img4_index = np.random.choice(self.label_to_indices[img4_label])
            img4 = self.test_data[img4_index]

            # print("img3_label = %d img3_index = %d img4_label = %d img4_index = %d"%(img3_label,img3_index,img4_label,img4_index))
            img3 = Image.fromarray(img3.numpy(), mode='L')
            img4= Image.fromarray(img4.numpy(), mode='L')
        if self.transform is not None and not self.train:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)
            img4 = self.transform(img4)

        elif self.transform is not None and  self.train:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        if self.train:



            return (img1, img2), target
        else:
            return (img1, img2,img3,img4), target

    def __len__(self):
        return len(self.mnist_dataset)

