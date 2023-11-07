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
            img1 = img1.convert('L')

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

class TripletMNIST(Dataset):
    """
    Train: For each sample (anchor) randomly chooses a positive and negative samples
    Test: Creates fixed triplets for testing
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

        else:
            self.test_labels = self.mnist_dataset.test_labels
            self.test_data = self.mnist_dataset.test_data
            # generate fixed triplets for testing
            self.labels_set = set(self.test_labels.numpy())
            self.label_to_indices = {label: np.where(self.test_labels.numpy() == label)[0]
                                     for label in self.labels_set}

            random_state = np.random.RandomState(29)

            triplets = [[i,
                         random_state.choice(self.label_to_indices[self.test_labels[i].item()]),
                         random_state.choice(self.label_to_indices[
                                                 np.random.choice(
                                                     list(self.labels_set - set([self.test_labels[i].item()]))
                                                 )
                                             ])
                         ]
                        for i in range(len(self.test_data))]
            self.test_triplets = triplets

    def __getitem__(self, index):
        if self.train:
            # img1, label1 = self.train_data[index], self.train_labels[index].item()
            img1, label1 = self.load(self.train_data[index]), self.train_labels[index].item()
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[label1])
            negative_label = np.random.choice(list(self.labels_set - set([label1])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])
            img2 = self.load(self.train_data[positive_index])
                # self.train_data)[positive_index]
            img3 = self.load(self.train_data[negative_index])


        else:
            # img1 = self.test_data[self.test_triplets[index][0]]
            # img2 = self.test_data[self.test_triplets[index][1]]
            # img3 = self.test_data[self.test_triplets[index][2]]

            img1 = self.load(self.test_data[self.test_triplets[index][0]])
            img2 = self.load(self.test_data[self.test_triplets[index][1]])
            img3 = self.load(self.test_data[self.test_triplets[index][2]])

        img1 = Image.fromarray(img1.numpy(), mode='L')
        img2 = Image.fromarray(img2.numpy(), mode='L')
        img3 = Image.fromarray(img3.numpy(), mode='L')
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)
        return (img1, img2, img3), []

    def __len__(self):
        return len(self.mnist_dataset)
    def load(self,str):
        img = np.array(Image.open(str))

        img = torch.tensor(img)
        return img


class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, labels, n_classes, n_samples):
        self.labels = labels
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.labels)
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            # print(self.n_classes,self.labels_set)
            classes = np.random.choice(self.labels_set, self.n_classes, replace=True)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
                # print(indices)
                # print('hello')
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return self.n_dataset // self.batch_size
    def load(self,str):
        img = np.array(Image.open(str))

        img = torch.tensor(img)
        return img