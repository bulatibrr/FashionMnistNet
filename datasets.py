import numpy as np
import torch
from torch.utils.data import Dataset


NUM_CLASSES = 10
IMG_SIZE = 28


class FashionMnistDataset(Dataset):
    def __init__(self, root):
        self.root = root

        data = np.loadtxt(fname=root, delimiter=',', skiprows=1)
        self.images = torch.from_numpy(data[:, 1:].reshape((data.shape[0],1,IMG_SIZE,IMG_SIZE))).float()
        self.labels = torch.from_numpy(data[:, 0]).long()


    def __getitem__(self, index):
        return self.images[index], self.labels[index]


    def __len__(self):
        return self.labels.shape[0]


class FashionMnistTrainDataset(FashionMnistDataset):
    def __init__(self, root='data/fashion-mnist_train.csv'):
        super().__init__(root)


class FashionMnistTestDataset(FashionMnistDataset):
    def __init__(self, root='data/fashion-mnist_test.csv'):
        super().__init__(root)
