import sys
import GCN as md
import util as ut
import util2 as ut2
from torch.nn.modules.module import Module
import torch.nn as nn
# import pandas as pd
import torch.optim as optim
import pickle
import torch.nn.functional as F
import numpy as np
from gensim.models import FastText
import torch.utils.data as utils
from torch.autograd import Variable
import torch
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')

class GraphDataset(Dataset):
    def __init__(self, x_tensor, y_tensor, ):
        super(GraphDataset, self).__init__()
        self.x = x_tensor
        self.y = y_tensor.to(device)
        y_one_hot = torch.zeros((1000, 15)).to(device)
        yUnsqueeze = y_tensor.unsqueeze(1).to(device)
        print(yUnsqueeze.is_cuda)
        
        y_one_hot.scatter_(1, yUnsqueeze, 1)
        
        self.attr = y_one_hot
        self.AdjList = []
        self.classList = []

    def __getitem__(self, index):
        # imgAdj = self.AdjList[idx]
        # label = self.classList[idx]
        imgAdj = self.x[index]
        label = self.y[index]
        attr = self.attr[index]
        return imgAdj, label, attr

    def __len__(self):
        return len(self.x)


if __name__ == "__main__":
    print(len(Images))
    print(len(labels))
    dataset = GraphDataset(Images, labels)
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, drop_last=False)

    for epoch in range(2):
        print(f"epoch : {epoch}")
        for adj, label, attr in dataloader:
            print("epoch",epoch, "label : ", label, "onehot ,attr : ", attr)
