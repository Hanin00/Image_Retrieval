import util as ut
import util2 as ut2
import YEmbedding as yed
import numpy as np
import torch
import math
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn
import torch.nn.functional as F
import time
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tnrange
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix
from scipy.sparse import diags
from scipy.sparse import eye
from pathlib import Path
from functools import partial
import csv

from scipy.sparse import csr_matrix


# utils 이용해 cluster, Adj 생성

''' 해야하는 것
                        freObj(column) / cluster(label) / Adjset(all of Nums)
- freObj 100개일 때 5000 - freObj5000/ cluster5000 / Adjset5000
- freObj 100개일 때 10000 - freObj10000/ cluster10000 / Adjset10000
- freObj 500개일 때 5000 - freObj5000_500/ cluster5000_500 / Adjset5000_500
- freObj 500개일 때 10000 - freObj10000_500/ cluster10000_500 / Adjset5000_500


'''

def main():
    '''
        csv 파일로 만들어야 하는 것(이미지 천 개에 대한 데이터들) : region_descriptiong> image_regions
            columns Name(31310)
            featuremap(31310)
            이미지 1000개에 대한 adjMatrix(train 용)
    '''

    #featuremap = np.load('./data/idFreFeature.npy')
    #idAdj = np.load('./data/idAdj.npy')

    # AllAdj = np.load('./data/frexfre1000.npy',allow_pickle=True) #각 이미지의 freObjxfreObj 한 인접행렬 1000개
    # print(AllAdj[1][1])
    # print(torch.Tensor(AllAdj[1][1]))


    ''' id, Adj, label List 만드는 코드 '''
    #빈출 단어 값
    testFile = open('../data/freObj5000.txt', 'r')  # 'r' read의 약자, 'rb' read binary 약자 (그림같은 이미지 파일 읽을때)
    readFile = testFile.readline()
    freObj = (readFile[1:-1].replace("'", '').replace(' ', '')).split(',')
    freObj= freObj[:100] #빈출 100 단어 만 사용

    testFile = open('../data/cluster5000.txt', 'r')  # 'r' read의 약자, 'rb' read binary 약자 (그림같은 이미지 파일 읽을때)
    readFile = testFile.readline()
    label = (readFile[1:-1].replace("'", '').replace(' ', '')).split(',')
    label = label[:5000]  # 라벨 이미지 개수만큼 가져옴


    # # 임베딩값 freObj x embedding(10)
    # feature = ut.objNameEmbedding(label)

    dataset = []
    for i in range(5000) :
        adj = ut.createAdj(i, freObj)
        dataset.append([i+1, adj, label[i]])

    np.save('./data/frexfre5000.npy', np.ndarray(dataset))


    # features = np.load('./data/frexfre1000.npy', allow_pickle=True)[0]
    # print(features)

   # # features = csr_matrix(np.load('./data/frexfre1000.npy'), dtype=np.float32)
   #  adj = torch.FloatTensor(np.load('./data/idAdj.npy'))


'''
    embedding_clustering = yed.YEmbedding(xlxspath)
    # YEmbedding
    idCluster = embedding_clustering[['image_id','cluster','distance_from_centroid']]

    features = csr_matrix(paper_features_label[:, 1:-1], dtype=np.float32)
    labels = paper_features_label[:, -1]
    lbl2idx = {k: v for v, k in enumerate(sorted(np.unique(labels)))}
    labels = [lbl2idx[e] for e in labels]
    labels[:5]

    n_labels = labels.max().item() + 1
    n_features = features.shape[1]
    n_labels, n_features

    #train/val set 나눔
    np.random.seed(34)
    n_train = 200
    n_val = 300
    n_test = len(featureEmbedding) - n_train - n_val
    idxs = np.random.permutation(len(featureEmbedding))
    idx_train = torch.LongTensor(idxs[:n_train])
    idx_val = torch.LongTensor(idxs[n_train:n_train + n_val])
    idx_test = torch.LongTensor(idxs[n_train + n_val:])

    torch.manual_seed(34)

    model = GCN(nfeat=featureEmbedding,
                nhid=20,  # hidden = 16
                nclass=n_labels,
                dropout=0.5)  # dropout = 0.5
    optimizer = optim.Adam(model.parameters(),
                           lr=0.001, weight_decay=5e-4)

    def step():
        t = time.time()
        model.train()
        optimizer.zero_grad()
        output = model(features, adj)
        loss = F.nll_loss(output[idx_train], labels[idx_train])
        acc = accuracy(output[idx_train], labels[idx_train])
        loss.backward()
        optimizer.step()

        return loss.item(), acc

    def evaluate(idx):
        model.eval()
        output = model(features, adj)
        loss = F.nll_loss(output[idx], labels[idx])
        acc = accuracy(output[idx], labels[idx])

        return loss.item(), acc

    def accuracy(output, labels):
        preds = output.max(1)[1].type_as(labels)
        correct = preds.eq(labels).double()
        correct = correct.sum()
        return correct / len(labels)

    epochs = 1000
    print_steps = 100
    train_loss, train_acc = [], []
    val_loss, val_acc = [], []

    for i in tnrange(epochs):
        tl, ta = step()
        train_loss += [tl]
        train_acc += [ta]

        if ((i + 1) % print_steps) == 0 or i == 0:
            tl, ta = evaluate(idx_train)
            vl, va = evaluate(idx_val)
            val_loss += [vl]
            val_acc += [va]

            print(
                'Epochs: {}, Train Loss: {:.3f}, Train Acc: {:.3f}, Validation Loss: {:.3f}, Validation Acc: {:.3f}'.format(i, tl, ta, vl, va))

'''

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
