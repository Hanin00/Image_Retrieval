# https://github.com/fahim-sikder/Node-Classification-GCN/blob/master/pytorch-GCN.ipynb
import sys

import numpy as np
import torch
import torch.nn.functional as F
import time
# import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse import diags
import util2 as ut2
import pickle

'''GCN 학습 및 검증 관련 utility'''


'''
GNN Node Classification
dataset : Visual Genome.Scence graph 

Adj : id x id (같은 cluster 값을 갖는 경우 1로 체크함), 1000x1000
Feature : id x freOBJ(해당 img에 FreOBJ가 있는 경우 1, 없는 경우 0)
freObj : 대상이 되는 이미지 1000개의 Scene graph에서 가장 언급량이 많은 Obj 100개
Label : 각 id 당 cluster 번호
cluster는 bert-base-nli-mean-tokens를 이용해 15개의 클러스터로 분류했음
-> 1000개의 region_graph의 phrase 값을 embedding 함
-> 이 부분도 relationship이 잘 나타나지 않은 것 같아 아쉬움
relationship을 더 잘 활용할 수 있는걸 하고싶음..

'''

# gpu 사용
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = (rowsum ** -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def loadData_raw():
    features = csr_matrix(np.load('./data/idFreFeature.npy'), dtype=np.float32)  # csr_matrix : (1000,100)
    adj = torch.FloatTensor(np.load('./data/idAdj.npy'))  # tensor(1000,1000)
    features = csr_matrix(features)

    features = normalize(features)

    testFile = open('./data/cluster.txt', 'r')  # 'r' read의 약자, 'rb' read binary 약자 (그림같은 이미지 파일 읽을때)
    readFile = testFile.readline()
    label = (readFile[1:].replace("'", '').replace(' ', '').split(','))
    labels = []
    for i in range(1000):
        labels.append(int(label[i]))

    features = torch.FloatTensor(
        features.todense())  # <class 'torch.Tensor'> (1000,100), sparseMatrix -> Tensor Matrix, torch.float32
    labels = torch.LongTensor(labels)  # <class 'torch.Tensor'> tensor(1000, ) int64

    # floatTensor, floatTensor, LongTensor로 반환

    return features, adj, labels


# 난수 seed 값, 개수들. 단순 랜덤 int 값 배열 반환이니까~(image id로 사용할)
def splitDataset(seed, n_train, n_val, n_features):
    # dataset train/test/val로 나눔
    np.random.seed(seed)
    n_train = n_train
    n_val = n_val
    n_test = n_features - n_train - n_val  # 500
    idxs = np.random.permutation(n_features)
    idx_train = torch.LongTensor(idxs[:n_train])
    idx_val = torch.LongTensor(idxs[n_train:n_train + n_val])
    idx_test = torch.LongTensor(idxs[n_train + n_val:])

    return idx_train, idx_val, idx_test


# cuda.. gpu로 보냄
def toDevice(adj, features, labels, idx_train, idx_val, idx_test):
    adj = adj.to(device)
    features = features.to(device)
    labels = labels.to(device)
    idx_train = idx_train.to(device)
    idx_val = idx_val.to(device)
    idx_test = idx_test.to(device)


# 학습
# def step():
def train(model, optimizer, features, adj, idx_train, labels):
    t = time.time()
    model.train()  # model 학습모드로
    optimizer.zero_grad()
    output = model(features, adj)  # model에 값 넣음  tensor(1000,15)
    loss = F.nll_loss(output[idx_train], labels[idx_train])  # loss 함수   tensor, torch.float32
    acc = accuracy(output[idx_train], labels[idx_train])  # accuracy 파악
    loss.backward()
    optimizer.step()

    return loss.item(), acc

# 평가
def evaluate(idx, model, features, adj, labels):
    model.eval()
    output = model(features, adj)  # 모델 돌림
    loss = F.nll_loss(output[idx], labels[idx])  # 모델이 분류한 값과 label 비교해서 loss 파악
    acc = accuracy(output[idx], labels[idx])

    return loss.item(), acc


# 정확도
def accuracy(output, labels):  # output : tensor(200, 15), labels : tensor(200,)
    preds = output.max(1)[1].type_as(labels)  # 비슷하다고 뽑은 것들중에 제일 비슷한 거
    correct = preds.eq(labels).double()  # tensor (300,)
    correct = correct.sum()
    return correct / len(labels)




#1000개 이미지에 대한 1000x(100x100) = imageId x (freObj x freObj) = (1000,10000) Feature map 생성
def loadData():
    with open("./data/feature_model2.pickle", "rb") as fr:
        dataset = pickle.load(fr) #type(dataset) : list
    #print(type(dataset))

    features = []
    for i in range(len(dataset)) :
        features.append(list((dataset[i]).flatten()))


    features = torch.FloatTensor(features)
    adj = torch.FloatTensor(np.load('./data/idAdj.npy'))  # tensor(1000,1000)
    testFile = open('./data/cluster.txt', 'r')  # 'r' read의 약자, 'rb' read binary 약자 (그림같은 이미지 파일 읽을때)
    readFile = testFile.readline()
    label = (readFile[1:].replace("'", '').replace(' ', '').split(','))
    labels = []
    for i in range(1000):
        labels.append(int(label[i]))
    labels = torch.LongTensor(labels)  # <class 'torch.Tensor'> tensor(1000, ) int64

    # floatTensor, floatTensor, LongTensor로 반환

    return features, adj, labels

features, adj, labels = loadData()



class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count