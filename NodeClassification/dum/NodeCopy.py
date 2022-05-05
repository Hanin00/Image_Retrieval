# https://github.com/fahim-sikder/Node-Classification-GCN/blob/master/pytorch-GCN.ipynb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import util2 as ut2
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
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix
from scipy.sparse import diags
from scipy.sparse import eye
from pathlib import Path
from functools import partial
import sys

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

features = csr_matrix(np.load('../data/idFreFeature.npy'), dtype=np.float32) #csr_matrix : (1000,100)
adj = torch.FloatTensor(np.load('../data/idAdj.npy')) #tensor(1000,1000)
features = csr_matrix(features)


def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = (rowsum ** -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


features = normalize(features)
testFile = open('../data/cluster.txt', 'r')  # 'r' read의 약자, 'rb' read binary 약자 (그림같은 이미지 파일 읽을때)
readFile = testFile.readline()
label = (readFile[1:].replace("'", '').replace(' ', '').split(','))
labels = []
for i in range(1000):
    labels.append(int(label[i]))

features = torch.FloatTensor(features.todense())  # <class 'torch.Tensor'> (1000,100), sparseMatrix -> Tensor Matrix, torch.float32
labels = torch.LongTensor(labels)  # <class 'torch.Tensor'> tensor(1000, ) int64

# dataset train/test/val로 나눔
np.random.seed(34)
n_train = 200
n_val = 300
n_test = len(features) - n_train - n_val #500
idxs = np.random.permutation(len(features))
idx_train = torch.LongTensor(idxs[:n_train])
idx_val = torch.LongTensor(idxs[n_train:n_train + n_val])
idx_test = torch.LongTensor(idxs[n_train + n_val:])

# cuda.. gpu로 보냄
adj = adj.to(device)
features = features.to(device)
labels = labels.to(device)
idx_train = idx_train.to(device)
idx_val = idx_val.to(device)
idx_test = idx_test.to(device)


# Model
class GraphConvolution(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features   # out_features : 20, in_features : 100, self : unable to get repr for <class'__main__.GraphConvolution'>, bias : True
        self.out_features = out_features

        # weight reset
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))  # # out_features : 20, in_features : 100, self : GraphConvolution(100->20), bias : True

        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ') '


class GCN(nn.Module):   #nhid : 20, nfeat : 100, self : GCN(), nclass : 15, dropout : 0.5
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        # self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc1 = GraphConvolution(nfeat, nhid)   #nhid : 20, nfeat : 100, self : GCN(), nclass : 15, dropout : 0.5
                                                    # out_features : 20, in_features : 100, self : unable to get repr for <class'__main__.GraphConvolution'>, bias : True
        self.gc2 = GraphConvolution(nhid, nclass)  #nhid : 20, nfeat : 100, self : (gc1) : GraphConvolution(100->20), nclass : 15, dropout : 0.5
                                                     # in_features : 20, out_features : 15,  self : unable to get repr for <class'__main__.GraphConvolution'>, bias : True

        self.dropout = dropout

    # X : 초기 랜덤값 -> 학습 하면서 변경
    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)

        return F.log_softmax(x, dim=1)


n_labels = labels.max().item() + 1  # 15
n_features = features.shape[1]  # 100

# seed 고정
torch.manual_seed(34)

# model
#GCN(  (gc1): GraphConvolution (100 -> 20)   (gc2): GraphConvolution (20 -> 15) )
model = GCN(nfeat=n_features,
            nhid=20,  # hidden = 16
            nclass=n_labels,
            dropout=0.5).to(device)  # dropout = 0.5
optimizer = optim.Adam(model.parameters(),
                       lr=0.001, weight_decay=5e-4)


def step():
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
def evaluate(idx):
    model.eval()
    output = model(features, adj)  # 모델 돌림
    loss = F.nll_loss(output[idx], labels[idx])  # 모델이 분류한 값과 label 비교해서 loss 파악
    acc = accuracy(output[idx], labels[idx])

    return loss.item(), acc


# 정확도
def accuracy(output, labels): #output : tensor(200, 15), labels : tensor(200,)
    preds = output.max(1)[1].type_as(labels)  # 비슷하다고 뽑은 것들중에 제일 비슷한 거
    correct = preds.eq(labels).double()  #tensor (300,)
    correct = correct.sum()
    return correct / len(labels)


epochs = 1000
print_steps = 100
train_loss, train_acc = [], []
val_loss, val_acc = [], []

for i in range(epochs):
    tl, ta = step()
    train_loss += [tl]
    train_acc += [ta]

    if ((i + 1) % print_steps) == 0 or i == 0:
        tl, ta = evaluate(idx_train)
        vl, va = evaluate(idx_val)
        val_loss += [vl]
        val_acc += [va]

        print(
            'Epochs: {}, Train Loss: {:.3f}, Train Acc: {:.3f}, Validation Loss: {:.3f}, Validation Acc: {:.3f}'.format(
                i, tl, ta, vl, va))

output = model(features, adj)

# test
samples = 10
# torch.randperm : 데이터 셔플
# idx_sample = idx_test[torch.randperm(len(idx_test))[:samples]]
idx_sample = idx_test[torch.randperm(len(idx_test))[:samples]]
print(idx_sample)

idx2lbl = ['0번 그림', '1번 그림', '2번 그림', '3번 그림', '4번 그림', '5번 그림', '6번 그림', '7번 그림'
    , '8번 그림', '9번 그림', '10번 그림', '11번 그림', '12번 그림', '13번 그림', '14번 그림']
# 실제 라벨 10개, predicate 결과 10개

# numpy.argmax() : 주어진 배열에서 가장 높은 값을 가진 인덱스 반환
# pd.tolist() : 동일 위치(레벨)에 있는 데이터끼리 묶어줌
df = pd.DataFrame({'Real': [idx2lbl[e] for e in labels[idx_sample].tolist()],
                   'Pred': [idx2lbl[e] for e in output[idx_sample].argmax(1).tolist()]})
# [idx2lbl[e] for e in output[idx_sample].argmax(1).tolist()] : 예측한 값이 label의 어디에 속하는지
# Pred
# idx_sample 개수만큼의 output을 가져오는데, argmax로 output 값 15개(각 cluster에 대한 값) 중 가장 큰 값을 뽑고, 그 위치를 받음
# tolist()로 각 feature 별 model의 output 값을 받음
# for 문을 통해 idx2Lble에서 해당하는 label을 반환받음.
print(df)

'''
    각 범위 별 이미지 값으로 동일 여부 확인
    범위 추가하는 변수2개 만들면 될 듯
    -> 이 T/F 값을 model에 반영 시키는 방법?
'''

idx_sample1 = idx_test[torch.randperm(len(idx_test))[1:10]]
realList1 = [idx2lbl[e] for e in labels[idx_sample1].tolist()]
predList1 = [idx2lbl[e] for e in output[idx_sample1].argmax(1).tolist()]

idx_sample2 = idx_test[torch.randperm(len(idx_test))[22:31]]
realList2 = [idx2lbl[e] for e in labels[idx_sample2].tolist()]
predList2 = [idx2lbl[e] for e in output[idx_sample2].argmax(1).tolist()]

resPred = []
for i in range(len(predList2)):
    if (predList1[i] == predList2[i]):
        resPred.append('T')
    else:
        resPred.append('F')

resReal = []
for i in range(len(realList1)):
    if (realList1[i] == realList2[i]):
        resReal.append('T')
    else:
        resReal.append('F')

# img_id 값으로 그림 확인하고 싶어서 id 찍어봄
print("idx_sample1_Imgid : ", idx_sample1)
print("idx_sample2_Imgid : ", idx_sample2)
idDF1 = pd.DataFrame({'idx_sample1_Imgid': idx_sample1,
                      'idx_sample2_Imgid': idx_sample2})
print(idDF1)

df1 = pd.DataFrame({'Pre1': [idx2lbl[e] for e in output[idx_sample1].argmax(1).tolist()],
                    'Pred2': [idx2lbl[e] for e in output[idx_sample2].argmax(1).tolist()],
                    'Res(T/F)': resPred})
df2 = pd.DataFrame({'Real1': [idx2lbl[e] for e in labels[idx_sample1].tolist()],
                    'Real2': [idx2lbl[e] for e in labels[idx_sample2].tolist()],
                    'Res(T/F)': resReal})
print(df1)
print(df2)
