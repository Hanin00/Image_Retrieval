import sys

import GCN as md
import torch
import util as ut
from torch.nn.modules.module import Module
import torch.nn as nn
import pandas as pd
from gensim.models import FastText
import torch.optim as optim
import numpy as np


# gpu 사용
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#freObj(100)의 fastEmbedding 값 100 x 10
testFile = open('./data/freObj.txt','r') # 'r' read의 약자, 'rb' read binary 약자 (그림같은 이미지 파일 읽을때)
readFile = testFile.readline()
freObjList = (readFile[1:-1].replace("'",'')).split(',')
freObjList = freObjList[:100]
a = []
a.append(freObjList)
model = FastText(freObjList, vector_size=10, workers=4, sg=1, word_ngrams=1)

features = []
for i in freObjList:
    features.append(list(model.wv[i]))

features = torch.Tensor(features) # tensor(100x10)

#id X freObj = 이미지와 obj간의 관계 = 1000 x 100
adj = torch.FloatTensor(np.load('./data/idFreFeature.npy'))  # tensor(1000,100)


#
# #Data 분포 보고 싶어서 시각화 해봄.
# import seaborn as sns
# import matplotlib.pyplot as plt
#
# data = adj
# #pd.set_option('display.max_rows',None)
# pd.set_option('display.max_columns',None)
# df = pd.DataFrame(data, columns=freObjList)
# print(df)
# ax = sns.heatmap(df)
# plt.show()
#
# sys.exit()


testFile = open('./data/cluster.txt', 'r')  # 'r' read의 약자, 'rb' read binary 약자 (그림같은 이미지 파일 읽을때)
readFile = testFile.readline()
label = (readFile[1:].replace("'", '').replace(' ', '').split(','))
labels = []
for i in range(1000):
    labels.append(int(label[i]))

#features = torch.FloatTensor(features.todense())  # <class 'torch.Tensor'> (1000,100), sparseMatrix -> Tensor Matrix, torch.float32
labels = torch.LongTensor(labels)


idx_train, idx_val, idx_test = ut.splitDataset(34,200,300,len(features)) #train/val/test로 나눌 imageId를 가진 배열 return
ut.toDevice(adj, features, labels, idx_train, idx_val, idx_test) # gpu면 gpu로 cpu면 cpu로~


n_labels = labels.max().item() + 1  # 15
n_features = features.shape[1]  # 10
# seed 고정
torch.manual_seed(34)

# model
#GCN(  (gc1): GraphConvolution (100 -> 20)   (gc2): GraphConvolution (20 -> 15) )
model = md.GCN(nfeat=n_features,
            nhid=100,  # hidden = 16
            nclass=n_labels,
            dropout=0.5)  # dropout = 0.5
optimizer = optim.Adam(model.parameters(),
                       lr=0.001, weight_decay=5e-4)

epochs = 1000
print_steps = 100
train_loss, train_acc = [], []
val_loss, val_acc = [], []


for i in range(epochs):
    tl, ta = ut.train(model,optimizer,features, adj, idx_train,labels)
    train_loss += [tl]
    train_acc += [ta]
    if ((i + 1) % print_steps) == 0 or i == 0:
        tl, ta = ut.evaluate(idx_train, model,features, adj,labels)
        vl, va = ut.evaluate(idx_train, model,features, adj,labels)
        val_loss += [vl]
        val_acc += [va]

        print('Epochs: {}, Train Loss: {:.3f}, Train Acc: {:.3f}, Validation Loss: {:.3f}, Validation Acc: {:.3f}'.format(
                i, tl, ta, vl, va))

output = model(features, adj) #[100,15]
print(output.shape)
print(output)

sys.exit()

# test
samples = 10
# torch.randperm : 데이터 셔플
# idx_sample = idx_test[torch.randperm(len(idx_test))[:samples]]
idx_sample = idx_test[torch.randperm(len(idx_test))[:samples]]
print(torch.randperm(len(idx_test))[:samples])
print(idx_test)
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
