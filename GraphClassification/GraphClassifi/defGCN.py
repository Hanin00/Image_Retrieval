import dgl
import torch
import torch.nn as nn
import sys
import torch
import util as ut
import util2 as ut2
import torch.optim as optim
import pickle
import torch.nn.functional as F
import numpy as np
from gensim.models import FastText
import torch.utils.data as utils
from torch.utils.data import Dataset, DataLoader
from datasetTest import GraphDataset
from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import defGCNModel as md
import random
from tqdm import tqdm

'''
데이터를 늘리지 않은 GraphClassification Model
'''
# gpu 사용
USE_CUDA = torch.cuda.is_available() # GPU를 사용가능하면 True, 아니라면 False를 리턴
device = torch.device("cuda" if USE_CUDA else "cpu") # GPU 사용 가능하면 사용하고 아니면 CPU 사용
print("다음 기기로 학습합니다:", device)


print("여기1")
#features, adj, labels = ut.loadData()


#text라 for문을 돌리든 뭘 하든 str -> int로 변경 후 tensor로 변경 해줘야함
# #labels
testFile = open('./data/cluster.txt', 'r')  # 'r' read의 약자, 'rb' read binary 약자 (그림같은 이미지 파일 읽을때)
readFile = testFile.readline()
labelsList = ((readFile[1:-1].replace("'", '')).replace(" ", '')).split(',')

labels = labelsList[:1000]
labels = list(map(int, labels))

labels = torch.FloatTensor(labels) 

print("여기2")

#features
# freObj(100)의 fastEmbedding 값 100 x 10
#testFile = open('./data/freObj5000.txt', 'r')  # 'r' read의 약자, 'rb' read binary 약자 (그림같은 이미지 파일 읽을때)

testFile = open('./data/freObj.txt', 'r')
readFile = testFile.readline()
freObjList = (readFile[1:-1].replace("'", '')).split(',')
freObjList = freObjList[:100]
model = FastText(freObjList, vector_size=10, workers=4, sg=1, word_ngrams=1)
 
features = []
for i in freObjList:
    features.append(list(model.wv[i]))
features = torch.FloatTensor(features)  # tensor(100x10)


with open("./data/frefre1000.pickle", "rb") as fr:
    data = pickle.load(fr)
adj = data


features, adj, labels  = features.to(device), adj.to(device), labels.to(device)
dataset = GraphDataset(adj, labels)
dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, drop_last=False)

num_examples = len(dataset)
num_train = int(num_examples * 0.8)

train_sampler = SubsetRandomSampler(torch.arange(num_train).to(device))
test_sampler = SubsetRandomSampler(torch.arange(num_train, num_examples).to(device))

train_dataloader = GraphDataLoader(
    dataset, sampler=train_sampler, batch_size=1, drop_last=False)
test_dataloader = GraphDataLoader(
    dataset, sampler=test_sampler, batch_size=1, drop_last=False)

#it = iter(train_dataloader)
#batch = next(it)

n_labels = 15  # 15
n_features = features.shape[1]  # 10  #features = Tensor(100,10)

model = md.GCN(n_features, 15, n_labels)  #n_features = 100, n_labels = 15
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-12)

for epoch in tqdm(range(1000)):
    random.seed(epoch)
    torch.manual_seed(epoch)
    if device == 'cuda':
        torch.cuda.manual_seed_all(epoch)
    #num_correct = 0

    #batched_graph : 1,100,100, labels :    attr : 1,15
    for batched_graph, labels,attr in train_dataloader:
        batched_graph, labels, attr = batched_graph.to(device), labels.to(device), attr.to(device)
        batched_graph = batched_graph.squeeze().to(device)
        pred = model(batched_graph, features).to(device) #tensor(1,100,100), features = Tensor(100,10)    
        #print(pred[0].argmax()) #동일값 출력하는 오류 발견
        loss = F.nll_loss(pred[0], attr.squeeze().long()).to(device)
 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

num_correct = 0
num_tests = 0

for batched_graph, labels,attr in test_dataloader:
    batched_graph, labels, attr = batched_graph.to(device), labels.to(device), attr.to(device)
    batched_graph = batched_graph.squeeze().to(device)
    attr = attr.squeeze().long().to(device)
    pred = model(batched_graph, features).to(device)
    num_correct += (torch.argmax(pred[0]) == torch.argmax(attr.squeeze().long())).sum().item()
    num_tests += len(labels)

print('Test accuracy:', num_correct / num_tests)




# dataloader 참고 :
# https://docs.dgl.ai/en/0.6.x/new-tutorial/5_graph_classification.html - dgl doc