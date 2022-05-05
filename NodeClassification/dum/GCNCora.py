# https://github.com/fahim-sikder/Node-Classification-GCN 코드 주석
# https://chowdera.com/2022/04/202204020727068216.html
# https://tutorials.pytorch.kr/beginner/blitz/neural_networks_tutorial.html pytorch nn tutorial
from __future__ import print_function
from __future__ import division
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
import matplotlib


# Running Environment 구축,
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')  # gpu 또는 cpu 설정
path = Path('data/cora') #데이터의 path 설정
paper_features_label = np.genfromtxt(path/'cora.content', dtype=np.str)  #데이터셋, (2078, 1435), [[논문 id, fea..ture...., 'label'] ,...,[논문 id, fea..ture...., 'label']]

# csr_matrix : indptr[i]는 i번째 행의 원소가 data의 어느 인덱스에서 시작되는지
features = csr_matrix(paper_features_label[:, 1:-1], dtype=np.float32) # csr_matrix(2078,1433), id와 label 떼어냄, float 형
labels = paper_features_label[:, -1]  #각 논문에 해당하는 labels, 2708
lbl2idx = {k:v for v,k in enumerate(sorted(np.unique(labels)))}# 라벨에 인덱싱(라벨-> 숫자) {'Case_Based': 0, 'Genetic_Algorithms': 1, 'Neural_Networks': 2, 'Probabilistic_Methods': 3, 'Reinforcement_Learning': 4, 'Rule_Learning': 5, 'Theory': 6}
labels = [lbl2idx[e] for e in labels] #인덱싱 한 값으로 변환해 리스트로 저장


papers = paper_features_label[:,0].astype(np.int32) # paper id str -> int / ndarray (2708) 각 페이퍼에 인덱싱
paper2idx = {k:v for v,k in enumerate(papers)} #{dict : 2708} papaer id-> idx
edges = np.genfromtxt(path/'cora.cites', dtype=np.int32)  # 각 논문간 인용 관계, AdjMatric의 relationship으로 사용 ndarray(5429,2 )
edges = np.asarray([paper2idx[e] for e in edges.flatten()], np.int32).reshape(edges.shape)
#edge.flatten()을 for문으로 돌려서 e에 해당하는 paper2idx를 ndarray로 변환하는데, 각 값은 int32임(10858,)-> reshape을 통해 edge.shape인 (5429,2)로 변환됨
#새 번호의 노드 간의 관계로 변환 : 데이터를 논문 id로 변경
# asarray : 입력 데이터를 ndarray로 변환, 이미 ndarray일 경우에는 새로 메모리에 ndarray가 생성되지는 않음



#np.ones : 1로 가득찬 array 생성
#coo_matrix(대상이 되는 매트릭스, (i[:,0], j[:,1]) -> i는 matrix의 행 색인을 이용, j는 열 색인을 이용
#(edges[:, 0], edges[:, 1]) -> edge matrix 전체 길이에 해당하는 0번과 1번
#coo_matrix, (2708,2708)
adj = coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                 shape=(len(labels), len(labels)), dtype=np.float32)
# 무방향 대칭 행렬 생성, 방향 그래프의 인접 행렬이 무방향 그래프의 인접 행렬로 변환됨
# coo_matrix.t : coo_matrix의 차원 반전
# coo_matrix.multiply : 행렬곱
# 인자에 > ? 마스크 씌우는 건가?
adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

# 행 정규화. feature 값을 0-1사이의 값으로 변환 시키는 과정?
# Normalization을 통해 학습을 더 빠르게 할 수 있고, Local Optimum에 빠지는 가능성을 줄임
# -> https://velog.io/@kjb0531/RL%EC%97%90%EC%84%9C-Normalize%ED%95%98%EB%8A%94-%EC%9D%B4%EC%9C%A0
# https://stml.tistory.com/43
def normalize(mx): #csr_matrix : (2708, 1433) -> features, type(features) == float32
    # csr_matrix.sum = 주어진 축에 대해 행렬을 더함, 축이 없으면 행과 열을 모두 합쳐 scalar를 반환
    rowsum = np.array(mx.sum(1)) # ndarray(2708,1) -> ex) [[20.],[17.][22.],...]
    r_inv = (rowsum ** -1).flatten() # ** : 제곱, **-1 -> 0-1 값 사이로 만들기 위해?: flatten()
    r_inv[np.isinf(r_inv)] = 0. #np.isnf(array객체) : 객체의 각 원소를 판단해 infinity일 경우 True -> infinite 값인 경우 0. 으로 변환?
    r_mat_inv = diags(r_inv) #diags(array) : array의 대각선 값 1차원 array로, type(r_mat_inv) : ndarray
    mx = r_mat_inv.dot(mx) #numpy.dot : 두 array의 내적(Dot product) 계산 ->  numpy array 곱할 때 사용
    return mx #(2708, 1433) -> 0.05
#feature matrix에 1차원 늘리고 **-1로 0-1사이의 값으로 변경 -> flatten -> isinf 이용해 infinity일 경우 True로 마스크 씌우고, 해당 값들을 0으로 변경 -> 이렇게 변경한 r_mat_inv와 mx(feature Matrix)를 곱해 필요한 값만 남김

features = normalize(features)

adj = normalize(adj + eye(adj.shape[0])) #Normalizing for removing gradient vanishing and exploding problem 기울기 소실과 폭주가 어떤 식으로 발생?
# np.identity : 대각 행렬, 대각선이 1인 행렬을 생성
# np.eye : 대각 행렬을 만드는데 identity와 다르게 k값을 이용해 원하는 위치에서 identity 시작 가능
# csr_martix(2708,2708),  data= {ndarray(10556,)}

#todense() : sparse matrix to dense tensor / FloatTenser :  dtype(torch.float32)
adj = torch.FloatTensor(adj.todense()) #tensor(2708,2708)
features = torch.FloatTensor(features.todense()) #tensor(2708,1433)
labels = torch.LongTensor(labels) #tensor(2708,) , tensor([2, 5, 4,  ..., 1, 0, 2]), LongTensor : dtype(torch.int64)


np.random.seed(34) #랜덤 씨드값 설정 -> 일정한 값이 나오도록 유도함
n_train = 200  #Train set의 양 지정
n_val = 300 #Validation set의 양 지정
n_test = len(features) - n_train - n_val # Test는 train과 val을 제외한 만큼
idxs = np.random.permutation(len(features)) # np.random.permutation() : 무작위로 섞인 배열을 만듦, 숫자가 들어가면 해당 숫자까지의 범위를 가진 배열을 만듦
# 무작위로 id를 섞은 배열에서 train/val/test의 id 리스트를 생성함
idx_train = torch.LongTensor(idxs[:n_train]) #tensor(200,)
idx_val   = torch.LongTensor(idxs[n_train:n_train+n_val]) #tensor(300,)
idx_test  = torch.LongTensor(idxs[n_train+n_val:]) #tensor(2208,)


#to device: 설정된 device에 tensor를 할당함
adj = adj.to(device)
features = features.to(device)
labels = labels.to(device)
idx_train = idx_train.to(device)
idx_val = idx_val.to(device)
idx_test = idx_test.to(device)

#ref) https://programs.wiki/wiki/am-gcn-model-structure-of-code-interpretation.html
# module 클래스 상속
# GCN Layer, single layer neural network
class GraphConvolution(Module):
    # 매개 변수나 모듈등을 초기화 하는데 사용
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__() #초기화
        self.in_features = in_features # 전체 클래스의 모든 함수 내에서 사용할 수 있는 초기화 변수를 정의하는데 사용
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features)) #weight 매개변수 설정. GCN을 정의하는 핵심동작
        #parameter : Layer가 아닌 파라미터 값만 갖고 있음. 모듈의 파라미터 값들을 iterator로 반환함. object type은 torch.Tensor임
        #module의 attribute로 할당하면 자동으로 파라메터 리스트에 추가됨 
        #self.weight.torch.mm의 사용범위는 2차원 행렬. input.shapt = [B,N,F]면 사용되지 않음
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters() #매개변수(가중치) 초기화. 대부분 인자로 weight를 받음. 여기서는 아예 함수를 따로 파서 self.weight를 이용해 초기화

#ref) https://discuss.pytorch.org/t/how-are-layer-weights-and-biases-initialized-by-default/13073/2
    # self : GraphConvolution (1433 -> 20)
    def reset_parameters(self):  # 초기화 메소드. 굳이 이렇게 안만들어도 모듈의 reset_parameters()를 호출해 사용 가능
        stdv = 1. / math.sqrt(self.weight.size(1)) #stdb : 0.22360679774997896
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight) #행렬 곱, input 데이터(feature)와 adj를 곱함
        output = torch.spmm(adj, support) #희소행렬 곱  -> Adj X (feature matrix X weight)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ') '


'''
    nn.Module class : pytorch에서 model을 만들 때 상속하는 ㄹ래스
    모델의 뼈대
    - 딥러닝을 구성하는 Layer의 base class
    - Input, Output, Forward, backward 등의
    - 학습의 대상이 되는 parameter(tensor) 정의
'''

'''
    nfeat == in_features == 입력되는 데이터의 차원
    nhid == hidden layer == 은닉층의 차원
    nclass == out_features == 출력되는 차원 == 출력되는 클래스(label)의 개수
    
    각 인자의 type은 tensor지만 Tensor를 직접적으로 입력하면 값이 나오지 않음 -> Tensor는 미분의 대상이 되지 않기 때문
'''
#GCN 모듈, 다층 신경망
class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout): # nfeat : 1433     nhid : 20     nclass : 7
        super(GCN, self).__init__()   #self : self
        self.gc1 = GraphConvolution(nfeat, nhid) #입력 데이터의 차원과 hidden layer의 차원                                  # self : gc1) : GraphConvolution(1433->20)
        self.gc2 = GraphConvolution(nhid, nclass) # hidden layer의 차원, 출력 데이터의 차원(해당 예제에서는 논문의 label 수)   # self : : GraphConvolution(20->7), in_features : 20, out_features : 7, bias = True
        self.dropout = dropout #Dropout의 비율, over fitting을 막기 위해서 사용, 학습 시 레어어 간 연결 중 일부를 랜덤하게 삭제해, 일반화 성능이 높아짐.


    def forward(self, x, adj): # 순전파, 학습
        x = F.relu(self.gc1(x, adj)) # relu, 활성화 함수, input = tensor , output = tensor
        x = F.dropout(x, self.dropout, training=self.training) # dropout을 통해 overfitting을 막음
        x = self.gc2(x, adj) #GCN레이어에 한 번 더 넣음
        return F.log_softmax(x, dim=1)  #
#soft max : NN 최상위 층에 사용해 classifiaction을 위한 function으로 사용됨 -> 여기서 사용한 log_softmax와 다른 것, 둘 다 Loss function
#


n_labels = labels.max().item() + 1  # 7
n_features = features.shape[1] #1433

torch.manual_seed(34) #난수 생성 seed 설정
#모델 정의
model = GCN(nfeat=n_features,
            nhid=20,
            nclass=n_labels,
            dropout=0.5)  # dropout = 0.5

#최적화, Train Data set을 이용해 모델을 학습할 때 데이터의 실제 결과와 모델이 예측한 결과를 기반으로 잘 줄일 수 있게 만들어줌
#loss function의 최소값을 찾는 것을 학습의 목표로 함
#loss function : 학습하면서 얼마나 틀리는 지 알게 하는 함수... 
optimizer = optim.Adam(model.parameters(),
                       lr=0.001, weight_decay=5e-4)
#learning rate : 한 걸음 보폭을 얼만큼으로 할 지
# weight_decay : weight의 제곱합을 패널티 텀으로 주어 (=제약을 걸어) loss를 최소화 하는 것

'''
zero_grad()에 대한 이해
ref) https://velog.io/@kjb0531/zerograd%EC%9D%98-%EC%9D%B4%ED%95%B4

미니 배치+루프 조합을 사용해 parameter를 업데이트함
한 루프에서 업데이트를 위해 loss.bvackward()를 호출하면 각 파라미터들의 .grad 값에 변화도가 저장됨
-> 다음 루프에서 zero_grad()하지 않고 역전파를 시키면 이전 루프에서 .grad에 저장된 값이 다음 루프의 업데이트에도 간섭을 해 원하는 방향으로 학습이 안됨
'''

#학습 과정 다른 곳에선 굳이 step으로 구현하지 않음
def step(): 
    t = time.time()
    model.train() #model을 학습 모드로 바꿈
    optimizer.zero_grad() #변화도 버퍼를 0으로
    output = model(features, adj)
    loss = F.nll_loss(output[idx_train], labels[idx_train]) #손실 함수. cross entropy 구함, Classification에서 오차함수로 사용함.
    acc = accuracy(output[idx_train], labels[idx_train]) # dtype = torch.float64,
    loss.backward() #전체 그래프는 신경망릐 매개변수에 따라 미분되며, 그래프 내의 requires_grad=True인 모든 Tensor는 변화도가 누적된 .grad Tensor를 갖게 됨
    optimizer.step() #업데이트 진행

    return loss.item(), acc


def evaluate(idx):
    model.eval() #evaluation 모드로 변경. Evaluation에서는 Dropout/BatchNorm 을 하지 않음
    output = model(features, adj) #결과값, Tensor(2718, 7)
    loss = F.nll_loss(output[idx], labels[idx]) #모델이 분류한 값과 label 비교해서 loss 파악
    acc = accuracy(output[idx], labels[idx])

    return loss.item(), acc


def accuracy(output, labels): #labels : Tensor(200,), output : Tensor(200,7)
    preds = output.max(1)[1].type_as(labels)  # 비슷하다고 뽑은 것들중에 제일 비슷한 거(제일 많이 나온 값)
    correct = preds.eq(labels).double() #Tensor.eq : 같은지 비교.
    correct = correct.sum()
    return correct / len(labels)


epochs = 1000
print_steps = 100
train_loss, train_acc = [], []
val_loss, val_acc = [], []

for i in range(epochs):
    tl, ta = step()
    train_loss += [tl] #step의 결과로 나온 loss와 acc를 각각 저장함
    train_acc += [ta]

    if ((i + 1) % print_steps) == 0 or i == 0:
        tl, ta = evaluate(idx_train)
        vl, va = evaluate(idx_val)
        val_loss += [vl]
        val_acc += [va]

        print(
            'Epochs: {}, Train Loss: {:.3f}, Train Acc: {:.3f}, Validation Loss: {:.3f}, Validation Acc: {:.3f}'.format(
                i, tl, ta, vl, va))


fig, axes = plt.subplots(1,2, figsize=(15,5))

ax = axes[0]
axes[0].plot(train_loss[::print_steps] + [train_loss[-1]], label='Train')
axes[0].plot(val_loss, label='Validation')
axes[1].plot(train_acc[::print_steps] + [train_acc[-1]], label='Train')
axes[1].plot(val_acc, label='Validation')
axes[0].grid()
axes[1].grid()

for ax, t in zip(axes, ['Loss','Accuracy']):
    ax.legend(), ax.set_title(t, size=15)

output = model(features, adj)



samples = 10
idx_sample = idx_test[torch.randperm(len(idx_test))[:samples]]

idx2lbl = {v:k for k,v in lbl2idx.items()}
df = pd.DataFrame({'Real': [idx2lbl[e] for e in labels[idx_sample].tolist()],
                   'Pred': [idx2lbl[e] for e in output[idx_sample].argmax(1).tolist()]})

df


