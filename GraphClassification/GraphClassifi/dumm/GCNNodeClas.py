import util2
import json
import util as ut
import numpy as np
import torch
import math
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn
import torch.nn.functional as F
import time
import torch.optim as optim
from tqdm import tnrange
import pandas as pd

# 각 그래프를 node 취급하면 될 듯..? 인접 행렬 틀(이미지 규격, rgb)이 동일한데,
# np.array()로 이미지, Label 하나씩 학습시키는 걸 볼 수 있었음
# gnn도 이케 하면 graph classification 아님? 일단 한다...


'''
    node classification과 동일하게 사용
    adj : 이미지 간 동일 클러스터 여부
    feature : freObj x freObj 간의 관계가 반영된 matrix를 flatten으로 펼침. 1000x10000
    + freObj x freObj 펼치기 전에 fasttext embedding 값을 곱하면 더 좋을 것 같기두
    Y : 각 id 별 cluster 값
    
    freObj5000, AdjSet5000, cluster5000
'''


testFile = open('../data/freObj5000.txt', 'r')  # 'r' read의 약자, 'rb' read binary 약자 (그림같은 이미지 파일 읽을때)
readFile = testFile.readline()
freObj = (readFile[1:-1].replace("'", '').replace(' ', '')).split(',')
freObj = freObj[:100]
adjList = []

with open('./data/scene_graphs.json') as file:
    data1 = json.load(file)
with open('./data/objects.json') as file:  # open json file
    data2 = json.load(file)
for i in range(1000):
    adj = ut.createAdj_model2(i, freObj,data1, data2)
    adjList.append(adj)
#adjList = np.ndarray(adjList)


# gpu 사용
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#features = csr_matrix(np.load('./data/idFreFeature.npy'), dtype=np.float32)
features = util2.objNameEmbedding(freObj)

adj = torch.FloatTensor(np.load('./data/idAdj.npy'))
#features = csr_matrix(features)

# def normalize(mx):
#     rowsum = np.array(mx.sum(1))
#     r_inv = (rowsum ** -1).flatten()
#     r_inv[np.isinf(r_inv)] = 0.
#     r_mat_inv = diags(r_inv)
#     mx = r_mat_inv.dot(mx)
#     return mx
#features = normalize(features)
#print("features after nomalize : ", features)


testFile = open('../data/cluster5000.txt', 'r')  # 'r' read의 약자, 'rb' read binary 약자 (그림같은 이미지 파일 읽을때)
readFile = testFile.readline()
label = (readFile[1:].replace("'", '').replace(' ', '').split(','))
labels = []
for i in range(1000):
    labels.append(int(label[i]))

features = torch.FloatTensor(features)  # <class 'torch.Tensor'>
print(features)
labels = torch.LongTensor(labels)  # <class 'torch.Tensor'>

# dataset train/test/val로 나눔
np.random.seed(34)
n_train = 200
n_val = 300
n_test = len(features) - n_train - n_val
idxs = np.random.permutation(len(features))
idx_train = torch.LongTensor(idxs[:n_train])
idx_val = torch.LongTensor(idxs[n_train:n_train + n_val])
idx_test = torch.LongTensor(idxs[n_train + n_val:])
#adj = torch.FloatTensor(adjList)

# cuda.. gpu로 보냄
adj = torch.FloatTensor(adjList).to(device)
features = features.to(device)
labels = labels.to(device)
idx_train = idx_train.to(device)
idx_val = idx_val.to(device)
idx_test = idx_test.to(device)


# Model
class GraphConvolution(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # weight를 reset해 줌. weight 변화에 따른 정확도 측정을 위해서? 왜하지? 이유 찾아보기
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
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
        #

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


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
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
model = GCN(nfeat=n_features,
            nhid=20,  # hidden = 16
            nclass=n_labels,
            dropout=0.5)  # dropout = 0.5
optimizer = optim.Adam(model.parameters(),
                       lr=0.001, weight_decay=5e-4)


def step():
    t = time.time()
    model.train()  # model 학습모드로
    optimizer.zero_grad()
    output = model(features, adj)  # model에 값 넣음
    loss = F.nll_loss(output[idx_train], labels[idx_train])  # loss 함수
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
def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)  #비슷하다고 뽑은 것들중에 제일 비슷한 거... label 예측한 거 전체list
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def alone(output, labels) :
    preds = output.max(1)[1].type_as(labels)
    print(preds)
    return preds


# seed 고정
torch.manual_seed(34)

# model
 # hidden = 16, dropout = 0.5
model = GCN(nfeat=n_features,
            nhid=20, nclass=n_labels, dropout=0.5)
optimizer = optim.Adam(model.parameters(),
                       lr=0.001, weight_decay=5e-4)
model = model.cuda()
epochs = 100
# epochs = 1000
# print_steps = 100
print_steps = 10
train_loss, train_acc = [], []
val_loss, val_acc = [], []


#여기서 행렬 하나씩 넣어주면 됨
for i in tnrange(epochs):
    for a in adj :  #self
        tl, ta = step(a)
        train_loss += [tl]
        train_acc += [ta]

        if ((i + 1) % print_steps) == 0 or i == 0:
            tl, ta = evaluate(idx_train,a)
            vl, va = evaluate(idx_val,a)
            val_loss += [vl]
            val_acc += [va]

            print('Epochs: {}, Train Loss: {:.3f}, Train Acc: {:.3f}, Validation Loss: {:.3f}, Validation Acc: {:.3f}'.format(i, tl, ta, vl, va))

output = model(features, adj[0])

# test - 범위 지정해서 Y or No 값 확인하기.. 이거 장난질 아닌가 하는 그런..마음.. 원래 의도랑 달라보임..

idx2lbl = ['0번 그림', '1번 그림', '2번 그림', '3번 그림', '4번 그림', '5번 그림', '6번 그림', '7번 그림'
    , '8번 그림', '9번 그림', '10번 그림', '11번 그림', '12번 그림', '13번 그림', '14번 그림']

#



#각 범위 별 이미지 값으로 동일 여부 확인
#범위 추가하는 변수2개 만들면 될 듯 -> list면 range 쓰겠는데 tensor는 안된다함.
#예제기도 하고, 굳이 쓰려면 a1,b1, a2,b2를 사용하면 됨


a1 = 1
b1 = 10
a2 = 22
b2 = 31

idx_sample1 = idx_test[torch.randperm(len(idx_test))[1:10]]
realList1 = [idx2lbl[e] for e in labels[idx_sample1].tolist()]
predList1 = [idx2lbl[e] for e in output[idx_sample1].argmax(1).tolist()]

idx_sample2 = idx_test[torch.randperm(len(idx_test))[22:31]]
realList2 = [idx2lbl[e] for e in labels[idx_sample2].tolist()]
predList2 = [idx2lbl[e] for e in output[idx_sample2].argmax(1).tolist()]

resPred = []
for i in range(len(predList2)) :
    if(predList1[i] == predList2[i]) :
        resPred.append('T')
    else :
        resPred.append('F')

resReal = []
for i in range(len(realList1)) :
    if(realList1[i] == realList2[i]) :
        resReal.append('T')
    else :
        resReal.append('F')

#img_id 값으로 그림 확인하고 싶어서 id 찍어봄
print("idx_sample1_Imgid : ",idx_sample1)
print("idx_sample2_Imgid : ", idx_sample2)
idDF1 = pd.DataFrame({'idx_sample1_Imgid': idx_sample1,
                   'idx_sample2_Imgid': idx_sample2})
print(idDF1)


df1 = pd.DataFrame({'Pre1': [idx2lbl[e] for e in output[idx_sample1].argmax(1).tolist()],
                   'Pred2': [idx2lbl[e] for e in output[idx_sample2].argmax(1).tolist()],
                   'Res(T/F)':resPred})
df2 = pd.DataFrame({'Real1':  [idx2lbl[e] for e in labels[idx_sample1].tolist()],
                   'Real2':  [idx2lbl[e] for e in labels[idx_sample2].tolist()],
                   'Res(T/F)':resReal})

print(df1)
print(df2)

