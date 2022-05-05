from torch.nn.modules.module import Module
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch
import math
import torch.optim as optim
import torch.nn.functional as F

class GCN_layer(nn.Module):
    def __init__(self, in_features, out_features, A):
        super(GCN_layer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        #self.A = A
        self.fc = nn.Linear(in_features, out_features)
        
    def forward(self, input, adj):
        support = torch.mm(input, self.weight) #행렬 곱, input 데이터(feature)와 adj를 곱함
        print(support.size)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output
        #return self.fc(torch.spmm(self.A, X)) #이웃 정보 종합

class GCN(nn.Module):
    def __init__(self, num_feature, num_class, A):
        super(GCN, self).__init__()

        self.feature_extractor = nn.Sequential(
                                    GCN_layer(num_feature, 16, A),
                                    nn.ReLU(),
                                    GCN_layer(16, num_class, A)
                                )
        
    def forward(self, X):
        return self.feature_extractor(X)