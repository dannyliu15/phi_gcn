import torch.nn as nn
import torch.nn.functional as F
from pygcn.layers import GraphConvolution, GraphConvolution_lightGCN


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass=2, dropout=False):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)

class lightGCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass=2, dropout=False):
        super(lightGCN, self).__init__()
        self.gc1 = GraphConvolution_lightGCN()
        self.gc2 = GraphConvolution_lightGCN()
        self.fc  = nn.Linear(nfeat, nclass)

    def forward(self, x, adj):
        x = self.gc1(x, adj)
        x = self.gc2(x, adj)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

