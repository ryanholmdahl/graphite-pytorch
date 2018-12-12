import torch
from torch import nn
import functools


class Graphite(nn.Module):
    def __init__(self, input_dim, output_dim, relu=True):
        super(Graphite, self).__init__()
        # self.linear = nn.Linear(input_dim, output_dim)
        # nn.init.xavier_uniform_(self.linear.weight)
        self.weight = nn.Parameter(torch.zeros(input_dim, output_dim))
        nn.init.xavier_uniform_(self.weight)
        if relu:
            self.relu = nn.ReLU()
        else:
            self.relu = None

    def forward(self, features, recon_1, recon_2):
        x = features.matmul(self.weight)
        x = recon_1.mm(recon_1.t().mm(x)) + recon_2.mm(recon_2.t().mm(x))
        return self.relu(x) if self.relu is not None else x


class MyGraphite(nn.Module):
    def __init__(self, num_features, z_dim, output_dim):
        super(MyGraphite, self).__init__()
        self.weight_f = nn.Parameter(torch.zeros(num_features, output_dim))
        nn.init.xavier_uniform_(self.weight_f)
        self.weight_z = nn.Parameter(torch.zeros(z_dim, output_dim))
        nn.init.xavier_uniform_(self.weight_z)

    def forward(self, adj_pred, features, z):
        # adj_pred: n_samples, n_samples
        # x: n_samples, hidden2 + n_features
        # adj_pred.mm(x): n_samples, hidden2 + n_features
        return adj_pred.mm(features).mm(self.weight_f) + adj_pred.mm(z).mm(self.weight_z)


class MyDirectNodeDecoder(nn.Module):
    def __init__(self, modes, z_dim, hidden_dim, output_dim):
        super(MyDirectNodeDecoder, self).__init__()
        self.linear1 = nn.Linear(z_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, modes * output_dim)
        self.modes = modes

    def forward(self, z):
        x = self.linear2(self.relu(self.linear1(z)))
        x = x.reshape(self.modes, z.shape[0], -1)
        return x[:, :, :z.shape[0]]


class MyMultiGraphite(nn.Module):
    def __init__(self, modes, num_features, z_dim, output_dim, relu=False):
        super(MyMultiGraphite, self).__init__()
        self.weight_f = nn.Parameter(torch.zeros(modes, num_features, output_dim))
        nn.init.xavier_uniform_(self.weight_f)
        self.weight_z = nn.Parameter(torch.zeros(modes, z_dim, output_dim))
        nn.init.xavier_uniform_(self.weight_z)
        if relu:
            self.relu = nn.ReLU()
        else:
            self.relu = None
        self.inner = InnerProductDecoder()

    # def forward(self, adj_pred, features, z):
    def forward(self, features, z):
        # adj_pred: n_samples, n_samples
        # x: n_samples, hidden2 + n_features
        # adj_pred.mm(x): n_samples, hidden2 + n_features
        adj_pred = self.inner(z) / z.pow(2).sum(dim=1) + 1.
        x = adj_pred.matmul(features).matmul(self.weight_f) + adj_pred.matmul(z).matmul(self.weight_z)
        return self.relu(x) if self.relu is not None else x


class MultiEdgeGraphite(nn.Module):
    def __init__(self, modes, input_dim, output_dim, mean=True, relu=True):
        super(MultiEdgeGraphite, self).__init__()
        self.weight = nn.Parameter(torch.zeros(modes, input_dim, output_dim))
        nn.init.xavier_uniform_(self.weight)
        # self.b = nn.Parameter(torch.zeros(modes, 1, output_dim))

        # self.linear = nn.Linear(input_dim, output_dim)
        # nn.init.xavier_uniform_(self.linear.weight)
        if relu:
            self.relu = nn.ReLU()
        else:
            self.relu = None
        self.mean = mean

    def forward(self, features, recon_1, recon_2):
        x = features.matmul(self.weight)
        # x = self.linear(features)
        x = recon_1.matmul(recon_1.transpose(0, 1).matmul(x)) + recon_2.matmul(recon_2.transpose(0, 1).matmul(x))
        x = self.relu(x) if self.relu is not None else x
        if self.mean:
            return x.mean(dim=0)
        else:
            return x


class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0., relu=True):
        super(GraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.zeros(input_dim, output_dim))
        nn.init.xavier_uniform_(self.weight)
        # self.linear = nn.Linear(input_dim, output_dim)
        # nn.init.xavier_uniform_(self.linear.weight)
        self.dropout = nn.Dropout(dropout)
        if relu:
            self.relu = nn.ReLU()
        else:
            self.relu = None

    def forward(self, features, adj):
        # x = self.dropout(features)
        # x = self.linear(x)
        x = adj.mm(features).mm(self.weight)
        return self.relu(x) if self.relu is not None else x


class MultiEdgeGraphConvolution(nn.Module):
    def __init__(self, modes, agg, input_dim, output_dim, dropout=0., relu=True, batch_norm=False):
        super(MultiEdgeGraphConvolution, self).__init__()
        # self.gcs = [GraphConvolution(input_dim, output_dim, dropout, relu) for _ in range(modes)]
        self.agg = agg
        # self.linear = nn.Linear(input_dim, output_dim)
        self.weight = nn.Parameter(torch.zeros(modes, input_dim, output_dim))
        nn.init.xavier_uniform_(self.weight)
        self.dropout = nn.Dropout(dropout)
        if relu:
            self.relu = nn.ReLU()
        else:
            self.relu = None
        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(output_dim)
        else:
            self.batch_norm = None

    def forward(self, features, adjs):
        x = self.dropout(features)
        # x = x.matmul(self.weight)
        x = adjs.matmul(x).matmul(self.weight)
        x = self.batch_norm(x.transpose(1, 2)).transpose(1, 2) if self.batch_norm is not None else x
        x = self.relu(x) if self.relu is not None else x
        # x[x != x] = 0
        if self.agg == 'mean':
            return x.mean(dim=0)
        if self.agg == 'max':
            return x.max(dim=0)[0]


class InnerProductDecoder(nn.Module):
    def __init__(self):
        super(InnerProductDecoder, self).__init__()

    def forward(self, x):
        return x.matmul(x.transpose(len(x.shape) - 2, len(x.shape) - 1))


class Scale(nn.Module):
    def __init__(self):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(data=torch.zeros(1))

    def forward(self, x, y):
        return x * (1 - self.sigmoid(self.scale)) + y * self.sigmoid(self.scale)
