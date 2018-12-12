import torch
from torch import nn
import torch.nn.functional as F

from layers import MultiEdgeGraphConvolution, InnerProductDecoder, Graphite, GraphConvolution, \
    MyGraphite, MyMultiGraphite, MyDirectNodeDecoder


# class GCNModelVAE(nn.Module):
#     def __init__(self, num_features, hidden1, hidden2, dropout):
#         super(GCNModelVAE, self).__init__()
#         self.gcn1 = GraphConvolution(num_features, hidden1, relu=False)
#         self.gcn2 = GraphConvolution(hidden1, hidden1)
#         self.z_mean = GraphConvolution(hidden1, hidden2, dropout=dropout, relu=False)
#         self.z_log_std = GraphConvolution(hidden1, hidden2, dropout=dropout, relu=False)
#
#         self.hidden1 = hidden1
#         self.hidden2 = hidden2
#
#     def encode(self, features, adjs):
#         # print(features)
#         # print(adjs)
#         x = self.gcn1(features, adjs)
#         # print(self.gcn1.linear.weight)
#         z_mean = self.z_mean(x, adjs)
#         z_log_std = self.z_log_std(x, adjs)
#         return z_mean + torch.randn(x.shape[0], self.hidden2) * torch.exp(z_log_std)
#
#     def get_z(self, features, adjs):
#         x = self.gcn1(features, adjs)
#         z_mean = self.z_mean(x, adjs)
#         z_log_std = self.z_log_std(x, adjs)
#         return z_mean, z_log_std
#
#     def decode(self, features, z):
#         raise NotImplementedError
#
#     def forward(self, features, adj):
#         return self.decode(features, self.encode(features, adj))


class MultiGCNModelVAE(nn.Module):
    def __init__(self, modes, num_features, encode_features, gcn_batch_norm, gcn_hiddens, gcn_aggs, gcn_relus, z_dim,
                 z_agg, dropout):
        super(MultiGCNModelVAE, self).__init__()
        self.gcns = []
        prev_hidden = num_features
        for hidden, agg, relu in zip(gcn_hiddens, gcn_aggs, gcn_relus):
            self.gcns.append(MultiEdgeGraphConvolution(modes, agg, prev_hidden, hidden, relu=relu,
                                                       batch_norm=gcn_batch_norm))
            prev_hidden = hidden
        self.z_mean = MultiEdgeGraphConvolution(modes, z_agg, prev_hidden, z_dim, dropout=dropout, relu=False)
        self.z_log_std = MultiEdgeGraphConvolution(modes, z_agg, prev_hidden, z_dim, dropout=dropout, relu=False)

        self.z_dim = z_dim
        self.encode_features = encode_features
        self.num_features = num_features

        for i, gcn in enumerate(self.gcns):
            self.add_module('gcn{}'.format(i), gcn)

    def encode(self, features, adjs):
        z_mean, z_log_std = self.get_z(features, adjs)
        return z_mean + torch.randn(z_mean.shape[0], self.z_dim) * torch.exp(z_log_std)

    def get_z(self, features, adjs):
        if self.encode_features:
            x = features
        else:
            x = torch.cat([torch.eye(features.shape[0]), torch.zeros(features.shape[0], self.num_features -
                                                                     features.shape[0])], dim=1)
        for gcn in self.gcns:
            x = gcn(x, adjs)
        z_mean = self.z_mean(x, adjs)
        z_log_std = self.z_log_std(x, adjs)
        return z_mean, z_log_std

    def decode(self, features, z):
        raise NotImplementedError

    def forward(self, features, adj):
        return self.decode(features, self.encode(features, adj))


# class GCNModelFeedback(GCNModelVAE):
#     def __init__(self, num_features, hidden1, hidden2, hidden3, dropout, autoregressive):
#         super(GCNModelFeedback, self).__init__(num_features, hidden1, hidden2, dropout)
#         self.graphite1 = Graphite(num_features, hidden3)
#         self.graphite2 = Graphite(hidden2, hidden3)
#         self.graphite3 = Graphite(hidden3, hidden2, relu=False)
#         self.inner = InnerProductDecoder()
#         self.linear1 = nn.Linear(hidden2, hidden3)
#         self.linear2 = nn.Linear(hidden1, 100)
#
#         self.linear_inner = nn.Linear(100, 100)
#         self.g = MyGraphite(num_features, hidden2, hidden3)
#
#         self.autoregressive = autoregressive
#
#     def decode(self, features, z):
#         # return self.linear2(F.relu(self.linear1(z))).reshape(-1)
#         recon_1 = F.normalize(z, p=2, dim=1)
#         recon_2 = torch.ones(*recon_1.shape)
#         recon_2 = recon_2 / torch.sqrt(recon_2.sum(dim=1, keepdim=True))
#         d = recon_1.mm(recon_1.sum(dim=0).unsqueeze(1)) + recon_2.mm(recon_2.sum(dim=0).unsqueeze(1))
#         d = d.pow(-0.5)
#         recon_1 = recon_1 * d
#         recon_2 = recon_2 * d
#         r = self.inner(recon_1 + recon_2)
#         recons = self.inner(self.g(r, features, z))
#
#         # recons = self.inner(recon_1 + recon_2)
#         # update = self.graphite2(z, recon_1, recon_2) + self.graphite1(features, recon_1, recon_2)
#         # update = self.graphite3(update, recon_1, recon_2)
#         # update = (1 - self.autoregressive) * z + self.autoregressive * update
#
#         # seems to work, but only on fixed dims!
#         # recons = self.linear_inner(recon_1 + recon_2)
#         # recons = self.linear_inner(update)
#         # recons = self.inner(update)
#         return recons.reshape(-1)
#
#     def sample(self, features, n_samples):
#         z = torch.randn(n_samples, self.hidden2)
#         return [torch.sigmoid(recon.reshape(n_samples, n_samples)) for recon in self.decode(features, z)]


class MultiGCNModelFeedback(MultiGCNModelVAE):
    def __init__(self, modes, num_features, encode_features, gcn_batch_norm, gcn_hiddens, gcn_aggs, gcn_relus, z_dim, \
                 z_agg, graphite_relu, graphite_layers, dropout, autoregressive):
        super(MultiGCNModelFeedback, self).__init__(modes, num_features, encode_features, gcn_batch_norm,
                                                    gcn_hiddens, gcn_aggs,
                                                    gcn_relus, z_dim,
                                                    z_agg, dropout)

        self.graphites = []
        for i in range(graphite_layers):
            self.graphites.append(MyMultiGraphite(modes, num_features, z_dim, z_dim,
                                                  relu=i != graphite_layers-1 and graphite_relu))
        self.inner = InnerProductDecoder()

        for i, graphite in enumerate(self.graphites):
            self.add_module('graphite{}'.format(i), graphite)

        self.autoregressive = autoregressive
        self.modes = modes

    def decode(self, features, z):
        # recon_1 = F.normalize(z, p=2, dim=1)
        # recon_2 = torch.ones(*recon_1.shape)
        # recon_2 = recon_2 / torch.sqrt(recon_2.sum(dim=1, keepdim=True))
        # d = recon_1.mm(recon_1.sum(dim=0).unsqueeze(1)) + recon_2.mm(recon_2.sum(dim=0).unsqueeze(1))
        # d = d.pow(-0.5)
        # recon_1 = recon_1 * d
        # recon_2 = recon_2 * d
        # update = recon_1 + recon_2
        update = z
        for graphite in self.graphites:
            update = graphite(features, update)
        update = (1 - self.autoregressive) * z + self.autoregressive * update
        recons = self.inner(update)
        # update = self.graphite2(z, recon_1, recon_2) + self.graphite1(features, recon_1, recon_2)
        # update = self.graphite3(update, recon_1, recon_2)
        # recons = self.inner(update)
        return recons

    def sample(self, features, n_samples):
        z = torch.randn(n_samples, self.z_dim)
        probs = self.decode(features, z)
        return probs.max(dim=0)[1], probs


class MultiGCNModelNodeDirect(MultiGCNModelVAE):
    def __init__(self, modes, num_features, hidden1, hidden2, hidden3, dropout, autoregressive):
        super(MultiGCNModelNodeDirect, self).__init__(modes, num_features, hidden1, hidden2, dropout)
        self.d = MyDirectNodeDecoder(modes, hidden2, hidden3, 9)
        self.hidden2 = hidden2
        self.modes = modes

    def decode(self, features, z):
        return self.d(z).reshape(-1)

    # TODO: use features in z

    def sample(self, features, n_samples):
        z = torch.randn(n_samples, self.hidden2)
        return torch.sigmoid(self.decode(features, z).reshape(self.modes, n_samples, n_samples))
