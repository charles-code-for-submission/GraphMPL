import torch
import torch.nn as nn
from dgi.layers import GCN, AvgReadout, Discriminator
import torch.nn.functional as F


class DGI(nn.Module):
    def __init__(self, n_in, n_h1, n_h2, n_way, activation):
        super(DGI, self).__init__()
        self.gcn1 = GCN(n_in, n_h1, activation)
        self.gcn2 = GCN(n_h1, n_h2, activation)
        # self.gcn_l2 = GCN(n_h, n_h, activation)
        self.vars = nn.ParameterList()
        self.read = AvgReadout()
        self.outdim = n_h2
        self.n_way = n_way
        # todo 去除sigmoid
        # self.sigm = nn.Sigmoid()
        self.layer = 2
        # self.disc = Discriminator(n_h)

        # net = nn.ModuleList()
        # net.append(encoder)
        self.cls = nn.Linear(n_h2, self.n_way, bias=True)
        #self.cls1 = nn.Linear(n_h2, n_h2, bias=True)
        #self.cls2 = nn.Linear(n_h2, self.n_way, bias=True)
    # def forward(self, seq1, seq2, adj, sparse, msk, samp_bias1, samp_bias2):
    #     h_l1 = self.gcn_l1(seq1, adj, sparse)
    #     h_l2 = self.gcn_l2(h_l1, adj, sparse)
    #
    #     c = self.read(h_l2, msk)
    #     c = self.sigm(c)
    #
    #     h2_l1 = self.gcn_l1(seq2, adj, sparse)
    #     h2_l2 = self.gcn_l2(h2_l1, adj, sparse)
    #
    #     ret = self.disc(c, h_l2, h2_l2, samp_bias1, samp_bias2)
    #     return ret
    def getdim(self):
        return self.outdim
    '''
    def forward(self, seq1, seq2, adj, sparse, msk, samp_bias1, samp_bias2):
        h_l1 = self.gcn_l1(seq1, adj, sparse)
        h_l1 = self.sigm(h_l1)
        h_l2 = self.gcn_l2(h_l1, adj, sparse)

        c = self.read(h_l2, msk)
        # c = self.sigm(c)

        h2_l1 = self.gcn_l1(seq2, adj, sparse)
        h2_l2 = self.gcn_l2(h2_l1, adj, sparse)

        ret = self.disc(c, h_l2, h2_l2, samp_bias1, samp_bias2)

        return ret
    '''
    # features, x_qry[i], adj, self.net.parameters())

    def forward(self, features,  seq1, adj, vars, sparse=True, bias=True):
        # print([i for i in vars])
        if vars is None:
            vars = [j for i, j in self.named_parameters()]
        else:
            vars = [i for i in vars]
            # print(vars[0])

        # print('vars', vars)
        # print('var len: ', len(vars), [i.shape for i in vars], seq1)
        # bias, weight, activation
        # print(vars[1].shape, seq1.shape)
        # print(vars[1].T.shape)
        ind = 0
        for i in range(self.layer):
            if i == 0:
                h1 = torch.matmul(features, vars[ind+1].T)
            else:
                h1 = torch.matmul(h1, vars[ind+1].T)
            if sparse:
                h1 = torch.spmm(adj, torch.squeeze(h1, 0))
            else:
                h1 = torch.bmm(adj, h1)

            if bias is True:
                # ind += 1
                h1 = h1 + vars[ind]

            h1 = F.prelu(h1, vars[ind+2])
            ind += 3
            # print('layer', i)
        output = h1[seq1]
        # print(h1.shape)

        logits = torch.matmul(output, vars[-2].T) + vars[-1]
        # print('pin2', logits)
        return logits

    # Detach the return variables
    def embed(self, seq, adj, sparse, msk):
        h_l1 = self.gcn1(seq, adj, sparse)
        h_l2 = self.gcn2(h_l1, adj, sparse)
        c = self.read(h_l2, msk)

        return h_l2.detach(), c.detach()

