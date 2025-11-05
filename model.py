import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import (cosine_sim,
                   view_distribution,
                   node_distribution,
                   kl_div_matrix,
                   scale_matrix,
                   pseudo_graph)



class Encoder(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.0, activate = 'leakyrelu'):
        super(Encoder, self).__init__()
        self.in_feature = in_dim
        self.out_feature = out_dim
        self.layer = nn.Linear(self.in_feature, self.out_feature)
        self.act = activate
        self.dropout = dropout

    def forward(self, x):
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.layer(x)
        if self.act == 'leakyrelu':
            x = F.leaky_relu(x, inplace=True)

        return x
    

class Model(nn.Module):
    def __init__(self, in_dim, out_dim, n_clusters=5, dropout=0.0, device = torch.device('cuda')):
        super(Model, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_clusters = n_clusters
        self.dropout = dropout
        self.act = 'leakyrelu'
        self.device = device
        self.emb_sim_exist = False

        self.encoder1 = Encoder(self.in_dim, self.out_dim, self.dropout, self.act)
        self.encoder2 = Encoder(self.in_dim, self.out_dim, self.dropout, self.act)


    def forward(self, x_1, x_2):

        x_1 = self.encoder1(x_1)
        x_2 = self.encoder2(x_2)

        emb_1 = F.normalize(x_1, dim=1, p=2)
        emb_2 = F.normalize(x_2, dim=1, p=2)
        self.emb_sim_exist = False

        return emb_1, emb_2


    def compute_emb_sim(self, emb1, emb2):
        self.emb_sim_matrix = cosine_sim(emb1, emb2, device=self.device)
        self.emb_sim_exist = True


    def compute_eta(self, adj):
        T_matrix = self.emb_sim_matrix
        n_sample = T_matrix.size()[0]

        T_max, _ = torch.max(T_matrix, dim=-1, keepdim=True)
        T_min, _ = torch.min(T_matrix, dim=-1, keepdim=True)
        T_matrix = (T_matrix - T_min) / (T_max - T_min)
        # xi = torch.trace(T_matrix) / n_sample - 0.1
        xi = torch.trace(T_matrix) / n_sample
        T_matrix = torch.where(adj == 1, T_matrix, 0)

        mask = (T_matrix >= xi).int()
        eta = torch.mean(torch.div(torch.sum(mask, dim=-1), torch.sum(adj, dim=-1).clamp(min=1)))

        return eta


    def high_confidence_adj(self, label_pred, dis, adj, k=0.1):
        dis = torch.min(dis, dim=-1).values

        # indices of non-high-confidence label
        values, indices = torch.topk(dis, int(len(dis) * (1 - k)), largest=True)

        pseudo_adj = pseudo_graph(label_pred, self.device)

        # delete connections of nodes with non-high-confidence label
        pseudo_adj[:, indices] = 0
        pseudo_adj[indices, :] = 0

        sim_matrix = self.emb_sim_matrix
        s_max, _ = torch.max(sim_matrix, dim=-1, keepdim=True)
        s_min, _ = torch.min(sim_matrix, dim=-1, keepdim=True)
        sim_matrix = (sim_matrix - s_min) / (s_max - s_min)

        sim_matrix = torch.where(adj > 0, sim_matrix, 0)
        pseudo_adj = pseudo_adj + sim_matrix

        pseudo_adj = torch.clamp(pseudo_adj, max=1)

        return pseudo_adj


    def AFC_loss(self, x_1, x_2, H):
        sim_inter_view = scale_matrix(self.emb_sim_matrix)

        pos_sim = torch.diag(sim_inter_view)
        # adj without self loops here.
        loss_afc = (pos_sim + torch.sum(sim_inter_view * H, dim=-1)) / (torch.sum(sim_inter_view, dim=-1))
        loss_afc = - torch.log(loss_afc).mean()

        return loss_afc


    def GDA_loss(self, x_1, x_2):

        p_x_1_view = view_distribution(x_1)
        p_x_2_view = view_distribution(x_2)

        kl_view = torch.sum(p_x_1_view * (torch.log(p_x_1_view) - torch.log(p_x_2_view)))
        kl_view += torch.sum(p_x_2_view * (torch.log(p_x_2_view) - torch.log(p_x_1_view)))

        p_x_1_node = node_distribution(x_1)
        p_x_2_node = node_distribution(x_2)

        kl_node = torch.sum(p_x_1_node * (torch.log(p_x_1_node) - torch.log(p_x_2_node))) / p_x_1_node.size(0)
        kl_node += torch.sum(p_x_2_node * (torch.log(p_x_2_node) - torch.log(p_x_1_node))) / p_x_2_node.size(0)
        kl_loss = kl_view + kl_node / 2

        return kl_loss


    def NCA_loss(self, x_1, x_2, adj, eta = 1.0):
        # Neighborhood distillation using KL divergence for each node.
        dist_matrix = kl_div_matrix(x_1, x_2, device=self.device)
        dist_matrix = (dist_matrix + kl_div_matrix(x_2, x_1, device=self.device).t())

        # adj without self loops here.
        n_samples = adj.size(0)
        dist_neighnors = dist_matrix * (adj * eta + torch.eye(n_samples, device=self.device))

        sum_dist_neighnors = torch.sum(dist_neighnors, dim=-1) / (torch.sum(adj, dim=-1) + 1)
        sum_sample_dist = torch.sum(dist_matrix * (1 - torch.eye(n_samples, device=self.device)), dim=-1) / (n_samples - 1)
        loss_nca = torch.mean(sum_dist_neighnors / sum_sample_dist)

        return loss_nca


