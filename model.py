import numpy as np
# import ot
# import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from utils import (get_fusion,
                   min_max_normlize,
                   cosine_distance,
                   cosine_sim,
                   euclidean_distance,
                   kl_div_matrix,
                   target_distribution,
                   soft_assignment,
                   pseudo_graph,
                   high_conf_graph)


# class GCN_net(nn.Module):
#     def __init__(self, in_dim, hid_dim, out_dim, dropout=0.0, activate = 'relu'):
#         super(GCN_net, self).__init__()
#         # self.gcn1 = GCNConv(in_dim, hid_dim)
#         # self.gcn2 = GCNConv(hid_dim, hid_dim)
#         # self.linear = nn.Linear(hid_dim, out_dim)
#         self.linear = nn.Linear(in_dim, out_dim)
#         self.act = activate
#         self.dropout = dropout
#
#     def forward(self, x):
#         # # x = F.dropout(x, self.dropout, training=self.training)
#         # x = self.gcn1(x, edge_index)
#         # if self.act == 'leakyrelu':
#         #     x = F.relu(x, inplace=True)
#         #
#         # # x = F.dropout(x, self.dropout, training=self.training)
#         # x = self.gcn2(x, edge_index)
#         # if self.act == 'leakyrelu':
#         #     x = F.relu(x, inplace=True)
#
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = self.linear(x)
#         # x = F.normalize(x, dim=1, p=2)
#         if self.act == 'leakyrelu':
#             x = F.leaky_relu(x, inplace=True)
#         # x = F.normalize(x, dim=1, p=2)
#
#         return x

class GCN_net(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.0, activate = 'leakyrelu'):
        super(GCN_net, self).__init__()
        self.gcn1 = GCNConv(in_dim, out_dim)
        self.act = activate
        self.dropout = dropout

    def forward(self, x, edge_index):
        # x = F.dropout(x, self.dropout, training=self.training)
        x = self.gcn1(x, edge_index)
        if self.act == 'leakyrelu':
            x = F.relu(x, inplace=True)

        return x

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
    def __init__(self, in_dim, hid_dim, n_clusters=5, dropout=0.0, device = torch.device('cuda')):
        super(Model, self).__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.n_clusters = n_clusters
        self.dropout = dropout
        self.act = 'leakyrelu'
        self.device = device
        self.emb_sim_exist = False

        self.encoder1 = nn.Linear(self.in_dim, self.hid_dim)
        # self.encoder2 = nn.Linear(self.in_dim, self.hid_dim)

        # self.encoder1 = Encoder(self.in_dim, self.hid_dim, self.dropout, self.act)
        self.encoder2 = Encoder(self.in_dim, self.hid_dim, self.dropout, self.act)

        # self.encoder1 = GCN_net(self.in_dim, self.hid_dim, self.dropout, self.act)
        # self.encoder2 = GCN_net(self.in_dim, self.hid_dim, self.dropout, self.act)

        # self.cluster_head = nn.Linear(self.hid_dim, self.out_dim)

        # if self.param_shared:
        #     self.predictor = nn.Linear(self.hid_dim, self.out_dim)
        # else:
        #     self.predictor1 = nn.Linear(self.hid_dim, self.out_dim)
        #     self.predictor2 = nn.Linear(self.hid_dim, self.out_dim)
        #
        # self.cluster_layer = nn.Parameter(torch.Tensor(self.n_clusters, self.hid_dim), requires_grad=True)
        # torch.nn.init.xavier_normal_(self.cluster_layer.data)
        # self.alpha = nn.Parameter(torch.Tensor(1, ))
        # self.alpha.data = torch.tensor(0.99999).to(self.device)

    def forward(self, x_1, x_2):
        # x_1 = self.encoder1(x_1)
        x_1 = F.dropout(x_1, self.dropout, training=self.training)
        x_1 = self.encoder1(x_1)
        if self.act == 'leakyrelu':
            x_1 = F.leaky_relu(x_1, inplace=True)

        # x_2 = F.dropout(x_2, self.dropout, training=self.training)
        # x_2 = self.encoder2(x_2)
        # if self.act == 'leakyrelu':
        #     x_2 = F.leaky_relu(x_2, inplace=True)

        # x_1 = self.encoder1(x_1, edge_index)
        x_2 = self.encoder2(x_2)

        # h_tch = F.dropout(h_tch, self.dropout, self.training)
        # h_tch = self.teacher_net2(h_tch)
        # if self.act == 'relu':
        #     h_tch = F.relu(h_tch)
        # h_tch = F.normalize(h_tch, dim=1, p=2)

        # emb_stu = x_s
        # emb_tch = x_t
        # x_s = F.normalize(x_s, dim=1, p=2)
        # x_t = F.normalize(x_t, dim=1, p=2)
        emb_1 = F.normalize(x_1, dim=1, p=2)
        emb_2 = F.normalize(x_2, dim=1, p=2)
        self.emb_sim_exist = False

        # x_1 = F.dropout(x_1, self.dropout, self.training)
        # x_2 = F.dropout(x_2, self.dropout, self.training)
        #
        # if self.param_shared:
        #     x_1 = self.predictor(x_1)
        #     x_2 = self.predictor(x_2)
        # else:
        #     x_1 = self.predictor1(x_1)
        #     x_2 = self.predictor2(x_2)
        # #
        # # # z_stu = x_s
        # # # z_tch = x_t
        # z_1 = F.normalize(x_1, dim=1, p=2)
        # z_2 = F.normalize(x_2, dim=1, p=2)
        #
        # return emb_1, emb_2, z_1, z_2
        return emb_1, emb_2


    def compute_emb_sim(self, emb1, emb2, device = torch.device('cuda')):
        self.emb_sim_matrix = cosine_sim(emb1, emb2, device=self.device)
        self.emb_sim_exist = True

    def compute_eta(self, z1, z2, adj, xi=0.5):

        # z_hat = torch.concat((z1, z2), dim=-1)
        # T_matrix = self.compute_cosine_sim(z_hat, z_hat)
        # T_matrix = self.sim_matrix

        # T_matrix = self.compute_cosine_sim(z1, z2)
        # T_matrix = self.sim_matrix

        # self.z_sim_matrix = cosine_sim(z1, z2, device=self.device)
        # T_matrix = self.z_sim_matrix
        T_matrix = self.emb_sim_matrix
        n_sample = T_matrix.size()[0]

        # T_matrix = torch.where(adj > 0, T_matrix, 0)
        T_max, _ = torch.max(T_matrix, dim=-1, keepdim=True)
        T_min, _ = torch.min(T_matrix, dim=-1, keepdim=True)
        T_matrix = (T_matrix - T_min) / (T_max - T_min)
        # print(f"T_matrix: {T_matrix.detach().cpu()}")
        xi = torch.trace(T_matrix) / n_sample - 0.1
        # xi = torch.min(torch.diag(T_matrix))
        T_matrix = torch.where(adj == 1, T_matrix, 0)

        mask = (T_matrix >= xi).int()
        eta = torch.mean(torch.div(torch.sum(mask, dim=-1), torch.sum(adj, dim=-1).clamp(min=1)))
        # eta = torch.sum(mask) / torch.sum(adj)

        return eta

    def compute_eta2(self, adj, xi=0.5):
        T_matrix = self.emb_sim_matrix

        T_max, _ = torch.max(T_matrix, dim=-1, keepdim=True)
        T_min, _ = torch.min(T_matrix, dim=-1, keepdim=True)
        T_matrix = (T_matrix - T_min) / (T_max - T_min)
        # xi = torch.trace(T_matrix) / n_sample - 0.1
        T_matrix = torch.where(adj == 1, T_matrix, 0)

        mask = (T_matrix >= xi).int()
        eta = torch.mean(torch.div(torch.sum(mask, dim=-1), torch.sum(adj, dim=-1).clamp(min=1)))

        return eta

    def high_confidence_adj(self, emb1, emb2, label_pred, dis, adj, k=0.1, device=torch.device('cuda:0')):
        dis = torch.min(dis, dim=-1).values
        # indices of non-high-confidence label
        values, indices = torch.topk(dis, int(len(dis) * (1 - k)), largest=True)

        pseudo_adj = pseudo_graph(label_pred, device)
        # delete connections of nodes with non-high-confidence label
        pseudo_adj[:, indices] = 0
        pseudo_adj[indices, :] = 0

        # self.emb_sim_matrix = cosine_sim(emb1, emb2, device=device)
        sim_matrix = self.emb_sim_matrix
        s_max, _ = torch.max(sim_matrix, dim=-1, keepdim=True)
        s_min, _ = torch.min(sim_matrix, dim=-1, keepdim=True)
        sim_matrix = (sim_matrix - s_min) / (s_max - s_min)

        # t = 1
        # sim_matrix = euclidean_distance(emb1, emb2, device)
        # sim_matrix = torch.exp(- sim_matrix / t)

        sim_matrix = torch.where(adj > 0, sim_matrix, 0)
        pseudo_adj = pseudo_adj + sim_matrix
        # pseudo_adj = pseudo_adj + T_matrix

        pseudo_adj = torch.clamp(pseudo_adj, max=1)

        return pseudo_adj

    def QD_graph(self, emb1, emb2, label_pred_list, dis_list, adj, k=0.1, device=torch.device('cuda:0')):
        pseudo_adj1 = pseudo_graph(label_pred_list[0], device)
        pseudo_adj1 = high_conf_graph(pseudo_adj1, dis_list[0], k)
        pseudo_adj2 = pseudo_graph(label_pred_list[1], device)
        pseudo_adj2 = high_conf_graph(pseudo_adj2, dis_list[1], k)
        pseudo_adj_fu = pseudo_graph(label_pred_list[2], device)
        pseudo_adj_fu = high_conf_graph(pseudo_adj_fu, dis_list[2], k)

        pseudo_adj = pseudo_adj1 + pseudo_adj2 + pseudo_adj_fu      # Decision 1, 2, 3.  without self loop

        self.emb_sim_matrix = cosine_sim(emb1, emb2, device=device)
        sim_matrix = self.emb_sim_matrix.clone().detach()

        sim_matrix = min_max_normlize(sim_matrix)                   # scale to range of 0 to 1
        pseudo_adj += sim_matrix                                    # Decision 4

        matrix1 = torch.where(pseudo_adj >= 1, 1, 0)
        matrix2 = torch.where(adj > 0, pseudo_adj/4, 0)
        pseudo_adj = matrix1 + matrix2
        pseudo_adj = torch.clamp(pseudo_adj, max=1)

        return pseudo_adj

    def MSE_loss(self, pseudo_adj, device = torch.device('cuda:0')):
        # H = pseudo_graph(label_pred, device = device)
        mse_loss = F.mse_loss(pseudo_adj, self.sim_matrix * pseudo_adj)

        return mse_loss


    def HSC_loss(self, x_1, x_2, TDCD_adj, tau = 0.1, device = torch.device('cuda:0')):

        # x_sim_matrix = cosine_sim(x_1, x_2, device=device)
        # x_sim_matrix = self.z_sim_matrix
        x_sim_matrix = self.emb_sim_matrix
        sim_inter_view = torch.exp(x_sim_matrix / tau)

        # pos_sim = sim_inter_view[range(n_samples), range(n_samples)]
        pos_sim = torch.diag(sim_inter_view)
        # neighbor_sim = sim_inter_view * adj         # adj without self loops here.
        # (P + nu * M) / (N + M)
        loss_hsc = (pos_sim + torch.sum(sim_inter_view * TDCD_adj, dim=-1)) / (torch.sum(sim_inter_view, dim=-1))
        # loss_hsc = (pos_sim) / (torch.sum(sim_inter_view, dim=-1))      # setting1
        # loss_hsc = (pos_sim + torch.sum(sim_inter_view * adj, dim=-1)) / (torch.sum(sim_inter_view, dim=-1))  # setting2

        loss_hsc = - torch.log(loss_hsc).mean()

        return loss_hsc

    def HSC_loss1(self, x_1, x_2, adj, eta=1.0, tau = 0.1, device = torch.device('cuda:0')):

        # x_sim_matrix = cosine_sim(x_1, x_2, device=device)
        # x_sim_matrix = self.z_sim_matrix
        x_sim_matrix = self.emb_sim_matrix
        sim_inter_view = torch.exp(x_sim_matrix / tau)

        # pos_sim = sim_inter_view[range(n_samples), range(n_samples)]
        pos_sim = torch.diag(sim_inter_view)
        # neighbor_sim = sim_inter_view * adj         # adj without self loops here.
        # (P + nu * M) / (N + M)
        loss_hsc = (pos_sim + torch.sum(sim_inter_view * adj * eta, dim=-1)) / (torch.sum(sim_inter_view, dim=-1))
        # loss_hsc = (pos_sim) / (torch.sum(sim_inter_view, dim=-1))      # setting1 xi = 0
        # loss_hsc = (pos_sim + torch.sum(sim_inter_view * adj, dim=-1)) / (torch.sum(sim_inter_view, dim=-1))  # setting2 xi = 1

        loss_hsc = - torch.log(loss_hsc).mean()

        return loss_hsc

    def neutral_hr_loss(self, x_1, x_2, adj, eta = 1.0, tao = 0.1, device = torch.device('cuda:0')):
        gamma = 1
        eta = eta ** gamma
        x_sim_matrix = cosine_sim(x_1, x_2, device=device)
        # x_sim_matrix = self.sim_matrix
        x_sim_matrix = torch.exp(x_sim_matrix / tao)

        # pos_sim = sim_inter_view[range(n_samples), range(n_samples)]
        pos_sim = torch.diag(x_sim_matrix)
        # neighbor_sim = sim_inter_view * adj         # adj without self loops here.
        # (P + nu * M) / (N + M)
        loss_hsc = (pos_sim + eta * torch.sum(x_sim_matrix * adj, dim=-1)) / (torch.sum(x_sim_matrix, dim=-1))
        # contrastive_loss = (pos_sim + eta_c  * torch.sum(sim_inter_view * adj, dim=-1)) / (torch.sum(sim_inter_view, dim=-1))
        # contrastive_loss = pos_sim + eta_c  * torch.sum(sim_inter_view * adj, dim=-1)
        # contrastive_loss = (pos_sim + eta_c  * torch.sum(sim_inter_view * adj, dim=-1)) / (torch.sum(sim_inter_view, dim=-1))
        loss_hsc = - torch.log(loss_hsc).mean()

        return loss_hsc

    def infoNCE_loss(self, x_1, x_2, tau = 0.1, device = torch.device('cuda:0')):

        n_samples = x_1.size(0)
        x_sim_matrix = cosine_sim(x_1, x_2, device=self.device)
        sim_inter_view = torch.exp(x_sim_matrix / tau)
        # pos_sim = sim_inter_view[range(n_samples), range(n_samples)]
        pos_sim = torch.diag(sim_inter_view)
        # P / (N + M)
        contrastive_loss = pos_sim / (torch.sum(sim_inter_view, dim=-1))
        contrastive_loss = - torch.log(contrastive_loss).mean()

        return contrastive_loss

    def infoNCE_NB_loss(self, x_1, x_2, adj, tao_c = 0.1, device = torch.device('cuda:0')):

        n_samples = x_1.size(0)
        x_sim_matrix = cosine_sim(x_1, x_2, device=self.device)
        sim_inter_view = torch.exp(x_sim_matrix / tao_c)
        # pos_sim = sim_inter_view[range(n_samples), range(n_samples)]
        pos_sim = torch.diag(sim_inter_view)
        # P / (N + M)
        # loss = pos_sim / (torch.sum(sim_inter_view, dim=-1))
        loss = (pos_sim + torch.sum(sim_inter_view * adj, dim=-1)) / (torch.sum(sim_inter_view, dim=-1))
        loss = - torch.log(loss).mean()

        return loss

    def clustering_loss(self, emb1, emb2, emb_fu, centers, alpha=1.0, device = torch.device('cuda:0')):

        y_wave_1 = soft_assignment(emb1, self.cluster_layer, alpha=alpha)
        y_wave_2 = soft_assignment(emb2, self.cluster_layer, alpha=alpha)
        y_wave_fu = soft_assignment(emb_fu, self.cluster_layer, alpha=alpha)
        target = target_distribution(y_wave_fu)

        loss_clustering = F.kl_div(y_wave_fu.log(), target, reduction='batchmean')
        loss_clustering += F.kl_div(y_wave_1.log(), target, reduction='batchmean')
        loss_clustering += F.kl_div(y_wave_2.log(), target, reduction='batchmean')
        # kl = torch.mul(target, torch.log(target / y_wave_1))
        # loss_clustering = torch.sum(kl)

        return loss_clustering, y_wave_fu

    def clustering_loss2(self, emb_fu, alpha=1.0, device = torch.device('cuda:0')):
        y_wave_fu = soft_assignment(emb_fu, self.cluster_layer, alpha=alpha)
        target = target_distribution(y_wave_fu)

        loss_clustering = F.kl_div(y_wave_fu.log(), target, reduction='batchmean')
        # kl = torch.mul(target, torch.log(target / y_wave_1))
        # loss_clustering = torch.sum(kl)

        return loss_clustering, y_wave_fu

    def clustering_fu_loss(self, emb_fu, centers, alpha=1.0, device = torch.device('cuda:0')):

        y_wave_fu = soft_assignment(emb_fu, self.cluster_layer, alpha=alpha)
        # target = target_distribution(y_wave_fu)
        t_wave = soft_assignment(emb_fu, centers, alpha=alpha)
        target = target_distribution(t_wave)

        loss_clustering = F.kl_div(y_wave_fu.log(), target, reduction='batchmean')
        # kl = torch.mul(target, torch.log(target / y_wave_1))
        # loss_clustering = torch.sum(kl)

        return loss_clustering, t_wave


    def GDA_loss(self, x_1, x_2, tao_kl=0.1, device=torch.device('cuda:0')):
        # View level KL divergence
        p_x_1_view = torch.exp(x_1 / tao_kl)
        p_x_1_view = p_x_1_view / torch.sum(p_x_1_view)
        p_x_2_view = torch.exp(x_2 / tao_kl)
        p_x_2_view = p_x_2_view / torch.sum(p_x_2_view)

        kl_view = torch.sum(p_x_1_view * (torch.log(p_x_1_view) - torch.log(p_x_2_view)))
        kl_view += torch.sum(p_x_2_view * (torch.log(p_x_2_view) - torch.log(p_x_1_view)))
        
        # Node level KL divergence
        p_x_1_node = F.softmax(x_1 / tao_kl, dim=-1)
        p_x_2_node = F.softmax(x_2 / tao_kl, dim=-1)

        kl_node = torch.sum(p_x_1_node * (torch.log(p_x_1_node) - torch.log(p_x_2_node))) / p_x_1_node.size(0)
        kl_node += torch.sum(p_x_2_node * (torch.log(p_x_2_node) - torch.log(p_x_1_node))) / p_x_2_node.size(0)
        kl_loss = kl_view + kl_node / 2
        # kl_loss = kl_node

        return kl_loss


    def ENC_loss(self, x_1, x_2, adj, eta = 1.0, tao_kl=0.1, device = torch.device('cuda:0')):
        # Neighborhood distillation using cosine distance for each node.
        # dist_matrix = cosine_distance(x_stu, x_tch, device=device)

        # Neighborhood distillation using KL divergence for each node.
        dist_matrix = kl_div_matrix(x_1, x_2, tao_kl, device=device)
        # dist_matrix = (dist_matrix + kl_div_matrix(x_tch, x_stu, tao_kl, device=device))
        dist_matrix = (dist_matrix + kl_div_matrix(x_2, x_1, tao_kl, device=device).t())
        
        # Use distance matrix, and compute the average.
        # adj without self loops here.
        n_samples = adj.size(0)
        dist_neighnors = dist_matrix * (adj * eta + torch.eye(n_samples, device=adj.device))
        # dist_neighnors = dist_matrix * (torch.eye(n_samples, device=adj.device))        # setting1 xi = 0
        # dist_neighnors = dist_matrix * (adj + torch.eye(n_samples, device=adj.device))  # setting2 xi = 1

        sum_dist_neighnors = torch.sum(dist_neighnors, dim=-1) / (torch.sum(adj, dim=-1) + 1)
        # sum_dist_neighnors = torch.sum(dist_neighnors, dim=-1) / 1
        sum_sample_dist = torch.sum(dist_matrix * (1 - torch.eye(n_samples, device=adj.device)), dim=-1) / (n_samples - 1)
        # (P + M) / (N + M)
        loss_enc = torch.mean(sum_dist_neighnors / sum_sample_dist)       # mean
        # loss_sample_level = torch.sum(sum_dist_neighnors / sum_sample_dist)        # sum

        return loss_enc
    
    
    # def cross_views_matching_loss(self, device = torch.device('cuda')):
    #
    #     Z11 = self.emb_stu
    #     Z12 = self.z_tch
    #     Z21 = self.emb_stu
    #     Z22 = self.z_tch
    #
    #     # cost1 = get_cosine_distance(Z11, Z22, device = device)
    #     # cost2 = get_cosine_distance(Z21, Z12, device = device)
    #     # cost = (cost1 + cost2)
    #
    #     cost = cosine_distance(Z12, Z22, device = device)
    #     # cost = 1 - torch.cosine_similarity(Z11, Z21)
    #     # cost = 1 - F.cosine_similarity(Z11.unsqueeze(1), Z21.unsqueeze(0), dim = -1)
    #
    #     mu = torch.ones((cost.shape[0],))
    #     mu = mu / mu.sum()
    #     nu = torch.ones((cost.shape[1],))
    #     nu = nu / nu.sum()
    #     mu = mu.to(device)
    #     nu = nu.to(device)
    #
    #     P = ot.sinkhorn(mu, nu, cost, 1.0, stopThr=1e-6)
    #
    #     # np.save('work/cost_matrix.npy', cost.cpu().detach().numpy())
    #     # np.save('work/P_matrix.npy', P.cpu().detach().numpy())
    #
    #     # show_heat_map(P.cpu().detach().numpy(), title = 'P Matrix', save_path= 'figures/P_heat_map.png')
    #
    #     pz21 = torch.mm(P, Z21)
    #     pz22 = torch.mm(P, Z22)
    #
    #     cvm_loss = F.mse_loss(Z12, pz21) + F.mse_loss(pz22, Z11)
    #
    #     return cvm_loss


class Model_Vanilla(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout=0.0, param_shared=True, device=torch.device('cuda:0')):
        super(Model_Vanilla, self).__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.param_shared = param_shared
        self.act = 'relu'
        self.device = device

        # self.teacher_net1 = GCN_net(self.in_dim, self.hid_dim, self.hid_dim, self.dropout, self.act)
        # self.teacher_net2 = nn.Linear(self.hid_dim, self.hid_dim)

        self.student_net1 = nn.Linear(self.in_dim, self.hid_dim)

        self.head = nn.Linear(self.hid_dim, self.out_dim)

        # if self.param_shared:
        #     self.shared_head = nn.Linear(self.hid_dim, self.out_dim)
        # else:
        #     self.teacher_head = nn.Linear(self.hid_dim, self.out_dim)
        #     self.student_head = nn.Linear(self.hid_dim, self.out_dim)

    def forward(self, x):
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.student_net1(x)
        # h_stu = F.normalize(h_stu, dim=1, p=2)
        if self.act == 'relu':
            x = F.leaky_relu(x, inplace=True)

        emb_stu = F.normalize(x, dim=1, p=2)

        x = F.dropout(emb_stu, self.dropout, self.training)

        x = self.head(x)

        z = F.normalize(x, dim=1, p=2)

        return emb_stu, z

    def vanilla_contrastive_loss(self, emb1, emb2, adj, eta_c=1.0, tao_c=1.0, device=torch.device('cuda:0')):

        n_samples = emb1.size(0)
        x_sim_matrix = cosine_sim(emb1, emb2, device=device)
        sim_inter_view = torch.exp(x_sim_matrix / tao_c)

        pos_sim = sim_inter_view[range(n_samples), range(n_samples)]
        # neighbor_sim = sim_inter_view * adj         # adj without self loops here.
        # (P + nu * M) / (N + M)
        contrastive_loss = (pos_sim + eta_c * torch.sum(sim_inter_view * adj, dim=-1)) / (
                    torch.sum(sim_inter_view, dim=-1))
        # contrastive_loss = (pos_sim + eta_c  * torch.sum(sim_inter_view * adj, dim=-1)) / (torch.sum(sim_inter_view, dim=-1))
        # contrastive_loss = pos_sim + eta_c  * torch.sum(sim_inter_view * adj, dim=-1)
        # contrastive_loss = (pos_sim + eta_c  * torch.sum(sim_inter_view * adj, dim=-1)) / (torch.sum(sim_inter_view, dim=-1))
        contrastive_loss = - torch.log(contrastive_loss).mean()

        return contrastive_loss

    def vanilla_infoNCE_loss(self, emb1, emb2, tao_c = 1.0, device = torch.device('cuda:0')):

        n_samples = emb1.size(0)
        x_sim_matrix = cosine_sim(emb1, emb2, device=device)
        sim_inter_view = torch.exp(x_sim_matrix / tao_c)

        pos_sim = sim_inter_view[range(n_samples), range(n_samples)]
        # P / (N + M)
        # contrastive_loss = pos_sim / (torch.sum(sim_inter_view, dim=-1) - pos_sim)
        contrastive_loss = pos_sim / torch.sum(sim_inter_view, dim=-1)
        contrastive_loss = - torch.log(contrastive_loss).mean()

        return contrastive_loss


# def HSC_loss(self, z1, z2, adj, tau = 0.1, device = torch.device('cuda:0')):
#
#     # x_sim_matrix = cosine_sim(x_1, x_2, device=device)
#     # x_sim_matrix = self.z_sim_matrix
#     sim_inter_view = self.emb_sim_matrix
#     sim_inter_view = torch.exp(sim_inter_view / tau)
#
#     sim_intra_z1 = torch.exp(cosine_sim(z1, z1, device) / tau)
#     sim_intra_z2 = torch.exp(cosine_sim(z2, z2, device) / tau)
#     diag1 = torch.diag(sim_intra_z1)
#     sim_intra_z1 = sim_intra_z1 - torch.diag_embed(diag1)
#     diag2 = torch.diag(sim_intra_z2)
#     sim_intra_z2 = sim_intra_z2 - torch.diag_embed(diag2)
#     # sim_inter_view = torch.exp(sim_inter_view / tau)
#
#     pos_sim = torch.diag(sim_inter_view) + torch.sum(sim_inter_view * adj, dim=-1) \
#               + torch.sum(sim_intra_z1 * adj, dim=-1) + torch.sum(sim_intra_z2 * adj, dim=-1)
#     # (P + nu * M) / (N + M)
#     loss_hsc = pos_sim / (torch.sum(sim_inter_view, dim=-1) + torch.sum(sim_intra_z1, dim=-1)
#                           + torch.sum(sim_intra_z2, dim=-1))
#     # loss_hsc = (pos_sim) / (torch.sum(sim_inter_view, dim=-1))      # setting1
#     # loss_hsc = (pos_sim + torch.sum(sim_inter_view * adj, dim=-1)) / (torch.sum(sim_inter_view, dim=-1))  # setting2
#
#     loss_hsc = - torch.log(loss_hsc).mean()
#
#     return loss_hsc
