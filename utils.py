import os
import torch
import random
import logging
import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import torch.nn.functional as F



def min_max_normlize(x):
    if isinstance(x, torch.Tensor):
        s_max, _ = torch.max(x, dim=-1, keepdim=True)
        s_min, _ = torch.min(x, dim=-1, keepdim=True)
    elif isinstance(x, np.ndarray):
        s_max, _ = np.max(x, axis=-1, keepdims=True)
        s_min, _ = np.min(x, axis=-1, keepdims=True)
    x = (x - s_min) / (s_max - s_min)
    return x


def add_self_loops(A, value=1.0):
    """Set the diagonal for sparse adjacency matrix."""
    A = A.tolil()  # make sure we work on a copy of the original matrix
    A.setdiag(value)
    A = A.tocsr()
    if value == 0:
        A.eliminate_zeros()
    return A


def eliminate_self_loops(A):
    """Remove self-loops from the sparse adjacency matrix."""
    A = A.tolil()
    A.setdiag(0)
    A = A.tocsr()
    A.eliminate_zeros()
    return A


def load_graph_dataset(root, data_name, all_info=False, remove_self_loop=True, add_self_loop=False):

    data = sio.loadmat(os.path.join(root, f'{data_name}.mat'))

    if remove_self_loop:
        data['adj'] = eliminate_self_loops(data['adj'])
        edge_index = np.where(data['adj'].toarray() > 0)
        data['edge_index'] = np.concatenate((np.expand_dims(edge_index[0], axis=0), np.expand_dims(edge_index[1], axis=0)), axis=0)
        data['n_edges'] = round(data['edge_index'].shape[1] / 2)
        data['Note'] = 'All edges have weights of 1.'

    if add_self_loop:
        data['adj'] = add_self_loops(data['adj'])
        edge_index = np.where(data['adj'].toarray() > 0)
        data['edge_index'] = np.concatenate((np.expand_dims(edge_index[0], axis=0), np.expand_dims(edge_index[1], axis=0)), axis=0)
        data['n_edges'] = round(data['edge_index'].shape[1] / 2)
        data['Note'] = 'All edges have weights of 1. And there are some nodes with self loops.'

    if all_info:
        data['label'] = data['label'][0]
        data['n_classes'] = len(np.unique(data['label']))
        data['label_shape'] = data['label'].shape
        data['n_samples'] = data['feature'].shape[0]

        return data
    
    else:
        return data['feature'], data['label'][0], data['adj'], data['edge_index']


def laplacian_smooth(feature, adj, num_layer, renorm=True, norm_mode='symm'):
    """
    Laplacian smoothing for feature matrix.

    Parameters
    - feature: NDArray. Feature matrix.
    - adj: NDArray | coo_matrix. Adjacent matrix.
    - num_layer: int. Number of laplacian smoothing layers.
    - renorm: bool. Whether to add self loop for adj matrix. If True, than adj add self-loop.
    - norm_mode: str. options: 'symm' or 'random'. If norm_mode='symm', than perform Symmetric 
        normalization, A_norm = D^-1/2 L D^-1/2 = I - D^-1/2 A D^-1/2; \n
        if 'random', than perform Random walk normalization,
        A_norm = D^-1 L = I - D^-1 A.

    Return
    - X: NDArray. Smoothed feature matrix.
    """
    adj = sp.coo_matrix(adj)
    I = sp.eye(adj.shape[0])
    A = adj if not renorm else adj + I
    row_sum = np.array(A.sum(1))

    if norm_mode == 'symm':
        """
        Symmetric normalization:\
        A_norm = D^-1/2 L D^-1/2 = I - D^-1/2 A D^-1/2. 
        """
        D_inv_sqrt = sp.diags(np.power(row_sum, -0.5).flatten())
        # A_norm = A.dot(D_inv_sqrt).transpose().dot(D_inv_sqrt).tocoo()
        A_norm = D_inv_sqrt.dot(A).dot(D_inv_sqrt).tocoo()
        L = I - A_norm

    elif norm_mode == 'random':
        """
        Random walk normalization:\
        A_norm = D^-1 L = I - D^-1 A. 
        """
        D_inv_sqrt = sp.diags(np.power(row_sum, -1.).flatten())
        A_norm = D_inv_sqrt.dot(A).tocoo()
        L = I - A_norm

    X = feature
    for i in range(num_layer):
        H = I - L
        X = H.dot(X)

    X = sp.csr_matrix(X).toarray()

    return X


def get_fusion(z1, z2, fusion_mode = 1):
    if fusion_mode == 1:
        fusion = torch.concat((z1, z2), dim=-1)

    elif fusion_mode == 2:
        fusion = (z1 + z2) / 2

    else:
        fusion = z1 + z2

    return fusion


def euclidean_distance(data1, data2, device=torch.device('cuda')):
    # transfer to device
    data1 = data1.to(device)
    data2 = data2.to(device)

    # N*1*M
    A = data1.unsqueeze(dim=1)
    # 1*N*M
    B = data2.unsqueeze(dim=0)

    dis = (A - B) ** 2.0
    # N*N matrix for pairwise euclidean distance
    dis = dis.sum(dim=-1).squeeze()

    return dis


def cosine_sim(data1, data2, device = torch.device('cuda')):
    """
    Calculate the cosine similarity between each row in matrix A and that in matrix B.\n
    计算矩阵 A 中的每行与矩阵 B 中的每行之间的余弦相似度

    Parameters
    - data1 (Tensor): input matrix (N x M)
    - data2 (Tensor): input matrix (N x M)
    - device (str | Optional): 'cpu' or 'cuda'

    Return
    - cos_sim (Tensor): (N x N) cosine similarity between each row in matrix data1 and\\ 
        that in matrix data2.
    """
    A, B = data1.to(device), data2.to(device)

    # # unitization eg.[3, 4] -> [3/sqrt(9 + 16), 4/sqrt(9 + 16)] = [3/5, 4/5]
    # A_norm = A / torch.norm(A, dim = 1, keepdim = True)
    # B_norm = B / torch.norm(B, dim = 1, keepdim = True)

    nume_A = torch.norm(A, dim = 1, keepdim = True)
    idxs_a = torch.where(nume_A == 0)           # (tensor([0, 3, 7, 8]),)
    if len(idxs_a[0]) > 0:
        nume_A[idxs_a[0]] += 1
    nume_B = torch.norm(B, dim = 1, keepdim=True)
    idxs_b = torch.where(nume_B == 0)
    if len(idxs_b[0]) > 0:
        nume_B[idxs_b[0]] += 1

    A_norm = A / nume_A
    B_norm = B / nume_B
    
    cos_sim = torch.mm(A_norm, B_norm.t())

    # OR
    # A_norm = A / A.norm(dim = -1, keepdim = True)
    # B_norm = B / B.norm(dim = -1, keepdim = True)
    # cos_sim = (A_norm * B_norm).sum(dim=-1)

    return cos_sim


def cosine_distance(data1, data2, device = torch.device('cuda')):
    """\
    Calculate the cosine distance between each row in matrix A and that in matrix B.

    Parameters
    - data1 (Tensor): input matrix (N x M)
    - data2 (Tensor): input matrix (N x M)
    - device (str | Optional): 'cpu' or 'cuda'

    Return
    - cos_distance (Tensor): (N x N) cosine distance between each row in matrix data1 and\\ 
        that in matrix data2.
    """
    cos_sim = cosine_sim(data1, data2, device)
    cos_distance = 1 - cos_sim

    return cos_distance


def view_distribution(data):
    temperature = 0.1
    p_view = torch.exp(data / temperature)
    p_view = p_view / torch.sum(p_view)

    return p_view


def node_distribution(data):
    temperature = 0.1
    p_node = F.softmax(data / temperature, dim=-1)

    return p_node


def kl_div_matrix(matrix1, matrix2, device=torch.device('cuda')):
    temperature = 0.1
    if matrix1.device != device or matrix2.device != device:
        matrix1, matrix2 = matrix1.to(device), matrix2.to(device)

    matrix1 = F.softmax(matrix1 / temperature, dim=-1)
    matrix2 = F.softmax(matrix2 / temperature, dim=-1)

    n_samples = matrix1.size(0)

    kl_div_matrix = torch.sum(matrix1 * torch.log(matrix1), dim=-1).view(n_samples, 1)
    kl_div_matrix = kl_div_matrix.expand(n_samples, n_samples) - torch.matmul(matrix1, torch.log(matrix2).t())
    
    return kl_div_matrix


def scale_matrix(matrix, mode=1, temperature = 0.1):
    if mode == 1:
        scale_matrix = torch.exp(matrix / temperature)
    elif mode == 2:
        scale_matrix = matrix / temperature
    elif mode == 3:
        s_max, _ = torch.max(matrix, dim=-1, keepdim=True)
        s_min, _ = torch.min(matrix, dim=-1, keepdim=True)
        scale_matrix = (matrix - s_min) / (s_max - s_min)
    else:
        scale_matrix = matrix

    return scale_matrix


def pseudo_graph(label_pred, device):
    if isinstance(label_pred, torch.Tensor):
        pseudo_label = label_pred.clone().to(device)
    elif isinstance(label_pred, np.ndarray):
        pseudo_label = torch.tensor(label_pred.copy()).to(device)
    pseudo_g = (pseudo_label == pseudo_label.unsqueeze(1)).float().to(device)

    diag = torch.diag(pseudo_g)
    pseudo_g = pseudo_g - torch.diag_embed(diag)

    return pseudo_g



def compute_knn_graph(sim_matrix, k=3, device=torch.device('cuda:0')):
    diag = torch.diag(sim_matrix)
    sim_matrix = sim_matrix - torch.diag_embed(diag)

    values, indices = torch.topk(sim_matrix, k, dim=-1, largest=True)
    adj = torch.zeros(sim_matrix.size(), device=device)
    # adj[indices] = 1
    adj = adj.scatter_(1, indices, 1)
    # adj = adj + adj.t()
    # adj = torch.clamp(adj, max=1)

    return adj


def pseudo_labels(label_pred, centers, dis, k, device=torch.device('cuda:0')):
    label_pred = torch.tensor(label_pred, device=device)
    high_confidence = torch.min(dis, dim=1).values
    threshold = torch.sort(high_confidence).values[int(len(high_confidence) * k)]
    # high_confidence_idx = np.argwhere(high_confidence <= threshold)[0]
    # h_i = high_confidence_idx.numpy()
    high_confidence_idx = torch.where(high_confidence <= threshold)[0]
    pseudo_labels = label_pred[high_confidence_idx]  # (1 x N*k)

    return pseudo_labels, high_confidence_idx


def pseudo_neighbor(emb1, emb2, adj, k=2, device=torch.device('cuda')):
    sim_matrix = cosine_sim(emb1, emb2, device=device)
    diag = torch.diag(sim_matrix)
    sim_matrix = sim_matrix - torch.diag_embed(diag)

    values, indices = torch.topk(sim_matrix, k, dim=-1)         # top k neighbor
    pseudo_adj = torch.zeros_like(sim_matrix, device=device)
    pseudo_adj = pseudo_adj.scatter_(1, indices, 1)              # 在每行中按照列索引将值 scatter 到目标张量中

    # pseudo_adj = pseudo_adj + adj
    # pseudo_adj = torch.where(pseudo_adj == 2, pseudo_adj, 1)

    return pseudo_adj


def homo_ratio(edge_index, label):
    homo_info = {}
    homo_info['n_edges'] = edge_index.shape[1]

    same_label = 0
    for i in range(edge_index.shape[1]):
        if label[edge_index[0, i]] == label[edge_index[1, i]]:
            same_label += 1
    homo_ratio = same_label / edge_index.shape[1]
    homo_info['homo_ratio'] = homo_ratio

    edge_list = []
    true_neighbor_list = []
    ratio_list = []
    last_node = edge_index[0, 0]
    true_nb_num = 0
    n_edge = 0
    for i in range(edge_index.shape[1]):
        curr_node = edge_index[0, i]
        if curr_node != last_node:
            ratio = true_nb_num / n_edge
            edge_list.append(n_edge)
            true_neighbor_list.append(true_nb_num)
            ratio_list.append(ratio)
            n_edge = 0
            true_nb_num = 0
            last_node = curr_node
            # for the last node, e.g. [[ ... 8 8 9], [ ... 7 9 10]]
            if i == edge_index.shape[1] - 1:
                n_edge += 1
                curr_neighbor = edge_index[1, i]
                if label[curr_node] == label[curr_neighbor]:
                    true_nb_num += 1
                ratio = true_nb_num / n_edge
                ratio_list.append(ratio)
                edge_list.append(n_edge)
                true_neighbor_list.append(true_nb_num)
                break
        # for each node
        n_edge += 1
        curr_neighbor = edge_index[1, i]
        if label[curr_node] == label[curr_neighbor]:
            true_nb_num += 1
        # for the last node, e.g. [[ ... 9 9 9], [ ... 7 8 10]]
        if curr_node == last_node and i == edge_index.shape[1] - 1:
            ratio = true_nb_num / n_edge
            ratio_list.append(ratio)
            edge_list.append(n_edge)
            true_neighbor_list.append(true_nb_num)

    neighbor_homo_ratio = np.mean(ratio_list)
    homo_info['neighbor_homo_ratio'] = neighbor_homo_ratio
    homo_info['n_edge_list'] = edge_list
    homo_info['true_neighbor_list'] = true_neighbor_list
    homo_info['ratio_list'] = ratio_list

    return homo_info


def graph_congener_ratio(edge_index, label):
    info = {}
    n_nodes = len(label)
    class_counts_dict = {key: np.sum(label == key) for key in label}
    info['num_per_class'] = class_counts_dict
    true_neighbor_per_node = np.zeros(n_nodes)
    same_class_per_node = np.zeros(n_nodes)
    ratio_per_node = np.zeros(n_nodes)
    for i in range(edge_index.shape[1]):
        if label[edge_index[0, i]] == label[edge_index[1, i]]:
            true_neighbor_per_node[edge_index[0, i]] += 1
    info['true_neighbor_per_node'] = true_neighbor_per_node
    for i in range(n_nodes):
        ratio_per_node[i] = true_neighbor_per_node[i] / class_counts_dict[label[i]]
        same_class_per_node[i] = class_counts_dict[label[i]]
    info['ratio_per_node'] = ratio_per_node
    info['same_class_per_node'] = same_class_per_node
    ratio_mean = ratio_per_node[ratio_per_node > 0].mean()
    info['ratio_mean'] = ratio_mean

    return info


def get_logger(root = './training_logs', filename = None):
    """
    Get logger.

    Parameters
    - root: str. Root directory of log files.
    - filename: str, Optional. The name of log files.

    return
    - logger: Logger
    """

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        fmt = '%(message)s')

    if filename is not None:
        """Save logs as files"""
        if not os.path.exists(root):
            os.makedirs(root)

        # mode = 'w', overwriting the previous content, 'a', appended to previous file.
        fh = logging.FileHandler(os.path.join(root, filename), "a")
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    """Print logs at terminal"""
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

