
from tqdm import tqdm
from torch import optim
from model import Model
from utils import *
from clustering import clustering
from evaluation import evaluate


def train(config, logger, seeds):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.info("Loading dataset...")
    data = load_graph_dataset(root='./datasets',
                              data_name=config['dataset'],
                              remove_self_loop=True)

    config['n_samples'] = data['n_samples']
    config['n_classes'] = data['n_classes']
    input_feature_1, input_feature_2 = data['feature_'], data['feature']
    label_true = data['label']
    adj = data['adj'].toarray()
    config['n_edges'] = data['n_edges']

    adj = torch.tensor(adj, dtype=torch.int64)
    adj = adj.fill_diagonal_(0)         # set the diagonal of adj as 0, remove self loop

    input_feature_1 = torch.FloatTensor(input_feature_1)
    input_feature_2 = torch.FloatTensor(input_feature_2)

    config['in_dim'] = data['feature'].shape[-1]

    logger.info('Show Config:')
    for (k, v) in config.items():
        logger.info(f'{k:>12} : {v}')

    best_result = {
        'acc': [], 'nmi': [], 'ari': [], 'f1':[],
    }

    for seed in seeds:
        set_random_seed(seed)
        logger.info(f'====================== SEED {seed} ======================')

        model = Model(in_dim=config['in_dim'],
                      out_dim=config['out_dim'],
                      dropout=config['dropout'],
                      device = device)
        optimizer = optim.Adam(params = model.parameters(), 
                               lr=float(config['lr']), 
                               weight_decay=float(config['weight_decay']))
        # logger.info(model)
        # logger.info(optimizer)

        model = model.to(device)
        input_feature_1 = input_feature_1.to(device)
        input_feature_2 = input_feature_2.to(device)
        adj = adj.to(device)

        acc_best, nmi_best, ari_best, f1_best, best_epoch = 0.0, 0.0, 0.0, 0.0, 0

        for epoch in tqdm(range(config['epochs'])):
            model.train()
            emb_1, emb_2 = model(input_feature_1, input_feature_2)
            model.compute_emb_sim(emb_1, emb_2)

            if epoch == 0:
                emb_fusion = get_fusion(z1=emb_1.detach(), z2=emb_2.detach())
                label_pred, centers, dis = clustering(feature=emb_fusion,
                                                      cluster_num=config['n_classes'],
                                                      method='kmeans',
                                                      device=device)
            eta = model.compute_eta(adj)
            pseudo_adj = model.high_confidence_adj(label_pred.copy(),
                                                   dis,
                                                   adj,
                                                   k=config['k'])
            loss_gda = model.GDA_loss(x_1=emb_1, x_2=emb_2)
            loss_nca = model.NCA_loss(x_1=emb_1,
                                      x_2=emb_2,
                                      adj=adj,
                                      eta=eta)
            loss_afc = model.AFC_loss(emb_1, emb_2, H=pseudo_adj)

            loss = loss_nca + config['lambda1'] * loss_afc + config['lambda2'] * loss_gda

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            model.eval()
            emb_1, emb_2 = model(input_feature_1, input_feature_2)

            emb_fusion = get_fusion(z1=emb_1.detach(),
                                    z2=emb_2.detach())

            label_pred, centers, dis = clustering(feature=emb_fusion,
                                                  cluster_num=config['n_classes'],
                                                  device=device)

            acc, nmi, ari, f1 = evaluate(label_true, label_pred)

            if acc >= acc_best:
                acc_best = acc
                nmi_best = nmi
                ari_best = ari
                f1_best = f1
                best_epoch = epoch+1

        logger.info(f'ACC {acc_best:.4f}, NMI {nmi_best:.4f}, ARI {ari_best:.4f}, F1 {f1_best:.4f}')

        best_result['acc'].append(acc_best)
        best_result['nmi'].append(nmi_best)
        best_result['ari'].append(ari_best)
        best_result['f1'].append(f1_best)

    return best_result

