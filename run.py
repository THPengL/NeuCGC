import os
import numpy as np
import argparse
from configure import set_default_config
import warnings

warnings.simplefilter("ignore")


def main(config):
    # after setting device id.
    from train import train
    from utils import get_logger

    logger = get_logger(root=f"./training_logs/{config['dataset']}",
                        filename=config['log_name'])
    # Training
    if config['train']:
        seeds = range(0, 10, 1)
    else:
        seeds = [5]

    best_result = train(config, logger, seeds)

    logger.info(f"Dataset: {config['dataset']}")
    # logger.info(f"l1: {config['lambda1']}, l2: {config['lambda2']}, k: {config['k']}, d: {config['out_dim']}")
    # logger.info(f"- ACC NMI ARI F1")
    # logger.info(f"\n{np.mean(best_result['acc']) * 100:.2f} ± {np.std(best_result['acc']) * 100:.2f}" +
    #             f"\n{np.mean(best_result['nmi']) * 100:.2f} ± {np.std(best_result['nmi']) * 100:.2f}" +
    #             f"\n{np.mean(best_result['ari']) * 100:.2f} ± {np.std(best_result['ari']) * 100:.2f}" +
    #             f"\n{np.mean(best_result['f1']) * 100:.2f} ± {np.std(best_result['f1']) * 100:.2f}")
    logger.info(f"- ACC: {np.mean(best_result['acc']) * 100:.2f} ± {np.std(best_result['acc']) * 100:.2f}")
    logger.info(f"- NMI: {np.mean(best_result['nmi']) * 100:.2f} ± {np.std(best_result['nmi']) * 100:.2f}")
    logger.info(f"- ARI: {np.mean(best_result['ari']) * 100:.2f} ± {np.std(best_result['ari']) * 100:.2f}")
    logger.info(f"- F1 : {np.mean(best_result['f1']) * 100:.2f} ± {np.std(best_result['f1']) * 100:.2f}")
    logger.info(f"========== Training Over ==========")


if __name__ == '__main__':
    # Set arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='wisconsin', help='name of dataset.',
                        choices=['wisconsin', 'cornell', 'texas', 'chameleon', 'crocodile',
                                 'cora', 'citeseer', 'pubmed', 'dblp', 'acm', 'photo'])
    parser.add_argument('--datasetid', type=int, default=0, help='Dataset number in datasets dictionary.')
    parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs.')
    parser.add_argument('--out_dim', type=int, default=1000, help='Output layer dim')
    parser.add_argument('--lr', type=float, default=1e-5, help='Initial learning rate.')
    parser.add_argument('--lambda1', type=float, default=1.0, help='Loss balance parameter.')
    parser.add_argument('--lambda2', type=float, default=1.0, help='Loss balance parameter.')
    parser.add_argument('--k', type=float, default=0.1, help='.')

    args = parser.parse_args()

    datasets = {
        0: "wisconsin", 1: "cornell", 2: "texas", 3: "chameleon", 4: "crocodile",
        5: "cora", 6: "citeseer", 7: "pubmed", 8: "acm", 9: "dblp", 10: "photo"
    }
    # Set environment
    args.deviceid = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.deviceid)

    args.datasetid = 5
    args.dataset = datasets[args.datasetid]

    config = set_default_config(args)

    config['train'] = True  # True: 10 seeds; False: 1 seed

    # config['lr'] = args.lr
    # config['lambda1'] = args.lambda1
    # config['lambda2'] = args.lambda2
    # config['out_dim'] = args.out_dim
    # config['k'] = args.k

    config['log_name'] = f"{config['dataset']}_20250624_log.txt"

    main(config)
