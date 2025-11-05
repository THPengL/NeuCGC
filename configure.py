

def set_default_config(args):
    if args.dataset == 'cora':
        return dict(
            dataset = 'cora',
            n_samples = 2708,
            n_classes = 7,
            n_edges = 5429,
            lr = 0.0001,
            epochs = 500,
            dropout=0.3,
            weight_decay = 0.01,
            lambda1 = 0.1,
            lambda2 = 1.0,
            k = 0.1,
            in_dim = 1433,
            out_dim = 1000
        )
    elif args.dataset == 'citeseer':
        return dict(
            dataset = 'citeseer',
            n_samples = 3327,
            n_classes = 6,
            n_edges = 4732,
            lr = 0.00002,
            epochs = 500,
            dropout=0.3,
            weight_decay = 0.01,
            lambda1= 1.0,
            lambda2 = 100.0,
            k = 0.1,
            in_dim = 3703,
            out_dim = 1000
        )
    elif args.dataset == 'pubmed':
        return dict(
            dataset = 'pubmed',
            n_samples = 19717,
            n_classes = 3,
            n_edges = 44324,
            lr = 0.0001,
            epochs = 500,
            dropout=0.3,
            weight_decay = 0.01,
            lambda1= 0.1,
            lambda2 = 1.0,
            k = 0.1,
            in_dim = 500,
            out_dim = 500
        )
    elif args.dataset == 'acm':
        return dict(
            dataset = 'acm',
            n_samples = 3025,
            n_classes = 3,
            n_edges = 13128,
            lr = 0.0001,
            epochs = 500,
            dropout=0.7,
            weight_decay = 0.01,
            lambda1= 1.0,
            lambda2 = 1.0,
            k=0.1,
            in_dim = 1870,
            out_dim = 1000
        )
    elif args.dataset == 'dblp':
        return dict(
            dataset = 'dblp',
            n_samples = 4057,
            n_classes = 4,
            n_edges = 3528,
            lr = 0.001,
            epochs = 500,
            dropout=0.3,
            weight_decay = 0.01,
            lambda1= 0.1,
            lambda2 = 1.0,
            k=0.1,
            in_dim = 334,
            out_dim = 300
        )
    elif args.dataset == 'texas':
        return dict(
            dataset = 'texas',
            n_samples = 183,
            n_classes = 5,
            n_edges = 309,
            lr = 0.00001,
            epochs = 500,
            dropout=0.3,
            weight_decay = 0.01,
            lambda1 = 100,
            lambda2 = 1.0,
            k=0.3,
            in_dim = 1793,
            out_dim = 1000
        )
    elif args.dataset == 'cornell':
        return dict(
            dataset = 'cornell',
            n_samples = 183,
            n_classes = 5,
            n_edges = 295,
            lr = 0.00001,
            epochs = 500,
            dropout=0.3,
            weight_decay = 0.01,
            lambda1 = 0.01,
            lambda2 = 1.0,
            k=0.3,
            in_dim = 1703,
            out_dim = 1000
        )
    elif args.dataset == 'wisconsin':
        return dict(
            dataset = 'wisconsin',
            n_samples = 251,
            n_classes = 5,
            n_edges = 466,
            lr = 0.00001,
            epochs = 500,
            dropout=0.3,
            weight_decay = 0.01,
            lambda1 = 0.01,
            lambda2 = 0.1,
            k=0.5,
            in_dim = 1703,
            out_dim = 1000
        )
    elif args.dataset == 'chameleon':
        return dict(
            dataset = 'chameleon',
            n_samples = 2277,
            n_classes = 5,
            n_edges = 31371,
            lr = 0.00001,
            epochs = 500,
            dropout=0.3,
            weight_decay = 0.01,
            lambda1 = 0.01,
            lambda2 = 1.0,
            k=0.1,
            in_dim = 2325,
            out_dim = 1000
        )
    elif args.dataset == 'crocodile':
        return dict(
            dataset = 'crocodile',
            n_samples = 11631,
            n_classes = 5,
            n_edges = 180020,
            lr = 0.0001,
            epochs = 500,
            dropout=0.3,
            weight_decay = 0.01,
            lambda1 = 1.0,
            lambda2 = 1.0,
            k = 0.1,
            in_dim = 128,
            out_dim = 1000
        )
    elif args.dataset == 'photo':
        return dict(
            dataset = 'photo',
            n_samples = 7650,
            n_classes = 8,
            n_edges = 119081,
            lr = 0.0002,
            epochs = 500,
            dropout=0.3,
            weight_decay = 0.01,
            lambda1 = 0.1,
            lambda2 = 0.1,
            k = 0.1,
            in_dim = 745,
            out_dim = 1000,
        )
    else:
        print("Using Default Configuration")
        return dict(
            dataset=args.dataset,
            n_samples=2000,
            n_classes=10,
            n_edges=3000,
            lr=0.0002,
            epochs=500,
            dropout=0.3,
            weight_decay=0.01,
            lambda1=1.0,
            lambda2=1.0,
            k=0.1,
            in_dim=745,
            out_dim=1000,
        )

