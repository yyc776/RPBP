import pandas as pd

from .utils import get_data_mask, split_manual_lt, separate_ht

import torch


class Embedder:
    def __init__(self, args):
        if args.gpu == 'cpu':
            args.device = 'cpu'
        else:
            args.device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
        all_smiles = []
        all_labels = []
        train_test_eval_idx = []
        numbers = 0
        datasets = ['train', 'test', 'eval']
        for idx, data_name in enumerate(datasets):
            data = pd.read_csv(r'%s/dataset/stage_one/%s/byproduct_rxn_%s.csv' % (args.task_dataset,data_name, data_name))
            all_smiles.extend(data['product'].tolist())
            all_labels.extend(data['label'].tolist())
            train_test_eval_idx.append((numbers, numbers + len(data['product'].tolist())))
            numbers += len(data['product'].tolist())

        labels = torch.tensor(all_labels)
        data_train_mask, data_eval_mask, data_test_mask = get_data_mask(train_test_eval_idx)

        total_nodes = len(labels)
        idx_train = torch.tensor(range(total_nodes))[data_train_mask]
        idx_eval = torch.tensor(range(total_nodes))[data_eval_mask]
        idx_test = torch.tensor(range(total_nodes))[data_test_mask]

        idx_train, idx_eval, idx_test, class_num_mat = split_manual_lt(labels, idx_train, idx_eval, idx_test)
        samples_per_label = torch.tensor(class_num_mat[:, 0])

        idx_train_set_class, ht_dict_class = separate_ht(samples_per_label, labels, idx_train,
                                                         method=args.sep_class)
        self.idx_train_set_class = idx_train_set_class
        self.sep_point = len(ht_dict_class['H'])
        self.nclass = labels.max().item() + 1
        self.args = args
