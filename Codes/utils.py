import os
import os.path as osp
import sys
import logging
import random
import pickle
import numpy as np
from rdkit import Chem
import scipy.sparse as sp
from datetime import datetime
from itertools import chain
import joblib
import re
from .mol_info import Label_Vocab

import torch
from torch_scatter import scatter_add
from torch_geometric.utils.loop import add_self_loops, remove_self_loops
from sklearn.metrics import f1_score, classification_report, confusion_matrix, balanced_accuracy_score
from imblearn.metrics import geometric_mean_score


def split_manual_lt(labels, idx_train, idx_val, idx_test):
    num_classes = len(set(labels.tolist()))
    c_num_mat = np.zeros((num_classes, 3)).astype(int)

    for i in range(num_classes):
        c_idx = (labels[idx_train] == i).nonzero()[:, -1].tolist()
        test_idx = (labels[idx_test] == i).nonzero()[:, -1].tolist()
        eval_idx = (labels[idx_val] == i).nonzero()[:, -1].tolist()
        print('{:d}-th class sample number: {:d}'.format(i, len(c_idx)))

        c_num_mat[i, 0] = len(c_idx)
        c_num_mat[i, 1] = len(eval_idx)
        c_num_mat[i, 2] = len(test_idx)

    train_idx = torch.LongTensor(idx_train)
    eval_idx = torch.LongTensor(idx_val)
    test_idx = torch.LongTensor(idx_test)

    return train_idx, eval_idx, test_idx, c_num_mat


def separate_ht(samples_per_label, labels, idx_train, method='pareto_28'):
    class_dict = {}
    idx_train_set = {}

    ht_dict = separator_ht(samples_per_label, method)  # H/T

    print('Samples per label:', samples_per_label)
    print('Separation:', ht_dict.items())

    for idx, value in ht_dict.items():
        class_dict[idx] = []
        idx_train_set[idx] = []
        idx = idx
        label_list = value

        for label in label_list:
            class_dict[idx].append(label)
            idx_train_set[idx].extend(torch.LongTensor(idx_train[labels[idx_train] == label]))

    for idx in list(ht_dict.keys()):
        random.shuffle(idx_train_set[idx])
        idx_train_set[idx] = torch.LongTensor(idx_train_set[idx])

    return idx_train_set, ht_dict


def separator_ht(dist, method='pareto_28', degree=False):  # Head / Tail separator
    head = int(method[-2])  # 2 in pareto_28
    tail = int(method[-1])  # 8 in pareto_28
    head_idx = int(len(dist) * (head / 10))
    ht_dict = {}

    if head_idx == 0:
        ht_dict['H'] = list(range(0, 1))
        ht_dict['T'] = list(range(1, len(dist)))
        return ht_dict

    else:
        crierion = dist[head_idx].item()

        case1_h = sum(np.array(dist) >= crierion)
        case1_t = sum(np.array(dist) < crierion)

        case2_h = sum(np.array(dist) > crierion)
        case2_t = sum(np.array(dist) <= crierion)

        gap_case1 = abs(case1_h / case1_t - head / tail)
        gap_case2 = abs(case2_h / case2_t - head / tail)

        if gap_case1 < gap_case2:
            idx = sum(np.array(dist) >= crierion)
            ht_dict['H'] = list(range(0, idx))
            ht_dict['T'] = list(range(idx, len(dist)))

        elif gap_case1 > gap_case2:
            idx = sum(np.array(dist) > crierion)
            ht_dict['H'] = list(range(0, idx))
            ht_dict['T'] = list(range(idx, len(dist)))

        else:
            rand = random.choice([1, 2])
            if rand == 1:
                idx = sum(np.array(dist) >= crierion)
                ht_dict['H'] = list(range(0, idx))
                ht_dict['T'] = list(range(idx, len(dist)))
            else:
                idx = sum(np.array(dist) > crierion)
                ht_dict['H'] = list(range(0, idx))
                ht_dict['T'] = list(range(idx, len(dist)))

        return ht_dict


def accuracy(output, labels, pre=None):
    # if sep in ['T', 'TH', 'TT']:
    #     labels = labels - sep_point  # [4,5,6] -> [0,1,2]

    if output.shape != labels.shape:
        if len(labels) == 0:
            return np.nan
        preds = output.max(1)[1].type_as(labels)
    else:
        preds = output

    correct = preds.eq(labels).double()
    correct = correct.sum()

    return correct / len(labels)


def classification(output, labels, sep_point=None, sep=None):
    target_names = []
    if len(labels) == 0:
        return np.nan
    else:
        if sep in ['T', 'TH', 'TT']:
            labels = labels - sep_point
        pred = output.max(1)[1].type_as(labels)
        for i in labels.unique():
            target_names.append(f'class_{int(i)}')

        return classification_report(labels, pred)


def confusion(output, labels, sep_point=None, sep=None):
    if len(labels) == 0:
        return np.nan
    else:
        if sep in ['T', 'TH', 'TT']:
            labels = labels - sep_point

        pred = output.max(1)[1].type_as(labels)

        return confusion_matrix(labels, pred)


def performance_measure(output, labels, pre=None):
    acc = accuracy(output, labels, pre=pre) * 100

    if len(labels) == 0:
        return np.nan

    if output.shape != labels.shape:
        output = torch.argmax(output, dim=-1)

    # if sep in ['T', 'TH', 'TT']:
    #     labels = labels - sep_point  # [4,5,6] -> [0,1,2]

    macro_F = f1_score(labels.cpu().detach(), output.cpu().detach(), average='macro') * 100
    gmean = geometric_mean_score(labels.cpu().detach(), output.cpu().detach(), average='macro') * 100
    bacc = balanced_accuracy_score(labels.cpu().detach(), output.cpu().detach()) * 100

    return acc, macro_F, gmean, bacc


def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def create_dirs(dirs):
    for dir_tree in dirs:
        sub_dirs = dir_tree.split("/")
        path = ""
        for sub_dir in sub_dirs:
            path = osp.join(path, sub_dir)
            os.makedirs(path, exist_ok=True)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def stu_scheduler(epoch, curriculum_ep=500, func='convex'):
    if func == 'convex':
        return np.cos((epoch * np.pi) / (curriculum_ep * 2))
    elif func == 'concave':
        return np.power(0.99, epoch)
    elif func == 'linear':
        return 1 - (epoch / curriculum_ep)
    elif func == 'composite':
        return (1 / 2) * np.cos((epoch * np.pi) / curriculum_ep) + 1 / 2


def setupt_logger(save_dir, text, filename='log.txt'):
    os.makedirs(save_dir, exist_ok=True)
    logger = logging.getLogger(text)
    # for each in logger.handlers:
    #     logger.removeHandler(each)
    logger.setLevel(4)
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, filename))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    logger.info("======================================================================================")
    return logger


def set_filename(args):
    rec_with_ep_pre = 'True_ep_pre_' + str(args.ep_pre) + '_rw_' + str(args.rw) if args.rec else 'False'

    if args.im_ratio == 1:  # Natural Setting
        results_path = f'./results/natural/{args.dataset}'
        logs_path = f'./logs/natural/{args.dataset}'
        os.makedirs(results_path, exist_ok=True)
        os.makedirs(logs_path, exist_ok=True)

        textname = f'cls_og_{args.cls_og}_rec_{rec_with_ep_pre}_cw_{args.class_weight}_gamma_{args.gamma}_alpha_{args.alpha}_sep_class_{args.sep_class}_degree_{args.sep_degree}_cur_ep_{args.curriculum_ep}_lr_{args.lr}_{args.lr_expert}_dropout_{args.dropout}.txt'
        text = open(f'./results/natural/{args.dataset}/({args.layer}){textname}', 'w')
        file = f'./logs/natural/{args.dataset}/({args.layer})lte4g.txt'

    else:  # Manual Imbalance Setting (0.2, 0.1, 0.05)
        results_path = f'./results/manual/{args.dataset}/{args.im_class_num}/{args.im_ratio}'
        logs_path = f'./logs/manual/{args.dataset}/{args.im_class_num}/{args.im_ratio}'
        os.makedirs(results_path, exist_ok=True)
        os.makedirs(logs_path, exist_ok=True)

        textname = f'cls_og_{args.cls_og}_rec_{rec_with_ep_pre}_cw_{args.class_weight}_gamma_{args.gamma}_alpha_{args.alpha}_sep_class_{args.sep_class}_degree_{args.sep_degree}_cur_ep_{args.curriculum_ep}_lr_{args.lr}_{args.lr_expert}_dropout_{args.dropout}.txt'
        text = open(f'./results/manual/{args.dataset}/{args.im_class_num}/{args.im_ratio}/({args.layer}){textname}',
                    'w')
        file = f'./logs/manual/{args.dataset}/{args.im_class_num}/{args.im_ratio}/({args.layer})lte4g.txt'

    return text, file


def get_index(products):
    edge_index_start = 0
    h_list = []
    t_list = []
    for product in products:
        mol = Chem.MolFromSmiles(product)
        mol_edge_list = []
        for bond in mol.GetBonds():
            h, t = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            mol_edge_list.append([h, t])
        h_list.extend([i[0] + edge_index_start for i in mol_edge_list])
        t_list.extend([i[1] + edge_index_start for i in mol_edge_list])
        edge_index_start += mol.GetNumAtoms()
    edge_index = torch.cat([torch.tensor(h_list).unsqueeze(0), torch.tensor(t_list).unsqueeze(0)], dim=0)
    return edge_index


def get_data_mask(train_test_eval_idx):
    mask_len = train_test_eval_idx[-1][-1]
    data_train_mask, data_test_mask, data_val_mask = torch.zeros(mask_len, dtype=torch.long), torch.zeros(mask_len,
                                                                                                          dtype=torch.long), torch.zeros(
        mask_len, dtype=torch.long)
    data_train_mask[train_test_eval_idx[0][0]:train_test_eval_idx[0][1]] = 1
    data_test_mask[train_test_eval_idx[1][0]:train_test_eval_idx[1][1]] = 1
    data_val_mask[train_test_eval_idx[2][0]:train_test_eval_idx[2][1]] = 1

    return torch.tensor(data_train_mask, dtype=torch.bool), torch.tensor(data_val_mask, dtype=torch.bool), torch.tensor(
        data_test_mask, dtype=torch.bool)


def get_adj(edge_index, all_nodes, add_self_loop=True):
    # edge_index tensor (2,n)
    if add_self_loop:
        # print("Add self loop")
        edge_index = remove_self_loops(edge_index)[0]
        edge_index = add_self_loops(edge_index)[0]
    adj = sp.coo_matrix((np.ones(edge_index.shape[1]), (edge_index[0, :], edge_index[1, :])),
                        shape=(all_nodes, all_nodes), dtype=np.float32)
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj


def get_edgeindex_adj_scope(mols):
    edge_index_start = 0
    h_list = []
    t_list = []
    # scope = []
    for idx, mol in enumerate(mols):
        mol = mol if mol.GetNumAtoms() > 1 else Chem.AddHs(mol)
        mol_edge_list = []
        for bond in mol.GetBonds():
            h, t = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            mol_edge_list.append([h, t])
            mol_edge_list.append([t, h])
        h_list.extend([i[0] + edge_index_start for i in mol_edge_list])
        t_list.extend([i[1] + edge_index_start for i in mol_edge_list])
        # scope.append(torch.tensor([edge_index_start,mol.GetNumAtoms()]).unsqueeze(0))
        edge_index_start += mol.GetNumAtoms()

    edge_index = torch.cat([torch.tensor(h_list).unsqueeze(0), torch.tensor(t_list).unsqueeze(0)], dim=0)
    adj = get_adj(edge_index, all_nodes=edge_index_start)
    # return edge_index,adj,torch.cat(scope,dim=0)
    return adj


def to_device(tensors, device):
    """Converts all inputs to the device used."""
    if isinstance(tensors, list) or isinstance(tensors, tuple):
        tensors = [tensor.to(device, non_blocking=True) if tensor is not None else None for tensor in tensors]
        return tensors
    elif isinstance(tensors, torch.Tensor):
        return tensors.to(device, non_blocking=True)
    else:
        raise ValueError(f"Tensors of type {type(tensors)} unsupported")


def get_input_from_data(data, obj, device='cpu'):
    if obj == 'pre':
        prod_tensors, atom_scopes, labels = data
        return to_device(prod_tensors, device), atom_scopes.to(device), labels.to(device)


def update_scope(scope):
    assert len(scope[0]) == 2
    start_list = [i[0] for i in scope]
    len_list = [i[1] for i in scope]
    new_scope = []
    for i in range(len(scope)):
        if i == 0:
            new_scope.append((start_list[i] + 1, len_list[i]))
            # new_scope.append((start_list[i], len_list[i]))
        else:
            new_scope.append((new_scope[i - 1][0] + len_list[i - 1], len_list[i]))
    return new_scope


def add_scope(scope, side_scope):
    return (scope[0] + side_scope[0], scope[1] + side_scope[1])


def create_pad_tensor(alist):
    max_len = max([len(a) for a in alist])
    new_list = []
    for a in alist:
        pad_len = max_len - len(a)
        try:
            a.extend([0] * pad_len)
            new_list.append(a)
        except:
            a = list(a)
            a.extend([0] * pad_len)
            new_list.append(a)
    return torch.tensor(new_list, dtype=torch.long)


def update_fmess(fmess, start=0):
    i_info = fmess[:, 2:]
    i_idx = fmess[:, :2]
    i_idx_add = torch.ones_like(i_idx) * start
    return torch.cat([i_idx + i_idx_add, i_info], dim=1), torch.max(i_idx)


def update_graph(graph, start=0):
    new_graph = [[x + start for x in y] for y in graph]
    try:
        return new_graph, start + max(list(map(max, filter(None, graph))))
    except:
        return new_graph, start


def update_tensors(prod_tensors):
    new_tensors = []
    for idx, i in enumerate(zip(*prod_tensors)):
        if idx == 0:
            i = torch.cat(i, dim=0)
            i = torch.cat([torch.zeros(size=(1, i.shape[1])), i], dim=0)
            new_tensors.append(i)
        elif idx == 1:
            start = 0
            new_i = []
            for fmess in i:
                new_fmess, number = update_fmess(fmess, start)
                start += number
                new_i.append(new_fmess)
            i = torch.cat(new_i, dim=0)
            i = torch.cat([torch.zeros(size=(1, i.shape[1])), i], dim=0)
            new_tensors.append(i)
        elif idx == 2 or idx == 3:
            start = 0
            new_i = [[]]
            for graph in i:
                new_graph, number = update_graph(graph, start)
                new_i.extend(new_graph)
                start = number
            i = create_pad_tensor(new_i)
            new_tensors.append(i)

    return tuple(new_tensors)


def get_acc(preds, labels):
    accuracy = torch.tensor(0.0)
    for i in range(len(preds)):
        try:
            if torch.argmax(preds[i]).item() == labels[i].item():
                accuracy += 1.0
        except ValueError:
            if torch.argmax(preds[i]).item() == torch.argmax(labels[i]).item():
                accuracy += 1.0
    return accuracy / len(labels)


def snapshot(model, epoch, save_path, acc):
    """
    Saving model w/ its params.
        Get rid of the ONNX Protocal.
    F-string feature new in Python 3.6+ is used.
    """
    os.makedirs(save_path, exist_ok=True)
    timestamp = datetime.now().strftime('%m%d_%H%M')
    save_path = os.path.join(save_path, f'{type(model).__name__}_{timestamp}_{epoch}th_epoch_%.5f.pkl' % acc)
    torch.save(model.state_dict(), save_path)


def str2bool(s):
    if s not in {'False', 'True', 'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return (s == 'True') or (s == 'true')


def get_max_acc_path(path_list):
    acc_list = []
    for path in path_list:
        acc = eval(path.split('_')[-1][:5])
        acc_list.append(acc)
        max_idx = acc_list.index(max(acc_list))
    return path_list[max_idx]


def check_model(model_path, obj='pre'):
    os.makedirs(model_path, exist_ok=True)
    file_names = os.listdir(model_path)
    if obj == 'pre':
        encoder_list, classifier_list = [], []
        for i in file_names:
            if 'GraphFeatEncoder' in i:
                encoder_list.append(os.path.join(model_path, i))
            elif 'Sequential' in i:
                classifier_list.append(os.path.join(model_path, i))
        if len(encoder_list) > 0 and len(classifier_list) > 0:
            return True, (get_max_acc_path(encoder_list), get_max_acc_path(classifier_list))
        else:
            return False, None
    elif obj == 'infer':
        model_list = []
        for i in file_names:
            model_list.append(os.path.join(model_path, i))
        if len(model_list) > 0:
            return True, get_max_acc_path(model_list)
        else:
            return False, None


def get_label_dict():
    label_dict = joblib.load(r'dataset/stage_one/side_smiles_to_label.dict')
    return label_dict


def compute_topn_acc(topn_output, label, Topn_acc):
    for idx, output in enumerate(topn_output):
        if output == label:
            for j in range(idx, len(topn_output)):
                Topn_acc[j] += 1
    return Topn_acc

def compute_class_topn_acc(topn_output, label, class_Topn_acc, rxn_class):
    for idx, output in enumerate(topn_output):
        if output == label:
            for j in range(idx, len(topn_output)):
                class_Topn_acc[rxn_class][j] += 1
    return class_Topn_acc


def is_number(string):
    value = re.compile(r'^[-+]?[0-9]+\.[0-9]+$')
    result = value.match(string)
    return result
