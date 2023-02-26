import pandas as pd
import argparse
import os
import sys
cur_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(cur_path, '../'))
from Codes.mol_info import Label_Vocab, RxnElement, pack_graph_feats
import pickle
import torch

torch.multiprocessing.set_sharing_strategy('file_system')
from rdkit import Chem
from tqdm import tqdm
import multiprocessing



def multi_process(product_info):
    global data_idx, with_class
    label = product_info['label']
    product = product_info['product']
    rxn_class = product_info['rxn_class']
    product_mol = Chem.MolFromSmiles(product)
    prod_graph = RxnElement(mol=Chem.MolFromSmiles(product))
    prod_inputs = pack_graph_feats(prod_graph, directed=True, use_rxn_class=with_class, rxn_class=rxn_class)

    rxn_data = {
        'product_mol': product_mol,
        'label': label,
        'prod_inputs': prod_inputs,
        'data_idx': data_idx,
    }
    data_idx += 1

    return rxn_data


def create_pad_tensor(alist):
    max_len = max([len(a) for a in alist])
    for a in alist:
        pad_len = max_len - len(a)
        a.extend([0] * pad_len)
    return torch.tensor(alist, dtype=torch.long)


def preprocess(save_dir, products, labels, rxns_class):
    """
    preprocess reaction data to extract graph adjacency matrix and features
    """
    global data_idx, with_class

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    products_info = [{
        "product": product,
        "label": label,
        'rxn_class': rxn_class
    } for product, label, rxn_class in zip(products, labels, rxns_class)]

    if len(products_info) < args.max_number:
        with torch.multiprocessing.Pool(processes=args.processes) as pool:
            rxn_datas = list(tqdm(pool.imap(multi_process, products_info), total=len(products_info)))
        pool.close()
        pool.join()
    else:
        rxn_datas = []
        for index in tqdm(range(len(products))):
            product = products[index]
            product_mol = Chem.MolFromSmiles(product)
            prod_graph = RxnElement(mol=Chem.MolFromSmiles(product))
            prod_inputs = pack_graph_feats(prod_graph, directed=True, use_rxn_class=with_class,
                                           rxn_class=rxns_class[index])
            rxn_data = {
                'product_mol': product_mol,
                'label': labels[index],
                'prod_inputs': prod_inputs,
                'data_idx': data_idx,
            }
            rxn_datas.append(rxn_data)
            data_idx += 1
    save_name = '_with_class' if with_class else ''

    with open(os.path.join(save_dir, 'side_rxn_datas%s.pkl' % save_name), 'wb') as f:
        pickle.dump(rxn_datas, f)



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='USPTO_50K',
                        help='dataset: USPTO_50K or USPTO-full')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--max_number', type=int, default=30000)
    parser.add_argument('--with_class', action='store_true')
    parser.add_argument("--exp_id", type=str, default="")
    parser.add_argument("--processes", type=int, default=multiprocessing.cpu_count())

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    os.chdir(os.path.join(os.getcwd(), '../%s/dataset/stage_one' % args.exp_id))
    print('preprocessing dataset {}...'.format(args.dataset))

    data_idx = 0
    with_class = args.with_class

    for data_set in ['train', 'test', 'eval']:
        save_dir = os.path.join(data_set)
        csv_path = os.path.join(data_set, 'side_product_rxn_' + data_set + '.csv')
        csv = pd.read_csv(csv_path)
        products = csv['product'].tolist()
        labels = csv['label'].tolist()
        side_products = csv['sideprod'].tolist()
        rxns_class = csv['class'].tolist()

        preprocess(
            save_dir,
            products,
            labels,
            rxns_class
        )
