#Input your own SMILES for side product prediction and
# data enhancement in preparation for input stagw two


import os
import argparse
import numpy as np
import multiprocessing
from tqdm import tqdm
import re
import random
import itertools
import joblib

from Codes.utils import snapshot, str2bool, get_label_dict, check_model, compute_topn_acc
from Codes.model import SideModel, Trainer
from Codes.data import RetroTestDatasets
from Codes.data import collate_pre, collate_emb
from Codes.embedder import Embedder
from Codes.mol_info import Label_Vocab, RxnElement, pack_graph_feats
from preprocess.generate_inputs_for_stage_two import get_root_id, get_cano_map_number, clear_map_canonical_smiles,smi_tokenizer
from rdkit import Chem

import torch
from torch.utils.data import DataLoader

def add_map(product):
    mol = Chem.MolFromSmiles(product)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx())
    return Chem.MolToSmiles(mol)


def multi_process(data):
    pt = re.compile(r':(\d+)]')
    product = data['product']
    sideprod = data['sideprod']
    augmentation = data['augmentation']
    pro_mol = Chem.MolFromSmiles(product)
    """checking data quality"""
    return_status = {
        "status": 0,
        "src_data": [],
        "sideprod_src_data": [],
    }
    pro_atom_map_numbers = list(map(int, re.findall(r"(?<=:)\d+", product)))
    if len(pro_atom_map_numbers) == 0:
        product = add_map(product)
        pro_atom_map_numbers = list(map(int, re.findall(r"(?<=:)\d+", product)))
        pro_mol = Chem.MolFromSmiles(product)
    product_roots = [-1]
    max_times = len(pro_atom_map_numbers)
    times = min(augmentation, max_times)
    if times < augmentation:
        product_roots.extend(pro_atom_map_numbers)
        product_roots.extend(random.choices(product_roots, k=augmentation - len(product_roots)))
    else:
        while len(product_roots) < times:
            product_roots.append(random.sample(pro_atom_map_numbers, 1)[0])
            if product_roots[-1] in product_roots[:-1]:
                product_roots.pop()
    times = len(product_roots)
    assert times == augmentation

    for k in range(times):
        pro_root_atom_map = product_roots[k]
        pro_root = get_root_id(pro_mol, root_map_number=pro_root_atom_map)
        cano_atom_map = get_cano_map_number(product, root=pro_root)
        if cano_atom_map is None:
            return_status["status"] = "error_mapping"
            return return_status
        pro_smi = clear_map_canonical_smiles(product, canonical=True, root=pro_root)
        product_tokens = smi_tokenizer(pro_smi)
        return_status['src_data'].append(product_tokens)

    assert len(return_status['src_data']) == data['augmentation']

    if sideprod not in ['<eos>']:
        for src in return_status['src_data']:
            try:
                side_prob_dict = lcs_side_prob_dict[sideprod]
                weights = list(side_prob_dict.values())
                random_sideprod = \
                    random.choices(list(side_prob_dict.keys()), weights=weights, k=1)[-1]
            except:
                random_sideprod = sideprod
            sideprod_src = src.replace(' ', '') + '.' + random_sideprod
            return_status['sideprod_src_data'].append(smi_tokenizer(sideprod_src))
    else:
        return_status['sideprod_src_data'] = return_status['src_data']

    return return_status


def run_side_model(args):
    self = Embedder(args)
    args.nclass = self.nclass
    args.sep_point = self.sep_point

    model = SideModel(args, device=self.args.device).to(self.args.device)
    trainer = Trainer(model, args, device=self.args.device)

    print(f"Converting model to device: {self.args.device}")
    print("Param Count: ", sum([x.nelement() for x in model.parameters()]) / 10 ** 6, "M", flush=True)

    data_path = args.data_path
    smiles_list = []
    with open(data_path, 'r+') as r:
        for line in r.readlines():
            smiles_list.append(line.strip())
    infos = []
    for smiles in smiles_list:
        prod_graph = RxnElement(mol=Chem.MolFromSmiles(smiles))
        prod_inputs = pack_graph_feats(prod_graph, directed=True, use_rxn_class=False)
        info = {
                'product_mol': Chem.MolFromSmiles(smiles),
                'label': 0,
                'prod_inputs': prod_inputs,
                'data_idx': 0,
                }
        infos.append(info)
    test_data = RetroTestDatasets(infos)
    test_dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=0,
                                 collate_fn=collate_pre)
    pre_model_state, model_path_list = check_model(os.path.join(args.save_root, 'pre_model'), obj='pre')
    if pre_model_state and args.load_pre:
        print('Load Pretrain Encoder !!!!', flush=True)
        print('Please ignore the acc of the display')
        trainer.load_state_dict(model_path_list[0], model_path_list[1])
        log_outputs, _ = trainer.test_step(test_dataloader, epoch='infer_test', data_name='test')
        label_dict = get_label_dict()
        Topn_Smiles = []
        Topn_Scores = []

        for idx, output in enumerate(log_outputs):
            value, index = torch.topk(output, k=args.topn)
            Topn_Scores.extend(value.cpu().numpy().tolist())
            Topn_Smiles.extend([label_dict.get_elem(i) for i in index.cpu().numpy().tolist()])
        # with open('test_top%s_smiles.txt' % args.topn, 'w+') as f:
        #     for smiles in Topn_Smiles:
        #         f.write('{}\n'.format(smiles))

        with open('test_top%s_scores.txt' % args.topn, 'w+') as f:
            for scores in Topn_Scores:
                f.write('{}\n'.format(scores))

        prod_list = np.repeat(np.array(smiles_list), args.topn).tolist()
        sideprod_list = Topn_Smiles
        augmentation = args.augmentation
        data = [{
            "product": p,
            "sideprod": sp,
            "augmentation": augmentation,
        } for p, sp in zip(prod_list, sideprod_list)]

        with multiprocessing.Pool(processes=1) as pool:
            results = list(tqdm(pool.imap(multi_process, data), total=len(data)))
        pool.close()
        pool.join()
        sideprod_src_data = []
        for result in tqdm(results):
            if result['status'] != 0:
                continue
            sideprod_src_data.extend(result['sideprod_src_data'])

        with open('sideprod_src-test.txt', 'w') as f:
            for sideprod_src in sideprod_src_data:
                f.write('{}\n'.format(sideprod_src))

    else:
        print('Chose True path !!!', flush=True)




def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_atom_feat', type=int, default=98)
    parser.add_argument('--n_bond_feat', type=int, default=6)
    parser.add_argument('--nhid', type=int, default=300)
    parser.add_argument('--depth', type=int, default=10)
    parser.add_argument('--dropout_mpn', type=int, default=0.15)
    parser.add_argument('--mlp_size', type=int, default=300)
    parser.add_argument('--mpn_size', type=int, default=300)
    parser.add_argument('--dropout_mlp', type=int, default=0.3)
    parser.add_argument('--nhead', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument("--patience", default=10, type=float)
    parser.add_argument("--anneal_rate", default=0.9, type=float)
    parser.add_argument("--metric_thresh", default=0.01, type=float)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--alpha', type=float, default=0.9)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument('--lr_cls', type=float, default=0.00001)

    parser.add_argument("--augmentation", type=int, default=20)
    parser.add_argument("--num_workers", default=0)
    parser.add_argument("--seed", default=916)
    parser.add_argument("--sep_degree", default=150)
    parser.add_argument('--im_ratio', type=float, default=0.01)
    parser.add_argument('--sep_class', type=str, default='pareto_19')
    parser.add_argument("--im_class_num", default=150)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument("--load_pre", default=False, type=str2bool)
    parser.add_argument('--topn', type=float, default=10)
    parser.add_argument('--gpu', type=str, default=0)

    parser.add_argument("--exp_id", type=str, default="")
    parser.add_argument("--save_root", default=r'save_model/stage_one/without_class')
    parser.add_argument("--data_path", default='')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    args.save_root = os.path.join(args.exp_id, args.save_root)
    args.data_path = r'f7-7.smi' if args.data_path == '' else args.data_path
    print(args)
    lcs_side_prob_dict_file = r'dataset/stage_two/lcs_side_prob.dict'
    lcs_side_prob_dict = joblib.load(lcs_side_prob_dict_file)
    run_side_model(args)
