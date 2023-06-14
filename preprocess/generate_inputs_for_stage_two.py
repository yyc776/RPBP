import joblib
import numpy as np
import pandas as pd
import argparse
import os
import re
import random
import textdistance
import multiprocessing
import itertools
from string import digits

from rdkit import Chem
from tqdm import tqdm

from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')


def smi_tokenizer(smi):
    pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    assert smi == ''.join(tokens)
    return ' '.join(tokens)


def clear_map_canonical_smiles(smi, canonical=True, root=-1):
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        for atom in mol.GetAtoms():
            if atom.HasProp('molAtomMapNumber'):
                atom.ClearProp('molAtomMapNumber')
        return Chem.MolToSmiles(mol, isomericSmiles=True, rootedAtAtom=root, canonical=canonical)
    else:
        return smi


def get_cano_map_number(smi, root=-1):
    atommap_mol = Chem.MolFromSmiles(smi)
    canonical_mol = Chem.MolFromSmiles(clear_map_canonical_smiles(smi, root=root))
    cano2atommapIdx = atommap_mol.GetSubstructMatch(canonical_mol)
    correct_mapped = [canonical_mol.GetAtomWithIdx(i).GetSymbol() == atommap_mol.GetAtomWithIdx(index).GetSymbol() for
                      i, index in enumerate(cano2atommapIdx)]
    atom_number = len(canonical_mol.GetAtoms())
    if np.sum(correct_mapped) < atom_number or len(cano2atommapIdx) < atom_number:
        cano2atommapIdx = [0] * atom_number
        atommap2canoIdx = canonical_mol.GetSubstructMatch(atommap_mol)
        if len(atommap2canoIdx) != atom_number:
            return None
        for i, index in enumerate(atommap2canoIdx):
            cano2atommapIdx[index] = i
    id2atommap = [atom.GetAtomMapNum() for atom in atommap_mol.GetAtoms()]

    return [id2atommap[cano2atommapIdx[i]] for i in range(atom_number)]


def get_root_id(mol, root_map_number):
    root = -1
    for i, atom in enumerate(mol.GetAtoms()):
        if atom.GetAtomMapNum() == root_map_number:
            root = i
            break
    return root


def analyse_save(lcs_byproduct):
    cano_lcs_byproduct = [Chem.MolToSmiles(Chem.MolFromSmiles(byproduct)) for byproduct in lcs_byproduct]
    lcs_byproduct_df = pd.DataFrame()
    lcs_byproduct_df['cano_lcs_byproduct'] = cano_lcs_byproduct
    lcs_byproduct_df['lcs_byproduct'] = lcs_byproduct
    byproduct_prob_dict = {}
    groups = lcs_byproduct_df.groupby(['cano_lcs_byproduct'])
    for group in groups:
        cano_lcs = group[0]
        byproduct_prob_dict[cano_lcs] = {}
        lcs_byproduct_list = group[1]['lcs_byproduct'].tolist()
        byproduct_list = set(lcs_byproduct_list)
        for byproduct in byproduct_list:
            byproduct_prob_dict[cano_lcs][byproduct] = lcs_byproduct_list.count(byproduct) / len(lcs_byproduct_list)

    joblib.dump(byproduct_prob_dict, os.path.join(r'../', args.task_dataset, 'dataset/stage_two/lcs_byproduct_prob.dict'))
    return byproduct_prob_dict


def prob_sample_train(args):
    save_dir = os.path.join(r'../', args.task_dataset, 'dataset/stage_two')
    csv_path = os.path.join(r'../', args.task_dataset, 'dataset/stage_one/train/byproduct_rxn_train.csv')
    csv = pd.read_csv(csv_path)
    reaction_list = list(csv["reactants>reagents>production"])
    byproduct_list = list(csv["byproduct"])
    reactant_smarts_list = list(
        map(lambda x: x.split('>')[0], reaction_list))
    reactant_smarts_list = list(
        map(lambda x: x.split(' ')[0], reactant_smarts_list))
    product_smarts_list = list(
        map(lambda x: x.split('>')[2], reaction_list))
    product_smarts_list = list(
        map(lambda x: x.split(' ')[0], product_smarts_list))
    reactants = reactant_smarts_list
    products = product_smarts_list
    byproducts = byproduct_list

    data = [{
        "reactant": i,
        "product": j,
        "byproduct": n,
        "augmentation": args.augmentation,
        "root_aligned": not args.canonical,
    } for i, j, n in zip(reactants, products, byproducts)]

    lcs_byproduct = []
    tgt_lcs_data = []
    byproduct_src_lcs_data = []
    skip_dict = {
        'invalid_p': 0,
        'invalid_r': 0,
        'small_p': 0,
        'small_r': 0,
        'error_mapping': 0,
        'error_mapping_p': 0,
        'empty_p': 0,
        'empty_r': 0,
    }

    processes = multiprocessing.cpu_count() if args.processes < 0 else args.processes
    with multiprocessing.Pool(processes=processes) as pool:
        results = list(tqdm(pool.imap(prob_sample_multi_process_6, data), total=len(data)))
    pool.close()
    pool.join()

    for result in tqdm(results):
        if result['status'] != 0:
            skip_dict[result['status']] += 1
            continue
        lcs_byproduct.extend(result['lcs_byproduct'])
        byproduct_src_lcs_data.extend(result['byproduct_src_data'])
        tgt_lcs_data.extend(result['tgt_data'])
    os.makedirs(save_dir,exist_ok=True)
    with open(os.path.join(save_dir, 'tgt_lcs-train.txt'), 'w') as f:
        for tgt in tgt_lcs_data:
            f.write('{}\n'.format(tgt))

    with open(
            os.path.join(save_dir, 'byproduct_src_lcs-train.txt'), 'w') as f:
        for byproduct_src_lcs in byproduct_src_lcs_data:
            f.write('{}\n'.format(byproduct_src_lcs))
    return lcs_byproduct


def prob_sample_multi_process(data):
    pt = re.compile(r':(\d+)]')
    product = data['product']
    reactant = data['reactant']
    byproduct = data['byproduct']
    augmentation = data['augmentation']
    pro_mol = Chem.MolFromSmiles(product)
    rea_mol = Chem.MolFromSmiles(reactant)
    """checking data quality"""
    rids = sorted(re.findall(pt, reactant))
    pids = sorted(re.findall(pt, product))
    return_status = {
        "status": 0,
        "src_data": [],
        "tgt_data": [],
        "byproduct_src_data": [],
        "edit_distance": 0,
        "byproduct_edit_distance": 0,
        "lcs_byproduct": [],
    }

    if "" == product:
        return_status["status"] = "empty_p"
    if "" == reactant:
        return_status["status"] = "empty_r"
    if rea_mol is None:
        return_status["status"] = "invalid_r"
    if len(rea_mol.GetAtoms()) < 5:
        return_status["status"] = "small_r"
    if pro_mol is None:
        return_status["status"] = "invalid_p"
    # if len(pro_mol.GetAtoms()) == 1:
    #     return_status["status"] = "small_p"
    if not all([a.HasProp('molAtomMapNumber') for a in pro_mol.GetAtoms()]):
        return_status["status"] = "error_mapping_p"
    """finishing checking data quality"""

    if return_status['status'] == 0:
        pro_atom_map_numbers = list(map(int, re.findall(r"(?<=:)\d+", product)))
        reactant = reactant.split(".")
        if data['root_aligned']:
            reversable = False  # no shuffle
            # augmentation = 100
            if augmentation == 999:
                product_roots = pro_atom_map_numbers
                times = len(product_roots)
            else:
                product_roots = [-1]
                # reversable = len(reactant) > 1

                max_times = len(pro_atom_map_numbers)
                times = min(augmentation, max_times)
                if times < augmentation:  # times = max_times
                    product_roots.extend(pro_atom_map_numbers)
                    product_roots.extend(random.choices(product_roots, k=augmentation - len(product_roots)))
                else:  # times = augmentation
                    while len(product_roots) < times:
                        product_roots.append(random.sample(pro_atom_map_numbers, 1)[0])
                        # pro_atom_map_numbers.remove(product_roots[-1])
                        if product_roots[-1] in product_roots[:-1]:
                            product_roots.pop()
                times = len(product_roots)
                assert times == augmentation
                if reversable:
                    times = int(times / 2)
            # candidates = []
            for k in range(times):
                pro_root_atom_map = product_roots[k]
                pro_root = get_root_id(pro_mol, root_map_number=pro_root_atom_map)
                cano_atom_map = get_cano_map_number(product, root=pro_root)
                if cano_atom_map is None:
                    return_status["status"] = "error_mapping"
                    return return_status
                pro_smi = clear_map_canonical_smiles(product, canonical=True, root=pro_root)
                aligned_reactants = []
                aligned_reactants_order = []
                rea_atom_map_numbers = [list(map(int, re.findall(r"(?<=:)\d+", rea))) for rea in reactant]
                used_indices = []
                for i, rea_map_number in enumerate(rea_atom_map_numbers):
                    for j, map_number in enumerate(cano_atom_map):
                        # select mapping reactans
                        if map_number in rea_map_number:
                            rea_root = get_root_id(Chem.MolFromSmiles(reactant[i]), root_map_number=map_number)
                            rea_smi = clear_map_canonical_smiles(reactant[i], canonical=True, root=rea_root)
                            aligned_reactants.append(rea_smi)
                            aligned_reactants_order.append(j)
                            used_indices.append(i)
                            break
                sorted_reactants = sorted(list(zip(aligned_reactants, aligned_reactants_order)), key=lambda x: x[1])
                aligned_reactants = [item[0] for item in sorted_reactants]
                reactant_smi = ".".join(aligned_reactants)
                product_tokens = smi_tokenizer(pro_smi)
                reactant_tokens = smi_tokenizer(reactant_smi)

                return_status['src_data'].append(product_tokens)
                return_status['tgt_data'].append(reactant_tokens)

                if reversable:
                    aligned_reactants.reverse()
                    reactant_smi = ".".join(aligned_reactants)
                    product_tokens = smi_tokenizer(pro_smi)
                    reactant_tokens = smi_tokenizer(reactant_smi)
                    return_status['src_data'].append(product_tokens)
                    return_status['tgt_data'].append(reactant_tokens)
            assert len(return_status['src_data']) == data['augmentation']
        else:
            cano_product = clear_map_canonical_smiles(product)
            cano_reactanct = ".".join([clear_map_canonical_smiles(rea) for rea in reactant if
                                       len(set(map(int, re.findall(r"(?<=:)\d+", rea))) & set(
                                           pro_atom_map_numbers)) > 0])
            return_status['src_data'].append(smi_tokenizer(cano_product))
            return_status['tgt_data'].append(smi_tokenizer(cano_reactanct))
            pro_mol = Chem.MolFromSmiles(cano_product)
            rea_mols = [Chem.MolFromSmiles(rea) for rea in cano_reactanct.split(".")]
            for i in range(int(augmentation - 1)):
                pro_smi = Chem.MolToSmiles(pro_mol, doRandom=True)
                rea_smi = [Chem.MolToSmiles(rea_mol, doRandom=True) for rea_mol in rea_mols]
                rea_smi = ".".join(rea_smi)
                return_status['src_data'].append(smi_tokenizer(pro_smi))
                return_status['tgt_data'].append(smi_tokenizer(rea_smi))
        if byproduct not in ['<eos>']:
            byproduct_mol = Chem.MolFromSmiles(byproduct)
            for src, tgt in zip(return_status['src_data'], return_status['tgt_data']):
                # byproduct_edit_distances = []
                min_dis = 10000
                min_byproduct = ''
                min_byproduct_list = []
                max_lcs_list = []
                if '.' in byproduct:
                    byproduct_list = byproduct.split('.')
                    lenth = len(byproduct_list)
                    byproduct_smi_list = [[] for _ in range(lenth)]
                    for n in range(lenth):
                        for n1 in range(Chem.MolFromSmiles(byproduct_list[n]).GetNumAtoms()):
                            byproduct_smi_list[n].append(
                                clear_map_canonical_smiles(byproduct_list[n], canonical=True, root=n1))
                    byproduct_comb_list = list(itertools.product(*byproduct_smi_list))
                    byproduct_comb_list = [('.').join(list(byproduct_comb)) for byproduct_comb in byproduct_comb_list]
                    for byproduct_smi in byproduct_comb_list:
                        byproduct_src = src.split()
                        byproduct_src.append('.')
                        byproduct_src.extend(smi_tokenizer(byproduct_smi).split())
                        distance = textdistance.levenshtein.distance(byproduct_src, tgt.split())
                        if distance <= min_dis:
                            min_dis = distance
                            min_byproduct = byproduct_smi
                    min_byproduct_list.append(min_byproduct)
                    for i in min_byproduct_list:
                        lcs_lenth = find_lcs_len(i, tgt.replace(' ', ''))
                        max_lcs_list.append(lcs_lenth)
                    assert len(max_lcs_list) == len(min_byproduct_list)
                    min_byproduct = min_byproduct_list[max_lcs_list.index(max(max_lcs_list))]
                    # min_byproduct = random.choice(byproduct_comb_list)
                    byproduct_src = src.replace(' ', '') + '.' + min_byproduct
                    return_status['byproduct_src_data'].append(smi_tokenizer(byproduct_src))
                    # byproduct_edit_distances.append(min_dis)
                    return_status['lcs_byproduct'].append(min_byproduct)
                else:
                    for i in range(byproduct_mol.GetNumAtoms()):
                        byproduct_smi = clear_map_canonical_smiles(byproduct, canonical=True, root=i)
                        byproduct_src = src.split()
                        byproduct_src.append('.')
                        byproduct_src.extend(smi_tokenizer(byproduct_smi).split())
                        distance = textdistance.levenshtein.distance(byproduct_src, tgt.split())
                        if distance < min_dis:
                            min_dis = distance
                            min_byproduct = byproduct_smi
                        elif distance == min_dis:
                            min_byproduct_list.append(byproduct_smi)
                    min_byproduct_list.append(min_byproduct)
                    for i in min_byproduct_list:
                        lcs_lenth = find_lcs_len(i, tgt.replace(' ', ''))
                        max_lcs_list.append(lcs_lenth)
                    assert len(max_lcs_list) == len(min_byproduct_list)
                    min_byproduct = min_byproduct_list[max_lcs_list.index(max(max_lcs_list))]

                    # byproduct_list = [clear_map_canonical_smiles(byproduct, canonical=True, root=n1) for n1 in range(Chem.MolFromSmiles(byproduct).GetNumAtoms())]
                    # min_byproduct = random.choice(byproduct_list)
                    byproduct_src = src.replace(' ', '') + '.' + min_byproduct
                    return_status['byproduct_src_data'].append(smi_tokenizer(byproduct_src))
                    return_status['lcs_byproduct'].append(min_byproduct)
                    # byproduct_edit_distances.append(min_dis)
                # assert min_dis < 10000
                # return_status['byproduct_edit_distance'] = np.mean(byproduct_edit_distances)
        else:
            return_status['byproduct_src_data'] = return_status['src_data']
        edit_distances = []
        for src, tgt in zip(return_status['src_data'], return_status['tgt_data']):
            edit_distances.append(textdistance.levenshtein.distance(src.split(), tgt.split()))

        return_status['edit_distance'] = np.mean(edit_distances)
    return return_status

def prob_sample_multi_process_2(data):
    pt = re.compile(r':(\d+)]')
    product = data['product']
    reactant = data['reactant']
    byproduct = data['byproduct']
    augmentation = data['augmentation']
    pro_mol = Chem.MolFromSmiles(product)
    rea_mol = Chem.MolFromSmiles(reactant)
    """checking data quality"""
    rids = sorted(re.findall(pt, reactant))
    pids = sorted(re.findall(pt, product))
    return_status = {
        "status": 0,
        "src_data": [],
        "tgt_data": [],
        "byproduct_src_data": [],
        "edit_distance": 0,
        "byproduct_edit_distance": 0,
        "lcs_byproduct": [],
    }

    if "" == product:
        return_status["status"] = "empty_p"
    if "" == reactant:
        return_status["status"] = "empty_r"
    if rea_mol is None:
        return_status["status"] = "invalid_r"
    if len(rea_mol.GetAtoms()) < 5:
        return_status["status"] = "small_r"
    if pro_mol is None:
        return_status["status"] = "invalid_p"
    # if len(pro_mol.GetAtoms()) == 1:
    #     return_status["status"] = "small_p"
    if not all([a.HasProp('molAtomMapNumber') for a in pro_mol.GetAtoms()]):
        return_status["status"] = "error_mapping_p"
    """finishing checking data quality"""

    if return_status['status'] == 0:
        pro_atom_map_numbers = list(map(int, re.findall(r"(?<=:)\d+", product)))
        reactant = reactant.split(".")
        if data['root_aligned']:
            reversable = False  # no shuffle
            # augmentation = 100
            if augmentation == 999:
                product_roots = pro_atom_map_numbers
                times = len(product_roots)
            else:
                product_roots = [-1]
                # reversable = len(reactant) > 1

                max_times = len(pro_atom_map_numbers)
                times = min(augmentation, max_times)
                if times < augmentation:  # times = max_times
                    product_roots.extend(pro_atom_map_numbers)
                    product_roots.extend(random.choices(product_roots, k=augmentation - len(product_roots)))
                else:  # times = augmentation
                    while len(product_roots) < times:
                        product_roots.append(random.sample(pro_atom_map_numbers, 1)[0])
                        # pro_atom_map_numbers.remove(product_roots[-1])
                        if product_roots[-1] in product_roots[:-1]:
                            product_roots.pop()
                times = len(product_roots)
                assert times == augmentation
                if reversable:
                    times = int(times / 2)
            # candidates = []
            for k in range(times):
                pro_root_atom_map = product_roots[k]
                pro_root = get_root_id(pro_mol, root_map_number=pro_root_atom_map)
                cano_atom_map = get_cano_map_number(product, root=pro_root)
                if cano_atom_map is None:
                    return_status["status"] = "error_mapping"
                    return return_status
                pro_smi = clear_map_canonical_smiles(product, canonical=True, root=pro_root)
                aligned_reactants = []
                aligned_reactants_order = []
                rea_atom_map_numbers = [list(map(int, re.findall(r"(?<=:)\d+", rea))) for rea in reactant]
                used_indices = []
                for i, rea_map_number in enumerate(rea_atom_map_numbers):
                    for j, map_number in enumerate(cano_atom_map):
                        # select mapping reactans
                        if map_number in rea_map_number:
                            rea_root = get_root_id(Chem.MolFromSmiles(reactant[i]), root_map_number=map_number)
                            rea_smi = clear_map_canonical_smiles(reactant[i], canonical=True, root=rea_root)
                            aligned_reactants.append(rea_smi)
                            aligned_reactants_order.append(j)
                            used_indices.append(i)
                            break
                sorted_reactants = sorted(list(zip(aligned_reactants, aligned_reactants_order)), key=lambda x: x[1])
                aligned_reactants = [item[0] for item in sorted_reactants]
                reactant_smi = ".".join(aligned_reactants)
                product_tokens = smi_tokenizer(pro_smi)
                reactant_tokens = smi_tokenizer(reactant_smi)

                return_status['src_data'].append(product_tokens)
                return_status['tgt_data'].append(reactant_tokens)

            assert len(return_status['src_data']) == data['augmentation']
        else:
            cano_product = clear_map_canonical_smiles(product)
            cano_reactanct = ".".join([clear_map_canonical_smiles(rea) for rea in reactant if
                                       len(set(map(int, re.findall(r"(?<=:)\d+", rea))) & set(
                                           pro_atom_map_numbers)) > 0])
            return_status['src_data'].append(smi_tokenizer(cano_product))
            return_status['tgt_data'].append(smi_tokenizer(cano_reactanct))
            pro_mol = Chem.MolFromSmiles(cano_product)
            rea_mols = [Chem.MolFromSmiles(rea) for rea in cano_reactanct.split(".")]
            for i in range(int(augmentation - 1)):
                pro_smi = Chem.MolToSmiles(pro_mol, doRandom=True)
                rea_smi = [Chem.MolToSmiles(rea_mol, doRandom=True) for rea_mol in rea_mols]
                rea_smi = ".".join(rea_smi)
                return_status['src_data'].append(smi_tokenizer(pro_smi))
                return_status['tgt_data'].append(smi_tokenizer(rea_smi))
        if byproduct not in ['<eos>']:
            byproduct_mol = Chem.MolFromSmiles(byproduct)
            for src, tgt in zip(return_status['src_data'], return_status['tgt_data']):
                # byproduct_edit_distances = []
                # min_dis = 10000
                # min_byproduct = ''
                # min_byproduct_list = []
                max_lcs_list = []
                if '.' in byproduct:
                    byproduct_list = byproduct.split('.')
                    lenth = len(byproduct_list)
                    byproduct_smi_list = [[] for _ in range(lenth)]
                    for n in range(lenth):
                        for n1 in range(Chem.MolFromSmiles(byproduct_list[n]).GetNumAtoms()):
                            byproduct_smi_list[n].append(
                                clear_map_canonical_smiles(byproduct_list[n], canonical=True, root=n1))
                    byproduct_comb_list = list(itertools.product(*byproduct_smi_list))
                    byproduct_comb_list = [('.').join(list(byproduct_comb)) for byproduct_comb in byproduct_comb_list]
                    # for byproduct_smi in byproduct_comb_list:
                    #     byproduct_src = src.split()
                    #     byproduct_src.append('.')
                    #     byproduct_src.extend(smi_tokenizer(byproduct_smi).split())
                    #     distance = textdistance.levenshtein.distance(byproduct_src, tgt.split())
                    #     if distance <= min_dis:
                    #         min_dis = distance
                    #         min_byproduct = byproduct_smi
                    # min_byproduct_list.append(min_byproduct)
                    for i in byproduct_comb_list:
                        # lcs_lenth = find_lcs_len(i, tgt.replace(' ', ''))
                        lcs_lenth = textdistance.lcsseq.distance(i, tgt.replace(' ', ''))
                        max_lcs_list.append(lcs_lenth)
                    # assert len(max_lcs_list) == len(min_byproduct_list)
                    lcs_byproduct = byproduct_comb_list[max_lcs_list.index(max(max_lcs_list))]
                    # min_byproduct = random.choice(byproduct_comb_list)
                    byproduct_src = src.replace(' ', '') + '.' + lcs_byproduct
                    return_status['byproduct_src_data'].append(smi_tokenizer(byproduct_src))
                    # byproduct_edit_distances.append(min_dis)
                    return_status['lcs_byproduct'].append(lcs_byproduct)
                else:
                    byproduct_list = []
                    for i in range(byproduct_mol.GetNumAtoms()):
                        byproduct_smi = clear_map_canonical_smiles(byproduct, canonical=True, root=i)
                        byproduct_list.append(byproduct_smi)
                        # byproduct_src = src.split()
                        # byproduct_src.append('.')
                        # byproduct_src.extend(smi_tokenizer(byproduct_smi).split())
                        # distance = textdistance.levenshtein.distance(byproduct_src, tgt.split())
                        # if distance < min_dis:
                        #     min_dis = distance
                        #     min_byproduct = byproduct_smi
                        # elif distance == min_dis:
                        #     min_byproduct_list.append(byproduct_smi)
                    # min_byproduct_list.append(min_byproduct)
                    for i in byproduct_list:
                        lcs_lenth = textdistance.lcsstr.distance(i, tgt.replace(' ', ''))
                        max_lcs_list.append(lcs_lenth)
                    # assert len(max_lcs_list) == len(min_byproduct_list)
                    lcs_byproduct = byproduct_list[max_lcs_list.index(max(max_lcs_list))]

                    # byproduct_list = [clear_map_canonical_smiles(byproduct, canonical=True, root=n1) for n1 in range(Chem.MolFromSmiles(byproduct).GetNumAtoms())]
                    # min_byproduct = random.choice(byproduct_list)
                    byproduct_src = src.replace(' ', '') + '.' + lcs_byproduct
                    return_status['byproduct_src_data'].append(smi_tokenizer(byproduct_src))
                    return_status['lcs_byproduct'].append(lcs_byproduct)
                    # byproduct_edit_distances.append(min_dis)
                # assert min_dis < 10000
                # return_status['byproduct_edit_distance'] = np.mean(byproduct_edit_distances)
        else:
            return_status['byproduct_src_data'] = return_status['src_data']
        edit_distances = []
        for src, tgt in zip(return_status['src_data'], return_status['tgt_data']):
            edit_distances.append(textdistance.levenshtein.distance(src.split(), tgt.split()))

        return_status['edit_distance'] = np.mean(edit_distances)
    return return_status

def prob_sample_multi_process_3(data):
    pt = re.compile(r':(\d+)]')
    product = data['product']
    reactant = data['reactant']
    byproduct = data['byproduct']
    augmentation = data['augmentation']
    pro_mol = Chem.MolFromSmiles(product)
    rea_mol = Chem.MolFromSmiles(reactant)
    """checking data quality"""
    rids = sorted(re.findall(pt, reactant))
    pids = sorted(re.findall(pt, product))
    return_status = {
        "status": 0,
        "src_data": [],
        "tgt_data": [],
        "byproduct_src_data": [],
        "edit_distance": 0,
        "byproduct_edit_distance": 0,
        "lcs_byproduct": [],
    }

    if "" == product:
        return_status["status"] = "empty_p"
    if "" == reactant:
        return_status["status"] = "empty_r"
    if rea_mol is None:
        return_status["status"] = "invalid_r"
    if len(rea_mol.GetAtoms()) < 5:
        return_status["status"] = "small_r"
    if pro_mol is None:
        return_status["status"] = "invalid_p"
    # if len(pro_mol.GetAtoms()) == 1:
    #     return_status["status"] = "small_p"
    if not all([a.HasProp('molAtomMapNumber') for a in pro_mol.GetAtoms()]):
        return_status["status"] = "error_mapping_p"
    """finishing checking data quality"""

    if return_status['status'] == 0:
        pro_atom_map_numbers = list(map(int, re.findall(r"(?<=:)\d+", product)))
        reactant = reactant.split(".")
        if data['root_aligned']:
            reversable = False  # no shuffle
            # augmentation = 100
            if augmentation == 999:
                product_roots = pro_atom_map_numbers
                times = len(product_roots)
            else:
                product_roots = [-1]
                # reversable = len(reactant) > 1

                max_times = len(pro_atom_map_numbers)
                times = min(augmentation, max_times)
                if times < augmentation:  # times = max_times
                    product_roots.extend(pro_atom_map_numbers)
                    product_roots.extend(random.choices(product_roots, k=augmentation - len(product_roots)))
                else:  # times = augmentation
                    while len(product_roots) < times:
                        product_roots.append(random.sample(pro_atom_map_numbers, 1)[0])
                        # pro_atom_map_numbers.remove(product_roots[-1])
                        if product_roots[-1] in product_roots[:-1]:
                            product_roots.pop()
                times = len(product_roots)
                assert times == augmentation
                if reversable:
                    times = int(times / 2)
            # candidates = []
            for k in range(times):
                pro_root_atom_map = product_roots[k]
                pro_root = get_root_id(pro_mol, root_map_number=pro_root_atom_map)
                cano_atom_map = get_cano_map_number(product, root=pro_root)
                if cano_atom_map is None:
                    return_status["status"] = "error_mapping"
                    return return_status
                pro_smi = clear_map_canonical_smiles(product, canonical=True, root=pro_root)
                aligned_reactants = []
                aligned_reactants_order = []
                rea_atom_map_numbers = [list(map(int, re.findall(r"(?<=:)\d+", rea))) for rea in reactant]
                used_indices = []
                for i, rea_map_number in enumerate(rea_atom_map_numbers):
                    for j, map_number in enumerate(cano_atom_map):
                        # select mapping reactans
                        if map_number in rea_map_number:
                            rea_root = get_root_id(Chem.MolFromSmiles(reactant[i]), root_map_number=map_number)
                            rea_smi = clear_map_canonical_smiles(reactant[i], canonical=True, root=rea_root)
                            aligned_reactants.append(rea_smi)
                            aligned_reactants_order.append(j)
                            used_indices.append(i)
                            break
                sorted_reactants = sorted(list(zip(aligned_reactants, aligned_reactants_order)), key=lambda x: x[1])
                aligned_reactants = [item[0] for item in sorted_reactants]
                reactant_smi = ".".join(aligned_reactants)
                product_tokens = smi_tokenizer(pro_smi)
                reactant_tokens = smi_tokenizer(reactant_smi)

                return_status['src_data'].append(product_tokens)
                return_status['tgt_data'].append(reactant_tokens)

                if reversable:
                    aligned_reactants.reverse()
                    reactant_smi = ".".join(aligned_reactants)
                    product_tokens = smi_tokenizer(pro_smi)
                    reactant_tokens = smi_tokenizer(reactant_smi)
                    return_status['src_data'].append(product_tokens)
                    return_status['tgt_data'].append(reactant_tokens)
            assert len(return_status['src_data']) == data['augmentation']
        else:
            cano_product = clear_map_canonical_smiles(product)
            cano_reactanct = ".".join([clear_map_canonical_smiles(rea) for rea in reactant if
                                       len(set(map(int, re.findall(r"(?<=:)\d+", rea))) & set(
                                           pro_atom_map_numbers)) > 0])
            return_status['src_data'].append(smi_tokenizer(cano_product))
            return_status['tgt_data'].append(smi_tokenizer(cano_reactanct))
            pro_mol = Chem.MolFromSmiles(cano_product)
            rea_mols = [Chem.MolFromSmiles(rea) for rea in cano_reactanct.split(".")]
            for i in range(int(augmentation - 1)):
                pro_smi = Chem.MolToSmiles(pro_mol, doRandom=True)
                rea_smi = [Chem.MolToSmiles(rea_mol, doRandom=True) for rea_mol in rea_mols]
                rea_smi = ".".join(rea_smi)
                return_status['src_data'].append(smi_tokenizer(pro_smi))
                return_status['tgt_data'].append(smi_tokenizer(rea_smi))
        if byproduct not in ['<eos>']:
            byproduct_mol = Chem.MolFromSmiles(byproduct)
            for src, tgt in zip(return_status['src_data'], return_status['tgt_data']):
                # byproduct_edit_distances = []
                min_dis = 10000
                min_byproduct = ''
                min_byproduct_list = []
                max_lcs_list = []
                if '.' in byproduct:
                    byproduct_list = byproduct.split('.')
                    lenth = len(byproduct_list)
                    byproduct_smi_list = [[] for _ in range(lenth)]
                    for n in range(lenth):
                        for n1 in range(Chem.MolFromSmiles(byproduct_list[n]).GetNumAtoms()):
                            byproduct_smi_list[n].append(
                                clear_map_canonical_smiles(byproduct_list[n], canonical=True, root=n1))
                    byproduct_comb_list = list(itertools.product(*byproduct_smi_list))
                    byproduct_comb_list = [('.').join(list(byproduct_comb)) for byproduct_comb in byproduct_comb_list]
                    for byproduct_smi in byproduct_comb_list:
                        byproduct_src = src.split()
                        byproduct_src.append('.')
                        byproduct_src.extend(smi_tokenizer(byproduct_smi).split())
                        distance = textdistance.levenshtein.distance(byproduct_src, tgt.split())
                        if distance <= min_dis:
                            min_dis = distance
                            min_byproduct = byproduct_smi
                    min_byproduct_list.append(min_byproduct)
                    for i in min_byproduct_list:
                        lcs_lenth = textdistance.lcsstr.distance(i, tgt.replace(' ', ''))
                        max_lcs_list.append(lcs_lenth)
                    assert len(max_lcs_list) == len(min_byproduct_list)
                    min_byproduct = min_byproduct_list[max_lcs_list.index(max(max_lcs_list))]
                    # min_byproduct = random.choice(byproduct_comb_list)
                    byproduct_src = src.replace(' ', '') + '.' + min_byproduct
                    return_status['byproduct_src_data'].append(smi_tokenizer(byproduct_src))
                    # byproduct_edit_distances.append(min_dis)
                    return_status['lcs_byproduct'].append(min_byproduct)
                else:
                    for i in range(byproduct_mol.GetNumAtoms()):
                        byproduct_smi = clear_map_canonical_smiles(byproduct, canonical=True, root=i)
                        byproduct_src = src.split()
                        byproduct_src.append('.')
                        byproduct_src.extend(smi_tokenizer(byproduct_smi).split())
                        distance = textdistance.levenshtein.distance(byproduct_src, tgt.split())
                        if distance < min_dis:
                            min_dis = distance
                            min_byproduct = byproduct_smi
                        elif distance == min_dis:
                            min_byproduct_list.append(byproduct_smi)
                    min_byproduct_list.append(min_byproduct)
                    for i in min_byproduct_list:
                        lcs_lenth = textdistance.lcsstr.distance(i, tgt.replace(' ', ''))
                        max_lcs_list.append(lcs_lenth)
                    assert len(max_lcs_list) == len(min_byproduct_list)
                    min_byproduct = min_byproduct_list[max_lcs_list.index(max(max_lcs_list))]

                    # byproduct_list = [clear_map_canonical_smiles(byproduct, canonical=True, root=n1) for n1 in range(Chem.MolFromSmiles(byproduct).GetNumAtoms())]
                    # min_byproduct = random.choice(byproduct_list)
                    byproduct_src = src.replace(' ', '') + '.' + min_byproduct
                    return_status['byproduct_src_data'].append(smi_tokenizer(byproduct_src))
                    return_status['lcs_byproduct'].append(min_byproduct)
                    # byproduct_edit_distances.append(min_dis)
                # assert min_dis < 10000
                # return_status['byproduct_edit_distance'] = np.mean(byproduct_edit_distances)
        else:
            return_status['byproduct_src_data'] = return_status['src_data']
        edit_distances = []
        for src, tgt in zip(return_status['src_data'], return_status['tgt_data']):
            edit_distances.append(textdistance.levenshtein.distance(src.split(), tgt.split()))

        return_status['edit_distance'] = np.mean(edit_distances)
    return return_status

def prob_sample_multi_process_4(data):
    pt = re.compile(r':(\d+)]')
    product = data['product']
    reactant = data['reactant']
    byproduct = data['byproduct']
    augmentation = data['augmentation']
    pro_mol = Chem.MolFromSmiles(product)
    rea_mol = Chem.MolFromSmiles(reactant)
    """checking data quality"""
    rids = sorted(re.findall(pt, reactant))
    pids = sorted(re.findall(pt, product))
    return_status = {
        "status": 0,
        "src_data": [],
        "tgt_data": [],
        "byproduct_src_data": [],
        "edit_distance": 0,
        "byproduct_edit_distance": 0,
        "lcs_byproduct": [],
    }

    if "" == product:
        return_status["status"] = "empty_p"
    if "" == reactant:
        return_status["status"] = "empty_r"
    if rea_mol is None:
        return_status["status"] = "invalid_r"
    if len(rea_mol.GetAtoms()) < 5:
        return_status["status"] = "small_r"
    if pro_mol is None:
        return_status["status"] = "invalid_p"
    # if len(pro_mol.GetAtoms()) == 1:
    #     return_status["status"] = "small_p"
    if not all([a.HasProp('molAtomMapNumber') for a in pro_mol.GetAtoms()]):
        return_status["status"] = "error_mapping_p"
    """finishing checking data quality"""
    if byproduct == 'CC(=O)O':
        a = 1

    if return_status['status'] == 0:
        pro_atom_map_numbers = list(map(int, re.findall(r"(?<=:)\d+", product)))
        reactant = reactant.split(".")
        if data['root_aligned']:
            reversable = False  # no shuffle
            # augmentation = 100
            if augmentation == 999:
                product_roots = pro_atom_map_numbers
                times = len(product_roots)
            else:
                product_roots = [-1]
                # reversable = len(reactant) > 1

                max_times = len(pro_atom_map_numbers)
                times = min(augmentation, max_times)
                if times < augmentation:  # times = max_times
                    product_roots.extend(pro_atom_map_numbers)
                    product_roots.extend(random.choices(product_roots, k=augmentation - len(product_roots)))
                else:  # times = augmentation
                    while len(product_roots) < times:
                        product_roots.append(random.sample(pro_atom_map_numbers, 1)[0])
                        # pro_atom_map_numbers.remove(product_roots[-1])
                        if product_roots[-1] in product_roots[:-1]:
                            product_roots.pop()
                times = len(product_roots)
                assert times == augmentation
                if reversable:
                    times = int(times / 2)
            # candidates = []
            for k in range(times):
                pro_root_atom_map = product_roots[k]
                pro_root = get_root_id(pro_mol, root_map_number=pro_root_atom_map)
                cano_atom_map = get_cano_map_number(product, root=pro_root)
                if cano_atom_map is None:
                    return_status["status"] = "error_mapping"
                    return return_status
                pro_smi = clear_map_canonical_smiles(product, canonical=True, root=pro_root)
                aligned_reactants = []
                aligned_reactants_order = []
                rea_atom_map_numbers = [list(map(int, re.findall(r"(?<=:)\d+", rea))) for rea in reactant]
                used_indices = []
                for i, rea_map_number in enumerate(rea_atom_map_numbers):
                    for j, map_number in enumerate(cano_atom_map):
                        # select mapping reactans
                        if map_number in rea_map_number:
                            rea_root = get_root_id(Chem.MolFromSmiles(reactant[i]), root_map_number=map_number)
                            rea_smi = clear_map_canonical_smiles(reactant[i], canonical=True, root=rea_root)
                            aligned_reactants.append(rea_smi)
                            aligned_reactants_order.append(j)
                            used_indices.append(i)
                            break
                sorted_reactants = sorted(list(zip(aligned_reactants, aligned_reactants_order)), key=lambda x: x[1])
                aligned_reactants = [item[0] for item in sorted_reactants]
                reactant_smi = ".".join(aligned_reactants)
                product_tokens = smi_tokenizer(pro_smi)
                reactant_tokens = smi_tokenizer(reactant_smi)

                return_status['src_data'].append(product_tokens)
                return_status['tgt_data'].append(reactant_tokens)

                if reversable:
                    aligned_reactants.reverse()
                    reactant_smi = ".".join(aligned_reactants)
                    product_tokens = smi_tokenizer(pro_smi)
                    reactant_tokens = smi_tokenizer(reactant_smi)
                    return_status['src_data'].append(product_tokens)
                    return_status['tgt_data'].append(reactant_tokens)
            assert len(return_status['src_data']) == data['augmentation']
        else:
            cano_product = clear_map_canonical_smiles(product)
            cano_reactanct = ".".join([clear_map_canonical_smiles(rea) for rea in reactant if
                                       len(set(map(int, re.findall(r"(?<=:)\d+", rea))) & set(
                                           pro_atom_map_numbers)) > 0])
            return_status['src_data'].append(smi_tokenizer(cano_product))
            return_status['tgt_data'].append(smi_tokenizer(cano_reactanct))
            pro_mol = Chem.MolFromSmiles(cano_product)
            rea_mols = [Chem.MolFromSmiles(rea) for rea in cano_reactanct.split(".")]
            for i in range(int(augmentation - 1)):
                pro_smi = Chem.MolToSmiles(pro_mol, doRandom=True)
                rea_smi = [Chem.MolToSmiles(rea_mol, doRandom=True) for rea_mol in rea_mols]
                rea_smi = ".".join(rea_smi)
                return_status['src_data'].append(smi_tokenizer(pro_smi))
                return_status['tgt_data'].append(smi_tokenizer(rea_smi))
        if byproduct not in ['<eos>']:
            byproduct_mol = Chem.MolFromSmiles(byproduct)
            if byproduct_mol.GetNumAtoms() == 1:
                byproduct_smi = byproduct
            else:
                if '.' in byproduct:
                    byproduct_list = byproduct.split('.')
                    lenth = len(byproduct_list)
                    byproduct_smi_list = [[] for _ in range(lenth)]
                    for n in range(lenth):
                        for n1 in range(Chem.MolFromSmiles(byproduct_list[n]).GetNumAtoms()):
                            byproduct_smi_list[n].append(
                                clear_map_canonical_smiles(byproduct_list[n], canonical=True, root=n1))
                    byproduct_comb_list = list(itertools.product(*byproduct_smi_list))
                    byproduct_smi_list = [('.').join(list(byproduct_comb)) for byproduct_comb in byproduct_comb_list]
                else:
                    byproduct_smi_list = []
                    for i in range(byproduct_mol.GetNumAtoms()):
                        byproduct_smi = clear_map_canonical_smiles(byproduct, canonical=True, root=i)
                        byproduct_smi_list.append(byproduct_smi)

                lcs_distance_smiles_list = []
                for tgt in return_status['tgt_data']:
                    lcs_distance_list = []
                    for byproduct_smi in byproduct_smi_list:
                        lcs_distance = len(byproduct_smi) - textdistance.lcsstr.similarity(byproduct_smi, tgt.replace(' ', ''))
                        lcs_distance_list.append(lcs_distance)

                    lcs_distance_smiles_list.append(lcs_distance_list)
                lcs_distance_smiles = np.array(lcs_distance_smiles_list)
                lcs_distance_smiles = (lcs_distance_smiles == lcs_distance_smiles.min(axis=1, keepdims=1)).astype(float)
                value = np.sum(lcs_distance_smiles, axis=0)
                max_lcs_index = np.where(value == np.max(value))
                if len(max_lcs_index) > 1:
                    print('eeeee')
                byproduct_smi = byproduct_smi_list[max_lcs_index[0][0]]

            for src, tgt in zip(return_status['src_data'], return_status['tgt_data']):
                byproduct_src = src.replace(' ', '') + '.' + byproduct_smi
                return_status['byproduct_src_data'].append(smi_tokenizer(byproduct_src))
                return_status['lcs_byproduct'].append(byproduct_smi)
        else:
            return_status['byproduct_src_data'] = return_status['src_data']
        edit_distances = []
        for src, tgt in zip(return_status['src_data'], return_status['tgt_data']):
            edit_distances.append(textdistance.levenshtein.distance(src.split(), tgt.split()))

        return_status['edit_distance'] = np.mean(edit_distances)
    return return_status

def prob_sample_multi_process_5(data):
    pt = re.compile(r':(\d+)]')
    product = data['product']
    reactant = data['reactant']
    byproduct = data['byproduct']
    augmentation = data['augmentation']
    pro_mol = Chem.MolFromSmiles(product)
    rea_mol = Chem.MolFromSmiles(reactant)
    """checking data quality"""
    rids = sorted(re.findall(pt, reactant))
    pids = sorted(re.findall(pt, product))
    return_status = {
        "status": 0,
        "src_data": [],
        "tgt_data": [],
        "byproduct_src_data": [],
        "edit_distance": 0,
        "byproduct_edit_distance": 0,
        "lcs_byproduct": [],
    }

    if "" == product:
        return_status["status"] = "empty_p"
    if "" == reactant:
        return_status["status"] = "empty_r"
    if rea_mol is None:
        return_status["status"] = "invalid_r"
    if len(rea_mol.GetAtoms()) < 5:
        return_status["status"] = "small_r"
    if pro_mol is None:
        return_status["status"] = "invalid_p"
    # if len(pro_mol.GetAtoms()) == 1:
    #     return_status["status"] = "small_p"
    if not all([a.HasProp('molAtomMapNumber') for a in pro_mol.GetAtoms()]):
        return_status["status"] = "error_mapping_p"
    """finishing checking data quality"""

    if return_status['status'] == 0:
        pro_atom_map_numbers = list(map(int, re.findall(r"(?<=:)\d+", product)))
        reactant = reactant.split(".")
        if data['root_aligned']:
            reversable = False  # no shuffle
            # augmentation = 100
            if augmentation == 999:
                product_roots = pro_atom_map_numbers
                times = len(product_roots)
            else:
                product_roots = [-1]
                # reversable = len(reactant) > 1

                max_times = len(pro_atom_map_numbers)
                times = min(augmentation, max_times)
                if times < augmentation:  # times = max_times
                    product_roots.extend(pro_atom_map_numbers)
                    product_roots.extend(random.choices(product_roots, k=augmentation - len(product_roots)))
                else:  # times = augmentation
                    while len(product_roots) < times:
                        product_roots.append(random.sample(pro_atom_map_numbers, 1)[0])
                        # pro_atom_map_numbers.remove(product_roots[-1])
                        if product_roots[-1] in product_roots[:-1]:
                            product_roots.pop()
                times = len(product_roots)
                assert times == augmentation
                if reversable:
                    times = int(times / 2)
            # candidates = []
            for k in range(times):
                pro_root_atom_map = product_roots[k]
                pro_root = get_root_id(pro_mol, root_map_number=pro_root_atom_map)
                cano_atom_map = get_cano_map_number(product, root=pro_root)
                if cano_atom_map is None:
                    return_status["status"] = "error_mapping"
                    return return_status
                pro_smi = clear_map_canonical_smiles(product, canonical=True, root=pro_root)
                aligned_reactants = []
                aligned_reactants_order = []
                rea_atom_map_numbers = [list(map(int, re.findall(r"(?<=:)\d+", rea))) for rea in reactant]
                used_indices = []
                for i, rea_map_number in enumerate(rea_atom_map_numbers):
                    for j, map_number in enumerate(cano_atom_map):
                        # select mapping reactans
                        if map_number in rea_map_number:
                            rea_root = get_root_id(Chem.MolFromSmiles(reactant[i]), root_map_number=map_number)
                            rea_smi = clear_map_canonical_smiles(reactant[i], canonical=True, root=rea_root)
                            aligned_reactants.append(rea_smi)
                            aligned_reactants_order.append(j)
                            used_indices.append(i)
                            break
                sorted_reactants = sorted(list(zip(aligned_reactants, aligned_reactants_order)), key=lambda x: x[1])
                aligned_reactants = [item[0] for item in sorted_reactants]
                reactant_smi = ".".join(aligned_reactants)
                product_tokens = smi_tokenizer(pro_smi)
                reactant_tokens = smi_tokenizer(reactant_smi)

                return_status['src_data'].append(product_tokens)
                return_status['tgt_data'].append(reactant_tokens)

                if reversable:
                    aligned_reactants.reverse()
                    reactant_smi = ".".join(aligned_reactants)
                    product_tokens = smi_tokenizer(pro_smi)
                    reactant_tokens = smi_tokenizer(reactant_smi)
                    return_status['src_data'].append(product_tokens)
                    return_status['tgt_data'].append(reactant_tokens)
            assert len(return_status['src_data']) == data['augmentation']
        else:
            cano_product = clear_map_canonical_smiles(product)
            cano_reactanct = ".".join([clear_map_canonical_smiles(rea) for rea in reactant if
                                       len(set(map(int, re.findall(r"(?<=:)\d+", rea))) & set(
                                           pro_atom_map_numbers)) > 0])
            return_status['src_data'].append(smi_tokenizer(cano_product))
            return_status['tgt_data'].append(smi_tokenizer(cano_reactanct))
            pro_mol = Chem.MolFromSmiles(cano_product)
            rea_mols = [Chem.MolFromSmiles(rea) for rea in cano_reactanct.split(".")]
            for i in range(int(augmentation - 1)):
                pro_smi = Chem.MolToSmiles(pro_mol, doRandom=True)
                rea_smi = [Chem.MolToSmiles(rea_mol, doRandom=True) for rea_mol in rea_mols]
                rea_smi = ".".join(rea_smi)
                return_status['src_data'].append(smi_tokenizer(pro_smi))
                return_status['tgt_data'].append(smi_tokenizer(rea_smi))
        if byproduct not in ['<eos>']:
            byproduct_mol = Chem.MolFromSmiles(byproduct)
            if byproduct_mol.GetNumAtoms() == 1:
                byproduct_smi = byproduct
            else:
                if '.' in byproduct:
                    byproduct_list = byproduct.split('.')
                    lenth = len(byproduct_list)
                    byproduct_smi_list = [[] for _ in range(lenth)]
                    for n in range(lenth):
                        for n1 in range(Chem.MolFromSmiles(byproduct_list[n]).GetNumAtoms()):
                            byproduct_smi_list[n].append(
                                clear_map_canonical_smiles(byproduct_list[n], canonical=True, root=n1))
                    byproduct_comb_list = list(itertools.product(*byproduct_smi_list))
                    byproduct_smi_list = [('.').join(list(byproduct_comb)) for byproduct_comb in byproduct_comb_list]
                else:
                    byproduct_smi_list = []
                    for i in range(byproduct_mol.GetNumAtoms()):
                        byproduct_smi = clear_map_canonical_smiles(byproduct, canonical=True, root=i)
                        byproduct_smi_list.append(byproduct_smi)
                if len(list(set(byproduct_smi_list))) == 1:
                    byproduct_smi = byproduct_smi_list[0]
                else:
                    lcs_distance_smiles_list = []
                    for tgt in return_status['tgt_data']:
                        lcs_distance_list = []
                        rule = str.maketrans('', '', digits)
                        for byproduct_smi in byproduct_smi_list:
                            o_number_byproduct = byproduct_smi.translate(rule)
                            o_number_tgt = tgt.replace(' ', '').translate(rule)
                            lcs_distance = len(o_number_byproduct) - textdistance.lcsstr.similarity(o_number_byproduct, o_number_tgt)
                            lcs_distance_list.append(lcs_distance)

                        lcs_distance_smiles_list.append(lcs_distance_list)
                    lcs_distance_smiles = np.array(lcs_distance_smiles_list)
                    lcs_distance_smiles = (lcs_distance_smiles == lcs_distance_smiles.min(axis=1, keepdims=1)).astype(float)
                    value = np.sum(lcs_distance_smiles, axis=0)
                    max_lcs_index = np.where(value == np.max(value))
                    # if len(max_lcs_index[0]) > 1:
                    #     print('eeeee')
                    byproduct_smi = byproduct_smi_list[max_lcs_index[0][0]]

            for src, tgt in zip(return_status['src_data'], return_status['tgt_data']):
                byproduct_src = src.replace(' ', '') + '.' + byproduct_smi
                return_status['byproduct_src_data'].append(smi_tokenizer(byproduct_src))
                return_status['lcs_byproduct'].append(byproduct_smi)
        else:
            return_status['byproduct_src_data'] = return_status['src_data']
        edit_distances = []
        for src, tgt in zip(return_status['src_data'], return_status['tgt_data']):
            edit_distances.append(textdistance.levenshtein.distance(src.split(), tgt.split()))

        return_status['edit_distance'] = np.mean(edit_distances)
    return return_status

def prob_sample_multi_process_6(data):

    product = data['product']
    reactant = data['reactant']
    byproduct = data['byproduct']
    augmentation = data['augmentation']
    pro_mol = Chem.MolFromSmiles(product)
    rea_mol = Chem.MolFromSmiles(reactant)
    """checking data quality"""
    return_status = {
        "status": 0,
        "src_data": [],
        "tgt_data": [],
        "byproduct_src_data": [],
        "edit_distance": 0,
        "byproduct_edit_distance": 0,
        "lcs_byproduct": [],
    }

    if "" == product:
        return_status["status"] = "empty_p"
    if "" == reactant:
        return_status["status"] = "empty_r"
    if rea_mol is None:
        return_status["status"] = "invalid_r"
    if len(rea_mol.GetAtoms()) < 5:
        return_status["status"] = "small_r"
    if pro_mol is None:
        return_status["status"] = "invalid_p"
    # if len(pro_mol.GetAtoms()) == 1:
    #     return_status["status"] = "small_p"
    if not all([a.HasProp('molAtomMapNumber') for a in pro_mol.GetAtoms()]):
        return_status["status"] = "error_mapping_p"
    """finishing checking data quality"""

    if return_status['status'] == 0:
        pro_atom_map_numbers = list(map(int, re.findall(r"(?<=:)\d+", product)))
        reactant = reactant.split(".")

        product_roots = [-1]
        max_times = len(pro_atom_map_numbers)
        times = min(augmentation, max_times)
        if times < augmentation:  # times = max_times
            product_roots.extend(pro_atom_map_numbers)
            product_roots.extend(random.choices(product_roots, k=augmentation - len(product_roots)))
        else:  # times = augmentation
            while len(product_roots) < times:
                product_roots.append(random.sample(pro_atom_map_numbers, 1)[0])
                # pro_atom_map_numbers.remove(product_roots[-1])
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
            aligned_reactants = []
            aligned_reactants_order = []
            rea_atom_map_numbers = [list(map(int, re.findall(r"(?<=:)\d+", rea))) for rea in reactant]
            used_indices = []
            for i, rea_map_number in enumerate(rea_atom_map_numbers):
                for j, map_number in enumerate(cano_atom_map):
                    # select mapping reactans
                    if map_number in rea_map_number:
                        rea_root = get_root_id(Chem.MolFromSmiles(reactant[i]), root_map_number=map_number)
                        rea_smi = clear_map_canonical_smiles(reactant[i], canonical=True, root=rea_root)
                        aligned_reactants.append(rea_smi)
                        aligned_reactants_order.append(j)
                        used_indices.append(i)
                        break
            sorted_reactants = sorted(list(zip(aligned_reactants, aligned_reactants_order)), key=lambda x: x[1])
            aligned_reactants = [item[0] for item in sorted_reactants]
            reactant_smi = ".".join(aligned_reactants)
            product_tokens = smi_tokenizer(pro_smi)
            reactant_tokens = smi_tokenizer(reactant_smi)

            return_status['src_data'].append(product_tokens)
            return_status['tgt_data'].append(reactant_tokens)

        assert len(return_status['src_data']) == data['augmentation']

        if byproduct not in ['<eos>']:
            byproduct_mol = Chem.MolFromSmiles(byproduct)

            if '.' in byproduct:
                byproduct_list = byproduct.split('.')
                lenth = len(byproduct_list)
                byproduct_smi_list = [[] for _ in range(lenth)]
                for n in range(lenth):
                    for n1 in range(Chem.MolFromSmiles(byproduct_list[n]).GetNumAtoms()):
                        byproduct_smi_list[n].append(
                            clear_map_canonical_smiles(byproduct_list[n], canonical=True, root=n1))
                byproduct_comb_list = list(itertools.product(*byproduct_smi_list))
                byproduct_list = [('.').join(list(byproduct_comb)) for byproduct_comb in byproduct_comb_list]
            else:
                byproduct_list = []
                for i in range(byproduct_mol.GetNumAtoms()):
                    byproduct_smi = clear_map_canonical_smiles(byproduct, canonical=True, root=i)
                    byproduct_list.append(byproduct_smi)
            byproduct_list = list(set(byproduct_list))
            for src, tgt in zip(return_status['src_data'], return_status['tgt_data']):
                max_lcs_list = []
                distance_list = []

                for byproduct_smi in byproduct_list:
                    new_src = src.split()
                    new_src.append('.')
                    new_src.extend(smi_tokenizer(byproduct_smi).split())
                    distance = textdistance.levenshtein.distance(new_src, tgt.split())
                    distance_list.append(distance)
                min_distance = min(distance_list)
                min_distance_idx = [idx for idx, distance in enumerate(distance_list) if distance == min_distance]
                min_byproduct_list = [byproduct_list[idx] for idx in min_distance_idx]
                rule = str.maketrans('', '', digits)
                for i in min_byproduct_list:
                    o_number_i = i.translate(rule)
                    o_number_tgt = tgt.replace(' ', '').translate(rule)
                    lcs_distance = len(o_number_i) - textdistance.lcsstr.similarity(o_number_i, o_number_tgt)
                    # lcs_lenth = find_lcs_len(i, tgt.replace(' ', ''))
                    max_lcs_list.append(lcs_distance)
                assert len(max_lcs_list) == len(min_byproduct_list)
                min_byproduct = min_byproduct_list[max_lcs_list.index(min(max_lcs_list))]
                byproduct_src = src.replace(' ', '') + '.' + min_byproduct
                return_status['byproduct_src_data'].append(smi_tokenizer(byproduct_src))
                return_status['lcs_byproduct'].append(byproduct)
        else:
            return_status['byproduct_src_data'] = return_status['src_data']

        edit_distances = []
        for src, tgt in zip(return_status['src_data'], return_status['tgt_data']):
            edit_distances.append(textdistance.levenshtein.distance(src.split(), tgt.split()))

        return_status['edit_distance'] = np.mean(edit_distances)
    return return_status

def multi_process(data):
    global args
    product = data['product']
    reactant = data['reactant']
    byproduct = data['byproduct']
    rxn_class = data['rxn_class']
    augmentation = data['augmentation']
    pro_mol = Chem.MolFromSmiles(product)
    rea_mol = Chem.MolFromSmiles(reactant)
    """checking data quality"""
    return_status = {
        "status": 0,
        "src_data": [],
        "tgt_data": [],
        "byproduct_src_data": [],
        "use_class_src_data": [],
        "edit_distance": 0,
        "byproduct_edit_distance": 0,
    }
    if "" == product:
        return_status["status"] = "empty_p"
    if "" == reactant:
        return_status["status"] = "empty_r"
    if rea_mol is None:
        return_status["status"] = "invalid_r"
    if len(rea_mol.GetAtoms()) < 5:
        return_status["status"] = "small_r"
    if pro_mol is None:
        return_status["status"] = "invalid_p"
    if not all([a.HasProp('molAtomMapNumber') for a in pro_mol.GetAtoms()]):
        return_status["status"] = "error_mapping_p"
    """finishing checking data quality"""

    if return_status['status'] == 0:
        pro_atom_map_numbers = list(map(int, re.findall(r"(?<=:)\d+", product)))
        reactant = reactant.split(".")

        product_roots = [-1]
        max_times = len(pro_atom_map_numbers)
        times = min(augmentation, max_times)
        if times < augmentation:  # times = max_times
            product_roots.extend(pro_atom_map_numbers)
            product_roots.extend(random.choices(product_roots, k=augmentation - len(product_roots)))
        else:  # times = augmentation
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
            aligned_reactants = []
            aligned_reactants_order = []
            rea_atom_map_numbers = [list(map(int, re.findall(r"(?<=:)\d+", rea))) for rea in reactant]
            used_indices = []
            for i, rea_map_number in enumerate(rea_atom_map_numbers):
                for j, map_number in enumerate(cano_atom_map):
                    # select mapping reactans
                    if map_number in rea_map_number:
                        rea_root = get_root_id(Chem.MolFromSmiles(reactant[i]), root_map_number=map_number)
                        rea_smi = clear_map_canonical_smiles(reactant[i], canonical=True, root=rea_root)
                        aligned_reactants.append(rea_smi)
                        aligned_reactants_order.append(j)
                        used_indices.append(i)
                        break
            sorted_reactants = sorted(list(zip(aligned_reactants, aligned_reactants_order)), key=lambda x: x[1])
            aligned_reactants = [item[0] for item in sorted_reactants]
            reactant_smi = ".".join(aligned_reactants)
            product_tokens = smi_tokenizer(pro_smi)
            reactant_tokens = smi_tokenizer(reactant_smi)

            return_status['src_data'].append(product_tokens)
            return_status['tgt_data'].append(reactant_tokens)
        assert len(return_status['src_data']) == data['augmentation']

        if byproduct not in ['<eos>']:
            for src in return_status['src_data']:
                try:
                    if args.sample in ['random', 'prob']:
                        byproduct_prob_dict = lcs_byproduct_prob_dict[byproduct]
                        weights = list(byproduct_prob_dict.values()) if args.sample == 'prob' else None
                        random_byproduct = \
                            random.choices(list(byproduct_prob_dict.keys()), weights=weights, k=1)[-1]
                    else:
                        assert args.sample == 'cano'
                        random_byproduct = clear_map_canonical_smiles(byproduct)
                except:
                    random_byproduct = byproduct
                byproduct_src = src.replace(' ', '') + '.' + random_byproduct
                return_status['byproduct_src_data'].append(smi_tokenizer(byproduct_src))
        else:
            return_status['byproduct_src_data'] = return_status['src_data']
        edit_distances = []
        for src, tgt in zip(return_status['src_data'], return_status['tgt_data']):
            edit_distances.append(textdistance.levenshtein.distance(src.split(), tgt.split()))
        return_status['edit_distance'] = np.mean(edit_distances)
        byproduct_edit_distances = []
        for src, tgt in zip(return_status['byproduct_src_data'], return_status['tgt_data']):
            byproduct_edit_distances.append(textdistance.levenshtein.distance(src.split(), tgt.split()))
        return_status['byproduct_edit_distances'] = np.mean(byproduct_edit_distances)

        for i in return_status['byproduct_src_data']:
            return_status['use_class_src_data'].append('RXN_%s %s' % (rxn_class, i))

    return return_status

def multi_process_2(data):
    global args
    product = data['product']
    reactant = data['reactant']
    byproduct = data['byproduct']
    rxn_class = data['rxn_class']
    augmentation = data['augmentation']
    pro_mol = Chem.MolFromSmiles(product)
    rea_mol = Chem.MolFromSmiles(reactant)
    """checking data quality"""
    return_status = {
        "status": 0,
        "src_data": [],
        "tgt_data": [],
        "byproduct_src_data": [],
        "use_class_src_data": [],
        "edit_distance": 0,
        "byproduct_edit_distance": 0,
    }
    if "" == product:
        return_status["status"] = "empty_p"
    if "" == reactant:
        return_status["status"] = "empty_r"
    if rea_mol is None:
        return_status["status"] = "invalid_r"
    if len(rea_mol.GetAtoms()) < 5:
        return_status["status"] = "small_r"
    if pro_mol is None:
        return_status["status"] = "invalid_p"
    if not all([a.HasProp('molAtomMapNumber') for a in pro_mol.GetAtoms()]):
        return_status["status"] = "error_mapping_p"
    """finishing checking data quality"""

    if return_status['status'] == 0:
        pro_atom_map_numbers = list(map(int, re.findall(r"(?<=:)\d+", product)))
        reactant = reactant.split(".")

        product_roots = [-1]
        max_times = len(pro_atom_map_numbers)
        times = min(augmentation, max_times)
        if times < augmentation:  # times = max_times
            product_roots.extend(pro_atom_map_numbers)
            product_roots.extend(random.choices(product_roots, k=augmentation - len(product_roots)))
        else:  # times = augmentation
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
            aligned_reactants = []
            aligned_reactants_order = []
            rea_atom_map_numbers = [list(map(int, re.findall(r"(?<=:)\d+", rea))) for rea in reactant]
            used_indices = []
            for i, rea_map_number in enumerate(rea_atom_map_numbers):
                for j, map_number in enumerate(cano_atom_map):
                    # select mapping reactans
                    if map_number in rea_map_number:
                        rea_root = get_root_id(Chem.MolFromSmiles(reactant[i]), root_map_number=map_number)
                        rea_smi = clear_map_canonical_smiles(reactant[i], canonical=True, root=rea_root)
                        aligned_reactants.append(rea_smi)
                        aligned_reactants_order.append(j)
                        used_indices.append(i)
                        break
            sorted_reactants = sorted(list(zip(aligned_reactants, aligned_reactants_order)), key=lambda x: x[1])
            aligned_reactants = [item[0] for item in sorted_reactants]
            reactant_smi = ".".join(aligned_reactants)
            product_tokens = smi_tokenizer(pro_smi)
            reactant_tokens = smi_tokenizer(reactant_smi)

            return_status['src_data'].append(product_tokens)
            return_status['tgt_data'].append(reactant_tokens)
        assert len(return_status['src_data']) == data['augmentation']

        if byproduct not in ['<eos>']:
            try:
                if args.sample == 'prob':
                    byproduct_prob_dict = lcs_byproduct_prob_dict[byproduct]
                    weights = list(byproduct_prob_dict.values())
                    byproduct = random.choices(list(byproduct_prob_dict.keys()), weights=weights, k=1)[-1]
                    return_status['byproduct_src_data'] = [smi_tokenizer(src.replace(' ', '') + '.' + byproduct) for src
                                                          in return_status['src_data']]
                elif args.sample == 'random':
                    byproduct_prod_mol = Chem.MolFromSmiles(byproduct)
                    if '.' in byproduct:
                        byproduct_list = byproduct.split('.')
                        lenth = len(byproduct_list)
                        byproduct_smi_list = [[] for _ in range(lenth)]
                        for n in range(lenth):
                            for n1 in range(Chem.MolFromSmiles(byproduct_list[n]).GetNumAtoms()):
                                byproduct_smi_list[n].append(
                                    clear_map_canonical_smiles(byproduct_list[n], canonical=True, root=n1))
                        byproduct_comb_list = list(itertools.product(*byproduct_smi_list))
                        byproduct_list = [('.').join(list(byproduct_comb)) for byproduct_comb in byproduct_comb_list]
                    else:
                        byproduct_list = []
                        for i in range(byproduct_prod_mol.GetNumAtoms()):
                            byproduct_smi = clear_map_canonical_smiles(byproduct, canonical=True, root=i)
                            byproduct_list.append(byproduct_smi)
                    for src in return_status['src_data']:
                        byproduct = random.choice(byproduct_list)
                        return_status['byproduct_src_data'].append(smi_tokenizer(src.replace(' ', '') + '.' + byproduct))
                else:
                    assert args.sample == 'cano'
                    byproduct = clear_map_canonical_smiles(byproduct)
                    return_status['byproduct_src_data'] = [smi_tokenizer(src.replace(' ', '') + '.' + byproduct) for src
                                                          in return_status['src_data']]
            except Exception as e:
                print(e)
                pass
        else:
            return_status['byproduct_src_data'] = return_status['src_data']
        edit_distances = []
        for src, tgt in zip(return_status['src_data'], return_status['tgt_data']):
            edit_distances.append(textdistance.levenshtein.distance(src.split(), tgt.split()))
        return_status['edit_distance'] = np.mean(edit_distances)
        byproduct_edit_distances = []
        for src, tgt in zip(return_status['byproduct_src_data'], return_status['tgt_data']):
            byproduct_edit_distances.append(textdistance.levenshtein.distance(src.split(), tgt.split()))
        return_status['byproduct_edit_distances'] = np.mean(byproduct_edit_distances)

        for i in return_status['byproduct_src_data']:
            return_status['use_class_src_data'].append('RXN_%s %s' % (rxn_class, i))

    return return_status

def preprocess(reactants, products, byproducts, rxns_class_list, set_name):

    global args

    augmentation = args.augmentation
    processes = args.processes

    data = [{
        "reactant": i,
        "product": j,
        "byproduct": n,
        "rxn_class": r,
        "augmentation": augmentation,
    } for i, j, n, r in zip(reactants, products, byproducts, rxns_class_list)]

    src_data = []
    tgt_data = []
    byproduct_src_data = []
    use_class_src_data = []
    skip_dict = {
        'invalid_p': 0,
        'invalid_r': 0,
        'small_p': 0,
        'small_r': 0,
        'error_mapping': 0,
        'error_mapping_p': 0,
        'empty_p': 0,
        'empty_r': 0,
    }

    processes = multiprocessing.cpu_count() if processes < 0 else processes
    with multiprocessing.Pool(processes=processes) as pool:
        results = list(tqdm(pool.imap(multi_process_2, data), total=len(data)))
    pool.close()
    pool.join()

    edit_distances = []
    byproduct_edit_distances = []
    for result in tqdm(results):
        if result['status'] != 0:
            skip_dict[result['status']] += 1
            continue
        edit_distances.append(result['edit_distance'])
        byproduct_edit_distances.append(result['byproduct_edit_distances'])
        src_data.extend(result['src_data'])
        tgt_data.extend(result['tgt_data'])
        byproduct_src_data.extend(result['byproduct_src_data'])
        use_class_src_data.extend(result['use_class_src_data'])

    print("Avg. edit distance:", np.mean(edit_distances))
    print("Avg. byproduct edit distance:", np.mean(byproduct_edit_distances))
    print('size', len(byproduct_src_data))
    # for key, value in skip_dict.items():
    #     print(f"{key}:{value},{value / len(reactants)}")
    if args.topn != 1:
        save_name = 'top%s_' % args.topn
    else:
        save_name = ''

    if args.mode == 'test':
        save_dir = os.path.join(r'../%s' % args.task_dataset, r'dataset/stage_two/with_class') if args.with_class else os.path.join(r'../%s' % args.task_dataset, r'dataset/stage_two/without_class')
    else:
        save_dir = os.path.join(r'../%s' % args.task_dataset, r'dataset/stage_two/%s' % set_name)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


    with open(os.path.join(save_dir, '{}tgt-{}.txt'.format(save_name, set_name)), 'w') as f:
        for tgt in tgt_data:
            f.write('{}\n'.format(tgt))

    if args.mode != 'test' or not args.with_class:
        with open(os.path.join(save_dir, '{}byproduct_src-{}.txt'.format(save_name, set_name)), 'w') as f:
            for byproduct_src in byproduct_src_data:
                f.write('{}\n'.format(byproduct_src))

    if args.mode != 'test' or args.with_class:
        with open(os.path.join(save_dir, '{}byproduct_src-{}_with_class.txt'.format(save_name, set_name)), 'w') as f:
            for use_class_src in use_class_src_data:
                f.write('{}\n'.format(use_class_src))

    return src_data, tgt_data, byproduct_src_data


def find_lcs_len(s1, s2):
    m = [[0 for x in s2] for y in s1]
    for p1 in range(len(s1)):
        for p2 in range(len(s2)):
            if s1[p1] == s2[p2]:
                if p1 == 0 or p2 == 0:
                    m[p1][p2] = 1
                else:
                    m[p1][p2] = m[p1 - 1][p2 - 1] + 1
            elif m[p1 - 1][p2] < m[p1][p2 - 1]:
                m[p1][p2] = m[p1][p2 - 1]
            else:  # m[p1][p2-1] < m[p1-1][p2]
                m[p1][p2] = m[p1 - 1][p2]
    return m[-1][-1]


def single_process(data):
    pt = re.compile(r':(\d+)]')
    product = data['product']
    reactant = data['reactant']
    byproduct = data['byproduct']
    augmentation = data['augmentation']
    pro_mol = Chem.MolFromSmiles(product)
    rea_mol = Chem.MolFromSmiles(reactant)
    """checking data quality"""
    rids = sorted(re.findall(pt, reactant))
    pids = sorted(re.findall(pt, product))
    return_status = {
        "status": 0,
        "src_data": [],
        "tgt_data": [],
        "byproduct_src_data": [],
        "edit_distance": 0,
        "byproduct_edit_distance": 0,
    }

    if "" == product:
        return_status["status"] = "empty_p"
    if "" == reactant:
        return_status["status"] = "empty_r"
    if rea_mol is None:
        return_status["status"] = "invalid_r"
    if len(rea_mol.GetAtoms()) < 5:
        return_status["status"] = "small_r"
    if pro_mol is None:
        return_status["status"] = "invalid_p"
    if len(pro_mol.GetAtoms()) == 1:
        return_status["status"] = "small_p"
    if not all([a.HasProp('molAtomMapNumber') for a in pro_mol.GetAtoms()]):
        return_status["status"] = "error_mapping_p"
    """finishing checking data quality"""

    if return_status['status'] == 0:
        pro_atom_map_numbers = list(map(int, re.findall(r"(?<=:)\d+", product)))
        reactant = reactant.split(".")
        if data['root_aligned']:
            reversable = False  # no shuffle
            # augmentation = 100
            if augmentation == 999:
                product_roots = pro_atom_map_numbers
                times = len(product_roots)
            else:
                product_roots = [-1]
                # reversable = len(reactant) > 1

                max_times = len(pro_atom_map_numbers)
                times = min(augmentation, max_times)
                if times < augmentation:  # times = max_times
                    product_roots.extend(pro_atom_map_numbers)
                    product_roots.extend(random.choices(product_roots, k=augmentation - len(product_roots)))
                else:  # times = augmentation
                    while len(product_roots) < times:
                        product_roots.append(random.sample(pro_atom_map_numbers, 1)[0])
                        # pro_atom_map_numbers.remove(product_roots[-1])
                        if product_roots[-1] in product_roots[:-1]:
                            product_roots.pop()
                times = len(product_roots)
                assert times == augmentation
                if reversable:
                    times = int(times / 2)
            # candidates = []
            for k in range(times):
                pro_root_atom_map = product_roots[k]
                pro_root = get_root_id(pro_mol, root_map_number=pro_root_atom_map)
                cano_atom_map = get_cano_map_number(product, root=pro_root)
                if cano_atom_map is None:
                    return_status["status"] = "error_mapping"
                    return return_status
                pro_smi = clear_map_canonical_smiles(product, canonical=True, root=pro_root)
                aligned_reactants = []
                aligned_reactants_order = []
                rea_atom_map_numbers = [list(map(int, re.findall(r"(?<=:)\d+", rea))) for rea in reactant]
                used_indices = []
                for i, rea_map_number in enumerate(rea_atom_map_numbers):
                    for j, map_number in enumerate(cano_atom_map):
                        # select mapping reactans
                        if map_number in rea_map_number:
                            rea_root = get_root_id(Chem.MolFromSmiles(reactant[i]), root_map_number=map_number)
                            rea_smi = clear_map_canonical_smiles(reactant[i], canonical=True, root=rea_root)
                            aligned_reactants.append(rea_smi)
                            aligned_reactants_order.append(j)
                            used_indices.append(i)
                            break
                sorted_reactants = sorted(list(zip(aligned_reactants, aligned_reactants_order)), key=lambda x: x[1])
                aligned_reactants = [item[0] for item in sorted_reactants]
                reactant_smi = ".".join(aligned_reactants)
                product_tokens = smi_tokenizer(pro_smi)
                reactant_tokens = smi_tokenizer(reactant_smi)

                return_status['src_data'].append(product_tokens)
                return_status['tgt_data'].append(reactant_tokens)

                if reversable:
                    aligned_reactants.reverse()
                    reactant_smi = ".".join(aligned_reactants)
                    product_tokens = smi_tokenizer(pro_smi)
                    reactant_tokens = smi_tokenizer(reactant_smi)
                    return_status['src_data'].append(product_tokens)
                    return_status['tgt_data'].append(reactant_tokens)
            assert len(return_status['src_data']) == data['augmentation']
        else:
            cano_product = clear_map_canonical_smiles(product)
            cano_reactanct = ".".join([clear_map_canonical_smiles(rea) for rea in reactant if
                                       len(set(map(int, re.findall(r"(?<=:)\d+", rea))) & set(
                                           pro_atom_map_numbers)) > 0])
            return_status['src_data'].append(smi_tokenizer(cano_product))
            return_status['tgt_data'].append(smi_tokenizer(cano_reactanct))
            pro_mol = Chem.MolFromSmiles(cano_product)
            rea_mols = [Chem.MolFromSmiles(rea) for rea in cano_reactanct.split(".")]
            for i in range(int(augmentation - 1)):
                pro_smi = Chem.MolToSmiles(pro_mol, doRandom=True)
                rea_smi = [Chem.MolToSmiles(rea_mol, doRandom=True) for rea_mol in rea_mols]
                rea_smi = ".".join(rea_smi)
                return_status['src_data'].append(smi_tokenizer(pro_smi))
                return_status['tgt_data'].append(smi_tokenizer(rea_smi))
        if byproduct not in ['<eos>', '<unk>']:
            byproduct_mol = Chem.MolFromSmiles(byproduct)
            for src, tgt in zip(return_status['src_data'], return_status['tgt_data']):
                byproduct_edit_distances = []
                min_dis = 10000
                min_byproduct = ''
                min_byproduct_list = []
                max_lcs_list = []
                if '.' in byproduct:
                    byproduct_list = byproduct.split('.')
                    lenth = len(byproduct_list)
                    byproduct_smi_list = [[] for _ in range(lenth)]
                    for n in range(lenth):
                        for n1 in range(Chem.MolFromSmiles(byproduct_list[n]).GetNumAtoms()):
                            byproduct_smi_list[n].append(
                                clear_map_canonical_smiles(byproduct_list[n], canonical=True, root=n1))
                    # byproduct_comb_list = combine(byproduct_smi_list)
                    byproduct_comb_list = list(itertools.product(*byproduct_smi_list))
                    byproduct_comb_list = [('.').join(list(byproduct_comb)) for byproduct_comb in byproduct_comb_list]
                    for byproduct_smi in list(set(byproduct_comb_list)):
                        byproduct_src = src.split()
                        byproduct_src.append('.')
                        byproduct_src.extend(smi_tokenizer(byproduct_smi).split())
                        distance = textdistance.levenshtein.distance(byproduct_src, tgt.split())
                        if distance < min_dis:
                            min_dis = distance
                            min_byproduct = byproduct_smi
                        elif distance == min_dis:
                            min_byproduct_list.append(byproduct_smi)
                    min_byproduct_list.append(min_byproduct)
                    for i in min_byproduct_list:
                        lcs_lenth = find_lcs_len(i, tgt.replace(' ', ''))
                        max_lcs_list.append(lcs_lenth)
                    assert len(max_lcs_list) == len(min_byproduct_list)
                    min_byproduct = min_byproduct_list[max_lcs_list.index(max(max_lcs_list))]
                    byproduct_src = src.replace(' ', '') + '.' + min_byproduct
                    return_status['byproduct_src_data'].append(smi_tokenizer(byproduct_src))
                    byproduct_edit_distances.append(min_dis)
                else:
                    for i in range(byproduct_mol.GetNumAtoms()):
                        byproduct_smi = clear_map_canonical_smiles(byproduct, canonical=True, root=i)
                        byproduct_src = src.split()
                        byproduct_src.append('.')
                        byproduct_src.extend(smi_tokenizer(byproduct_smi).split())
                        distance = textdistance.levenshtein.distance(byproduct_src, tgt.split())
                        if distance <= min_dis:
                            min_dis = distance
                            min_byproduct = byproduct_smi
                    min_byproduct_list.append(min_byproduct)
                    for i in min_byproduct_list:
                        lcs_lenth = find_lcs_len(i, tgt.replace(' ', ''))
                        max_lcs_list.append(lcs_lenth)
                    assert len(max_lcs_list) == len(min_byproduct_list)
                    min_byproduct = min_byproduct_list[max_lcs_list.index(max(max_lcs_list))]
                    byproduct_src = src.replace(' ', '') + '.' + min_byproduct
                    return_status['byproduct_src_data'].append(smi_tokenizer(byproduct_src))
                    byproduct_edit_distances.append(min_dis)
                assert min_dis < 10000
                return_status['byproduct_edit_distance'] = np.mean(byproduct_edit_distances)
        else:
            return_status['byproduct_src_data'] = return_status['src_data']
        edit_distances = []
        for src, tgt in zip(return_status['src_data'], return_status['tgt_data']):
            edit_distances.append(textdistance.levenshtein.distance(src.split(), tgt.split()))
        return_status['edit_distance'] = np.mean(edit_distances)
    return return_status


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_dataset", type=str, default="USPTO-50K")
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--sample', type=str, default='prob', choices=['prob', 'random', 'cano'])
    parser.add_argument("--augmentation", type=int, default=20)
    parser.add_argument("--with_class", action="store_true")
    parser.add_argument("--seed", type=int, default=33)
    parser.add_argument("--processes", type=int, default=-1)
    parser.add_argument("--character", action="store_true")
    parser.add_argument("--canonical", action="store_true")
    parser.add_argument("--topn", type=int, default=10)
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()
    print('preprocessing dataset {}...'.format(args.task_dataset))
    lcs_byproduct_prob_dict_file = os.path.join(r'../%s' % args.task_dataset, r'dataset/stage_two/lcs_byproduct_prob.dict')
    datadir = os.path.join(r'../%s' % args.task_dataset, r'dataset/stage_one')
    resultsdir = os.path.join(r'../%s' % args.task_dataset, r'results/stage_one/with_class') if args.with_class else os.path.join(r'../%s' % args.task_dataset, r'results/stage_one/without_class')

    if args.mode == 'test':
        datasets = ['test', 'eval']
    else:
        datasets = ['test', 'eval', 'train']
        args.topn = 1
    print(args)
    random.seed(args.seed)

    if args.sample == 'prob':
        if os.path.exists(lcs_byproduct_prob_dict_file):
            lcs_byproduct_prob_dict = joblib.load(lcs_byproduct_prob_dict_file)
        else:
            print('Compute LCS dictionary from Train dataset')
            lcs_byproduct = prob_sample_train(args)
            lcs_byproduct_prob_dict = analyse_save(lcs_byproduct)

    print('%s select by-product SMILES' % args.sample)

    for i, data_set in enumerate(datasets):
        csv_path = '%s/%s/byproduct_rxn_%s.csv' % (datadir, data_set, data_set)
        csv = pd.read_csv(csv_path)
        reaction_list = list(csv["reactants>reagents>production"])
        reaction_list = np.repeat(np.array(reaction_list), args.topn).tolist()
        rxns_class_list = list(csv["class"])
        rxns_class_list = np.repeat(np.array(rxns_class_list), args.topn).tolist()

        if data_set == 'train':
            byproduct_topn_list = list(csv["byproduct"])
        else:
            if args.topn == 1:
                byproduct_topn_list = list(csv["byproduct"])
            else:
                with open(os.path.join(resultsdir, '%s_top%s_smiles.txt' % (data_set, args.topn)), 'r+') as f:
                    byproduct_topn_list = f.readlines()
                    byproduct_topn_list = [byproduct_topn.strip() for byproduct_topn in byproduct_topn_list]

        # if data_set == 'test':
        #     assert len(byproduct_topn_list) == 5007 * args.topn
        # elif data_set == 'eval':
        #     assert len(byproduct_topn_list) == 5001 * args.topn

        reactant_smarts_list = list(
            map(lambda x: x.split('>')[0], reaction_list))
        reactant_smarts_list = list(
            map(lambda x: x.split(' ')[0], reactant_smarts_list))
        reagent_smarts_list = list(
            map(lambda x: x.split('>')[1], reaction_list))
        product_smarts_list = list(
            map(lambda x: x.split('>')[2], reaction_list))
        product_smarts_list = list(
            map(lambda x: x.split(' ')[0], product_smarts_list))  # remove ' |f:1...'
        byproduct_smarts_list = byproduct_topn_list

        print("Total Data Size", len(reaction_list))

        sub_react_list = reactant_smarts_list
        sub_prod_list = product_smarts_list
        sub_byproduct_list = byproduct_smarts_list

        src_data, tgt_data, byproduct_src_data = preprocess(
            sub_react_list,
            sub_prod_list,
            sub_byproduct_list,
            rxns_class_list,
            data_set,
        )
