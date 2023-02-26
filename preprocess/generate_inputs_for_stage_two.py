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


def analyse_save(lcs_side):
    cano_lcs_side = [Chem.MolToSmiles(Chem.MolFromSmiles(side)) for side in lcs_side]
    lcs_side_df = pd.DataFrame()
    lcs_side_df['cano_lcs_side'] = cano_lcs_side
    lcs_side_df['lcs_side'] = lcs_side
    side_prob_dict = {}
    groups = lcs_side_df.groupby(['cano_lcs_side'])
    for group in groups:
        cano_lcs = group[0]
        side_prob_dict[cano_lcs] = {}
        lcs_side_list = group[1]['lcs_side'].tolist()
        side_list = set(lcs_side_list)
        for side in side_list:
            side_prob_dict[cano_lcs][side] = lcs_side_list.count(side) / len(lcs_side_list)

    joblib.dump(side_prob_dict, os.path.join(r'../', args.exp_id, 'dataset/stage_two/lcs_side_prob.dict'))
    return side_prob_dict


def prob_sample_train(args):
    save_dir = os.path.join(r'../', args.exp_id, 'dataset/stage_two')
    csv_path = os.path.join(r'../', args.exp_id, 'dataset/stage_one/train/side_product_rxn_train.csv')
    csv = pd.read_csv(csv_path)
    reaction_list = list(csv["reactants>reagents>production"])
    sideprod_list = list(csv["sideprod"])
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
    sideprods = sideprod_list

    data = [{
        "reactant": i,
        "product": j,
        "sideprod": n,
        "augmentation": args.augmentation,
        "root_aligned": not args.canonical,
    } for i, j, n in zip(reactants, products, sideprods)]

    lcs_side = []
    tgt_lcs_data = []
    sideprod_src_lcs_data = []
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
        lcs_side.extend(result['lcs_side'])
        sideprod_src_lcs_data.extend(result['sideprod_src_data'])
        tgt_lcs_data.extend(result['tgt_data'])

    with open(
            os.path.join(save_dir, 'tgt_lcs-train.txt'), 'w') as f:
        for tgt in tgt_lcs_data:
            f.write('{}\n'.format(tgt))

    with open(
            os.path.join(save_dir, 'sideprod_src_lcs-train.txt'), 'w') as f:
        for sideprod_src_lcs in sideprod_src_lcs_data:
            f.write('{}\n'.format(sideprod_src_lcs))
    return lcs_side


def prob_sample_multi_process(data):
    pt = re.compile(r':(\d+)]')
    product = data['product']
    reactant = data['reactant']
    sideprod = data['sideprod']
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
        "sideprod_src_data": [],
        "edit_distance": 0,
        "sideprod_edit_distance": 0,
        "lcs_side": [],
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
        if sideprod not in ['<eos>']:
            sideprod_mol = Chem.MolFromSmiles(sideprod)
            for src, tgt in zip(return_status['src_data'], return_status['tgt_data']):
                # sideprod_edit_distances = []
                min_dis = 10000
                min_sideprod = ''
                min_sideprod_list = []
                max_lcs_list = []
                if '.' in sideprod:
                    sideprod_list = sideprod.split('.')
                    lenth = len(sideprod_list)
                    sideprod_smi_list = [[] for _ in range(lenth)]
                    for n in range(lenth):
                        for n1 in range(Chem.MolFromSmiles(sideprod_list[n]).GetNumAtoms()):
                            sideprod_smi_list[n].append(
                                clear_map_canonical_smiles(sideprod_list[n], canonical=True, root=n1))
                    sideprod_comb_list = list(itertools.product(*sideprod_smi_list))
                    sideprod_comb_list = [('.').join(list(sideprod_comb)) for sideprod_comb in sideprod_comb_list]
                    for sideprod_smi in sideprod_comb_list:
                        sideprod_src = src.split()
                        sideprod_src.append('.')
                        sideprod_src.extend(smi_tokenizer(sideprod_smi).split())
                        distance = textdistance.levenshtein.distance(sideprod_src, tgt.split())
                        if distance <= min_dis:
                            min_dis = distance
                            min_sideprod = sideprod_smi
                    min_sideprod_list.append(min_sideprod)
                    for i in min_sideprod_list:
                        lcs_lenth = find_lcs_len(i, tgt.replace(' ', ''))
                        max_lcs_list.append(lcs_lenth)
                    assert len(max_lcs_list) == len(min_sideprod_list)
                    min_sideprod = min_sideprod_list[max_lcs_list.index(max(max_lcs_list))]
                    # min_sideprod = random.choice(sideprod_comb_list)
                    sideprod_src = src.replace(' ', '') + '.' + min_sideprod
                    return_status['sideprod_src_data'].append(smi_tokenizer(sideprod_src))
                    # sideprod_edit_distances.append(min_dis)
                    return_status['lcs_side'].append(min_sideprod)
                else:
                    for i in range(sideprod_mol.GetNumAtoms()):
                        sideprod_smi = clear_map_canonical_smiles(sideprod, canonical=True, root=i)
                        sideprod_src = src.split()
                        sideprod_src.append('.')
                        sideprod_src.extend(smi_tokenizer(sideprod_smi).split())
                        distance = textdistance.levenshtein.distance(sideprod_src, tgt.split())
                        if distance < min_dis:
                            min_dis = distance
                            min_sideprod = sideprod_smi
                        elif distance == min_dis:
                            min_sideprod_list.append(sideprod_smi)
                    min_sideprod_list.append(min_sideprod)
                    for i in min_sideprod_list:
                        lcs_lenth = find_lcs_len(i, tgt.replace(' ', ''))
                        max_lcs_list.append(lcs_lenth)
                    assert len(max_lcs_list) == len(min_sideprod_list)
                    min_sideprod = min_sideprod_list[max_lcs_list.index(max(max_lcs_list))]

                    # sideprod_list = [clear_map_canonical_smiles(sideprod, canonical=True, root=n1) for n1 in range(Chem.MolFromSmiles(sideprod).GetNumAtoms())]
                    # min_sideprod = random.choice(sideprod_list)
                    sideprod_src = src.replace(' ', '') + '.' + min_sideprod
                    return_status['sideprod_src_data'].append(smi_tokenizer(sideprod_src))
                    return_status['lcs_side'].append(min_sideprod)
                    # sideprod_edit_distances.append(min_dis)
                # assert min_dis < 10000
                # return_status['sideprod_edit_distance'] = np.mean(sideprod_edit_distances)
        else:
            return_status['sideprod_src_data'] = return_status['src_data']
        edit_distances = []
        for src, tgt in zip(return_status['src_data'], return_status['tgt_data']):
            edit_distances.append(textdistance.levenshtein.distance(src.split(), tgt.split()))

        return_status['edit_distance'] = np.mean(edit_distances)
    return return_status

def prob_sample_multi_process_2(data):
    pt = re.compile(r':(\d+)]')
    product = data['product']
    reactant = data['reactant']
    sideprod = data['sideprod']
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
        "sideprod_src_data": [],
        "edit_distance": 0,
        "sideprod_edit_distance": 0,
        "lcs_side": [],
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
        if sideprod not in ['<eos>']:
            sideprod_mol = Chem.MolFromSmiles(sideprod)
            for src, tgt in zip(return_status['src_data'], return_status['tgt_data']):
                # sideprod_edit_distances = []
                # min_dis = 10000
                # min_sideprod = ''
                # min_sideprod_list = []
                max_lcs_list = []
                if '.' in sideprod:
                    sideprod_list = sideprod.split('.')
                    lenth = len(sideprod_list)
                    sideprod_smi_list = [[] for _ in range(lenth)]
                    for n in range(lenth):
                        for n1 in range(Chem.MolFromSmiles(sideprod_list[n]).GetNumAtoms()):
                            sideprod_smi_list[n].append(
                                clear_map_canonical_smiles(sideprod_list[n], canonical=True, root=n1))
                    sideprod_comb_list = list(itertools.product(*sideprod_smi_list))
                    sideprod_comb_list = [('.').join(list(sideprod_comb)) for sideprod_comb in sideprod_comb_list]
                    # for sideprod_smi in sideprod_comb_list:
                    #     sideprod_src = src.split()
                    #     sideprod_src.append('.')
                    #     sideprod_src.extend(smi_tokenizer(sideprod_smi).split())
                    #     distance = textdistance.levenshtein.distance(sideprod_src, tgt.split())
                    #     if distance <= min_dis:
                    #         min_dis = distance
                    #         min_sideprod = sideprod_smi
                    # min_sideprod_list.append(min_sideprod)
                    for i in sideprod_comb_list:
                        # lcs_lenth = find_lcs_len(i, tgt.replace(' ', ''))
                        lcs_lenth = textdistance.lcsseq.distance(i, tgt.replace(' ', ''))
                        max_lcs_list.append(lcs_lenth)
                    # assert len(max_lcs_list) == len(min_sideprod_list)
                    lcs_sideprod = sideprod_comb_list[max_lcs_list.index(max(max_lcs_list))]
                    # min_sideprod = random.choice(sideprod_comb_list)
                    sideprod_src = src.replace(' ', '') + '.' + lcs_sideprod
                    return_status['sideprod_src_data'].append(smi_tokenizer(sideprod_src))
                    # sideprod_edit_distances.append(min_dis)
                    return_status['lcs_side'].append(lcs_sideprod)
                else:
                    sideprod_list = []
                    for i in range(sideprod_mol.GetNumAtoms()):
                        sideprod_smi = clear_map_canonical_smiles(sideprod, canonical=True, root=i)
                        sideprod_list.append(sideprod_smi)
                        # sideprod_src = src.split()
                        # sideprod_src.append('.')
                        # sideprod_src.extend(smi_tokenizer(sideprod_smi).split())
                        # distance = textdistance.levenshtein.distance(sideprod_src, tgt.split())
                        # if distance < min_dis:
                        #     min_dis = distance
                        #     min_sideprod = sideprod_smi
                        # elif distance == min_dis:
                        #     min_sideprod_list.append(sideprod_smi)
                    # min_sideprod_list.append(min_sideprod)
                    for i in sideprod_list:
                        lcs_lenth = textdistance.lcsstr.distance(i, tgt.replace(' ', ''))
                        max_lcs_list.append(lcs_lenth)
                    # assert len(max_lcs_list) == len(min_sideprod_list)
                    lcs_sideprod = sideprod_list[max_lcs_list.index(max(max_lcs_list))]

                    # sideprod_list = [clear_map_canonical_smiles(sideprod, canonical=True, root=n1) for n1 in range(Chem.MolFromSmiles(sideprod).GetNumAtoms())]
                    # min_sideprod = random.choice(sideprod_list)
                    sideprod_src = src.replace(' ', '') + '.' + lcs_sideprod
                    return_status['sideprod_src_data'].append(smi_tokenizer(sideprod_src))
                    return_status['lcs_side'].append(lcs_sideprod)
                    # sideprod_edit_distances.append(min_dis)
                # assert min_dis < 10000
                # return_status['sideprod_edit_distance'] = np.mean(sideprod_edit_distances)
        else:
            return_status['sideprod_src_data'] = return_status['src_data']
        edit_distances = []
        for src, tgt in zip(return_status['src_data'], return_status['tgt_data']):
            edit_distances.append(textdistance.levenshtein.distance(src.split(), tgt.split()))

        return_status['edit_distance'] = np.mean(edit_distances)
    return return_status

def prob_sample_multi_process_3(data):
    pt = re.compile(r':(\d+)]')
    product = data['product']
    reactant = data['reactant']
    sideprod = data['sideprod']
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
        "sideprod_src_data": [],
        "edit_distance": 0,
        "sideprod_edit_distance": 0,
        "lcs_side": [],
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
        if sideprod not in ['<eos>']:
            sideprod_mol = Chem.MolFromSmiles(sideprod)
            for src, tgt in zip(return_status['src_data'], return_status['tgt_data']):
                # sideprod_edit_distances = []
                min_dis = 10000
                min_sideprod = ''
                min_sideprod_list = []
                max_lcs_list = []
                if '.' in sideprod:
                    sideprod_list = sideprod.split('.')
                    lenth = len(sideprod_list)
                    sideprod_smi_list = [[] for _ in range(lenth)]
                    for n in range(lenth):
                        for n1 in range(Chem.MolFromSmiles(sideprod_list[n]).GetNumAtoms()):
                            sideprod_smi_list[n].append(
                                clear_map_canonical_smiles(sideprod_list[n], canonical=True, root=n1))
                    sideprod_comb_list = list(itertools.product(*sideprod_smi_list))
                    sideprod_comb_list = [('.').join(list(sideprod_comb)) for sideprod_comb in sideprod_comb_list]
                    for sideprod_smi in sideprod_comb_list:
                        sideprod_src = src.split()
                        sideprod_src.append('.')
                        sideprod_src.extend(smi_tokenizer(sideprod_smi).split())
                        distance = textdistance.levenshtein.distance(sideprod_src, tgt.split())
                        if distance <= min_dis:
                            min_dis = distance
                            min_sideprod = sideprod_smi
                    min_sideprod_list.append(min_sideprod)
                    for i in min_sideprod_list:
                        lcs_lenth = textdistance.lcsstr.distance(i, tgt.replace(' ', ''))
                        max_lcs_list.append(lcs_lenth)
                    assert len(max_lcs_list) == len(min_sideprod_list)
                    min_sideprod = min_sideprod_list[max_lcs_list.index(max(max_lcs_list))]
                    # min_sideprod = random.choice(sideprod_comb_list)
                    sideprod_src = src.replace(' ', '') + '.' + min_sideprod
                    return_status['sideprod_src_data'].append(smi_tokenizer(sideprod_src))
                    # sideprod_edit_distances.append(min_dis)
                    return_status['lcs_side'].append(min_sideprod)
                else:
                    for i in range(sideprod_mol.GetNumAtoms()):
                        sideprod_smi = clear_map_canonical_smiles(sideprod, canonical=True, root=i)
                        sideprod_src = src.split()
                        sideprod_src.append('.')
                        sideprod_src.extend(smi_tokenizer(sideprod_smi).split())
                        distance = textdistance.levenshtein.distance(sideprod_src, tgt.split())
                        if distance < min_dis:
                            min_dis = distance
                            min_sideprod = sideprod_smi
                        elif distance == min_dis:
                            min_sideprod_list.append(sideprod_smi)
                    min_sideprod_list.append(min_sideprod)
                    for i in min_sideprod_list:
                        lcs_lenth = textdistance.lcsstr.distance(i, tgt.replace(' ', ''))
                        max_lcs_list.append(lcs_lenth)
                    assert len(max_lcs_list) == len(min_sideprod_list)
                    min_sideprod = min_sideprod_list[max_lcs_list.index(max(max_lcs_list))]

                    # sideprod_list = [clear_map_canonical_smiles(sideprod, canonical=True, root=n1) for n1 in range(Chem.MolFromSmiles(sideprod).GetNumAtoms())]
                    # min_sideprod = random.choice(sideprod_list)
                    sideprod_src = src.replace(' ', '') + '.' + min_sideprod
                    return_status['sideprod_src_data'].append(smi_tokenizer(sideprod_src))
                    return_status['lcs_side'].append(min_sideprod)
                    # sideprod_edit_distances.append(min_dis)
                # assert min_dis < 10000
                # return_status['sideprod_edit_distance'] = np.mean(sideprod_edit_distances)
        else:
            return_status['sideprod_src_data'] = return_status['src_data']
        edit_distances = []
        for src, tgt in zip(return_status['src_data'], return_status['tgt_data']):
            edit_distances.append(textdistance.levenshtein.distance(src.split(), tgt.split()))

        return_status['edit_distance'] = np.mean(edit_distances)
    return return_status

def prob_sample_multi_process_4(data):
    pt = re.compile(r':(\d+)]')
    product = data['product']
    reactant = data['reactant']
    sideprod = data['sideprod']
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
        "sideprod_src_data": [],
        "edit_distance": 0,
        "sideprod_edit_distance": 0,
        "lcs_side": [],
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
    if sideprod == 'CC(=O)O':
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
        if sideprod not in ['<eos>']:
            sideprod_mol = Chem.MolFromSmiles(sideprod)
            if sideprod_mol.GetNumAtoms() == 1:
                sideprod_smi = sideprod
            else:
                if '.' in sideprod:
                    sideprod_list = sideprod.split('.')
                    lenth = len(sideprod_list)
                    sideprod_smi_list = [[] for _ in range(lenth)]
                    for n in range(lenth):
                        for n1 in range(Chem.MolFromSmiles(sideprod_list[n]).GetNumAtoms()):
                            sideprod_smi_list[n].append(
                                clear_map_canonical_smiles(sideprod_list[n], canonical=True, root=n1))
                    sideprod_comb_list = list(itertools.product(*sideprod_smi_list))
                    sideprod_smi_list = [('.').join(list(sideprod_comb)) for sideprod_comb in sideprod_comb_list]
                else:
                    sideprod_smi_list = []
                    for i in range(sideprod_mol.GetNumAtoms()):
                        sideprod_smi = clear_map_canonical_smiles(sideprod, canonical=True, root=i)
                        sideprod_smi_list.append(sideprod_smi)

                lcs_distance_smiles_list = []
                for tgt in return_status['tgt_data']:
                    lcs_distance_list = []
                    for sideprod_smi in sideprod_smi_list:
                        lcs_distance = len(sideprod_smi) - textdistance.lcsstr.similarity(sideprod_smi, tgt.replace(' ', ''))
                        lcs_distance_list.append(lcs_distance)

                    lcs_distance_smiles_list.append(lcs_distance_list)
                lcs_distance_smiles = np.array(lcs_distance_smiles_list)
                lcs_distance_smiles = (lcs_distance_smiles == lcs_distance_smiles.min(axis=1, keepdims=1)).astype(float)
                value = np.sum(lcs_distance_smiles, axis=0)
                max_lcs_index = np.where(value == np.max(value))
                if len(max_lcs_index) > 1:
                    print('eeeee')
                sideprod_smi = sideprod_smi_list[max_lcs_index[0][0]]

            for src, tgt in zip(return_status['src_data'], return_status['tgt_data']):
                sideprod_src = src.replace(' ', '') + '.' + sideprod_smi
                return_status['sideprod_src_data'].append(smi_tokenizer(sideprod_src))
                return_status['lcs_side'].append(sideprod_smi)
        else:
            return_status['sideprod_src_data'] = return_status['src_data']
        edit_distances = []
        for src, tgt in zip(return_status['src_data'], return_status['tgt_data']):
            edit_distances.append(textdistance.levenshtein.distance(src.split(), tgt.split()))

        return_status['edit_distance'] = np.mean(edit_distances)
    return return_status

def prob_sample_multi_process_5(data):
    pt = re.compile(r':(\d+)]')
    product = data['product']
    reactant = data['reactant']
    sideprod = data['sideprod']
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
        "sideprod_src_data": [],
        "edit_distance": 0,
        "sideprod_edit_distance": 0,
        "lcs_side": [],
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
        if sideprod not in ['<eos>']:
            sideprod_mol = Chem.MolFromSmiles(sideprod)
            if sideprod_mol.GetNumAtoms() == 1:
                sideprod_smi = sideprod
            else:
                if '.' in sideprod:
                    sideprod_list = sideprod.split('.')
                    lenth = len(sideprod_list)
                    sideprod_smi_list = [[] for _ in range(lenth)]
                    for n in range(lenth):
                        for n1 in range(Chem.MolFromSmiles(sideprod_list[n]).GetNumAtoms()):
                            sideprod_smi_list[n].append(
                                clear_map_canonical_smiles(sideprod_list[n], canonical=True, root=n1))
                    sideprod_comb_list = list(itertools.product(*sideprod_smi_list))
                    sideprod_smi_list = [('.').join(list(sideprod_comb)) for sideprod_comb in sideprod_comb_list]
                else:
                    sideprod_smi_list = []
                    for i in range(sideprod_mol.GetNumAtoms()):
                        sideprod_smi = clear_map_canonical_smiles(sideprod, canonical=True, root=i)
                        sideprod_smi_list.append(sideprod_smi)
                if len(list(set(sideprod_smi_list))) == 1:
                    sideprod_smi = sideprod_smi_list[0]
                else:
                    lcs_distance_smiles_list = []
                    for tgt in return_status['tgt_data']:
                        lcs_distance_list = []
                        rule = str.maketrans('', '', digits)
                        for sideprod_smi in sideprod_smi_list:
                            o_number_side = sideprod_smi.translate(rule)
                            o_number_tgt = tgt.replace(' ', '').translate(rule)
                            lcs_distance = len(o_number_side) - textdistance.lcsstr.similarity(o_number_side, o_number_tgt)
                            lcs_distance_list.append(lcs_distance)

                        lcs_distance_smiles_list.append(lcs_distance_list)
                    lcs_distance_smiles = np.array(lcs_distance_smiles_list)
                    lcs_distance_smiles = (lcs_distance_smiles == lcs_distance_smiles.min(axis=1, keepdims=1)).astype(float)
                    value = np.sum(lcs_distance_smiles, axis=0)
                    max_lcs_index = np.where(value == np.max(value))
                    # if len(max_lcs_index[0]) > 1:
                    #     print('eeeee')
                    sideprod_smi = sideprod_smi_list[max_lcs_index[0][0]]

            for src, tgt in zip(return_status['src_data'], return_status['tgt_data']):
                sideprod_src = src.replace(' ', '') + '.' + sideprod_smi
                return_status['sideprod_src_data'].append(smi_tokenizer(sideprod_src))
                return_status['lcs_side'].append(sideprod_smi)
        else:
            return_status['sideprod_src_data'] = return_status['src_data']
        edit_distances = []
        for src, tgt in zip(return_status['src_data'], return_status['tgt_data']):
            edit_distances.append(textdistance.levenshtein.distance(src.split(), tgt.split()))

        return_status['edit_distance'] = np.mean(edit_distances)
    return return_status

def prob_sample_multi_process_6(data):

    product = data['product']
    reactant = data['reactant']
    sideprod = data['sideprod']
    augmentation = data['augmentation']
    pro_mol = Chem.MolFromSmiles(product)
    rea_mol = Chem.MolFromSmiles(reactant)
    """checking data quality"""
    return_status = {
        "status": 0,
        "src_data": [],
        "tgt_data": [],
        "sideprod_src_data": [],
        "edit_distance": 0,
        "sideprod_edit_distance": 0,
        "lcs_side": [],
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

        if sideprod not in ['<eos>']:
            sideprod_mol = Chem.MolFromSmiles(sideprod)

            if '.' in sideprod:
                sideprod_list = sideprod.split('.')
                lenth = len(sideprod_list)
                sideprod_smi_list = [[] for _ in range(lenth)]
                for n in range(lenth):
                    for n1 in range(Chem.MolFromSmiles(sideprod_list[n]).GetNumAtoms()):
                        sideprod_smi_list[n].append(
                            clear_map_canonical_smiles(sideprod_list[n], canonical=True, root=n1))
                sideprod_comb_list = list(itertools.product(*sideprod_smi_list))
                sideprod_list = [('.').join(list(sideprod_comb)) for sideprod_comb in sideprod_comb_list]
            else:
                sideprod_list = []
                for i in range(sideprod_mol.GetNumAtoms()):
                    sideprod_smi = clear_map_canonical_smiles(sideprod, canonical=True, root=i)
                    sideprod_list.append(sideprod_smi)
            sideprod_list = list(set(sideprod_list))
            for src, tgt in zip(return_status['src_data'], return_status['tgt_data']):
                max_lcs_list = []
                distance_list = []

                for sideprod_smi in sideprod_list:
                    new_src = src.split()
                    new_src.append('.')
                    new_src.extend(smi_tokenizer(sideprod_smi).split())
                    distance = textdistance.levenshtein.distance(new_src, tgt.split())
                    distance_list.append(distance)
                min_distance = min(distance_list)
                min_distance_idx = [idx for idx, distance in enumerate(distance_list) if distance == min_distance]
                min_sideprod_list = [sideprod_list[idx] for idx in min_distance_idx]
                rule = str.maketrans('', '', digits)
                for i in min_sideprod_list:
                    o_number_i = i.translate(rule)
                    o_number_tgt = tgt.replace(' ', '').translate(rule)
                    lcs_distance = len(o_number_i) - textdistance.lcsstr.similarity(o_number_i, o_number_tgt)
                    # lcs_lenth = find_lcs_len(i, tgt.replace(' ', ''))
                    max_lcs_list.append(lcs_distance)
                assert len(max_lcs_list) == len(min_sideprod_list)
                min_sideprod = min_sideprod_list[max_lcs_list.index(min(max_lcs_list))]
                sideprod_src = src.replace(' ', '') + '.' + min_sideprod
                return_status['sideprod_src_data'].append(smi_tokenizer(sideprod_src))
                return_status['lcs_side'].append(min_sideprod)
        else:
            return_status['sideprod_src_data'] = return_status['src_data']

        edit_distances = []
        for src, tgt in zip(return_status['src_data'], return_status['tgt_data']):
            edit_distances.append(textdistance.levenshtein.distance(src.split(), tgt.split()))

        return_status['edit_distance'] = np.mean(edit_distances)
    return return_status

def multi_process(data):
    global args
    product = data['product']
    reactant = data['reactant']
    sideprod = data['sideprod']
    rxn_class = data['rxn_class']
    augmentation = data['augmentation']
    pro_mol = Chem.MolFromSmiles(product)
    rea_mol = Chem.MolFromSmiles(reactant)
    """checking data quality"""
    return_status = {
        "status": 0,
        "src_data": [],
        "tgt_data": [],
        "sideprod_src_data": [],
        "use_class_src_data": [],
        "edit_distance": 0,
        "sideprod_edit_distance": 0,
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

        if sideprod not in ['<eos>']:
            for src in return_status['src_data']:
                try:
                    if args.sample in ['random', 'prob']:
                        side_prob_dict = lcs_side_prob_dict[sideprod]
                        weights = list(side_prob_dict.values()) if args.sample == 'prob' else None
                        random_sideprod = \
                            random.choices(list(side_prob_dict.keys()), weights=weights, k=1)[-1]
                    else:
                        assert args.sample == 'cano'
                        random_sideprod = clear_map_canonical_smiles(sideprod)
                except:
                    random_sideprod = sideprod
                sideprod_src = src.replace(' ', '') + '.' + random_sideprod
                return_status['sideprod_src_data'].append(smi_tokenizer(sideprod_src))
        else:
            return_status['sideprod_src_data'] = return_status['src_data']
        edit_distances = []
        for src, tgt in zip(return_status['src_data'], return_status['tgt_data']):
            edit_distances.append(textdistance.levenshtein.distance(src.split(), tgt.split()))
        return_status['edit_distance'] = np.mean(edit_distances)
        sideprod_edit_distances = []
        for src, tgt in zip(return_status['sideprod_src_data'], return_status['tgt_data']):
            sideprod_edit_distances.append(textdistance.levenshtein.distance(src.split(), tgt.split()))
        return_status['sideprod_edit_distances'] = np.mean(sideprod_edit_distances)

        for i in return_status['sideprod_src_data']:
            return_status['use_class_src_data'].append('RXN_%s %s' % (rxn_class, i))

    return return_status

def multi_process_2(data):
    global args
    product = data['product']
    reactant = data['reactant']
    sideprod = data['sideprod']
    rxn_class = data['rxn_class']
    augmentation = data['augmentation']
    pro_mol = Chem.MolFromSmiles(product)
    rea_mol = Chem.MolFromSmiles(reactant)
    """checking data quality"""
    return_status = {
        "status": 0,
        "src_data": [],
        "tgt_data": [],
        "sideprod_src_data": [],
        "use_class_src_data": [],
        "edit_distance": 0,
        "sideprod_edit_distance": 0,
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

        if sideprod not in ['<eos>']:
            try:
                if args.sample == 'prob':
                    side_prob_dict = lcs_side_prob_dict[sideprod]
                    weights = list(side_prob_dict.values())
                    sideprod = random.choices(list(side_prob_dict.keys()), weights=weights, k=1)[-1]
                    return_status['sideprod_src_data'] = [smi_tokenizer(src.replace(' ', '') + '.' + sideprod) for src
                                                          in return_status['src_data']]
                elif args.sample == 'random':
                    side_prod_mol = Chem.MolFromSmiles(sideprod)
                    if '.' in sideprod:
                        sideprod_list = sideprod.split('.')
                        lenth = len(sideprod_list)
                        sideprod_smi_list = [[] for _ in range(lenth)]
                        for n in range(lenth):
                            for n1 in range(Chem.MolFromSmiles(sideprod_list[n]).GetNumAtoms()):
                                sideprod_smi_list[n].append(
                                    clear_map_canonical_smiles(sideprod_list[n], canonical=True, root=n1))
                        sideprod_comb_list = list(itertools.product(*sideprod_smi_list))
                        sideprod_list = [('.').join(list(sideprod_comb)) for sideprod_comb in sideprod_comb_list]
                    else:
                        sideprod_list = []
                        for i in range(side_prod_mol.GetNumAtoms()):
                            sideprod_smi = clear_map_canonical_smiles(sideprod, canonical=True, root=i)
                            sideprod_list.append(sideprod_smi)
                    for src in return_status['src_data']:
                        sideprod = random.choice(sideprod_list)
                        return_status['sideprod_src_data'].append(smi_tokenizer(src.replace(' ', '') + '.' + sideprod))
                else:
                    assert args.sample == 'cano'
                    sideprod = clear_map_canonical_smiles(sideprod)
                    return_status['sideprod_src_data'] = [smi_tokenizer(src.replace(' ', '') + '.' + sideprod) for src
                                                          in return_status['src_data']]
            except Exception as e:
                print(e)
                pass
        else:
            return_status['sideprod_src_data'] = return_status['src_data']
        edit_distances = []
        for src, tgt in zip(return_status['src_data'], return_status['tgt_data']):
            edit_distances.append(textdistance.levenshtein.distance(src.split(), tgt.split()))
        return_status['edit_distance'] = np.mean(edit_distances)
        sideprod_edit_distances = []
        for src, tgt in zip(return_status['sideprod_src_data'], return_status['tgt_data']):
            sideprod_edit_distances.append(textdistance.levenshtein.distance(src.split(), tgt.split()))
        return_status['sideprod_edit_distances'] = np.mean(sideprod_edit_distances)

        for i in return_status['sideprod_src_data']:
            return_status['use_class_src_data'].append('RXN_%s %s' % (rxn_class, i))

    return return_status

def preprocess(reactants, products, sideprods, rxns_class_list, set_name):

    global args

    augmentation = args.augmentation
    processes = args.processes

    data = [{
        "reactant": i,
        "product": j,
        "sideprod": n,
        "rxn_class": r,
        "augmentation": augmentation,
    } for i, j, n, r in zip(reactants, products, sideprods, rxns_class_list)]

    src_data = []
    tgt_data = []
    sideprod_src_data = []
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
    sideprod_edit_distances = []
    for result in tqdm(results):
        if result['status'] != 0:
            skip_dict[result['status']] += 1
            continue
        edit_distances.append(result['edit_distance'])
        sideprod_edit_distances.append(result['sideprod_edit_distances'])
        src_data.extend(result['src_data'])
        tgt_data.extend(result['tgt_data'])
        sideprod_src_data.extend(result['sideprod_src_data'])
        use_class_src_data.extend(result['use_class_src_data'])

    print("Avg. edit distance:", np.mean(edit_distances))
    print("Avg. side_prod edit distance:", np.mean(sideprod_edit_distances))
    print('size', len(sideprod_src_data))
    # for key, value in skip_dict.items():
    #     print(f"{key}:{value},{value / len(reactants)}")
    if args.topn != 1:
        save_name = 'top%s_' % args.topn
    else:
        save_name = ''

    if args.mode == 'test':
        save_dir = os.path.join(r'../%s' % args.exp_id, r'dataset/stage_two/with_class') if args.with_class else os.path.join(r'../%s' % args.exp_id, r'dataset/stage_two/without_class')
    else:
        save_dir = os.path.join(r'../%s' % args.exp_id, r'dataset/stage_two/%s' % set_name)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


    with open(os.path.join(save_dir, '{}tgt-{}.txt'.format(save_name, set_name)), 'w') as f:
        for tgt in tgt_data:
            f.write('{}\n'.format(tgt))

    if args.mode != 'test' or not args.with_class:
        with open(os.path.join(save_dir, '{}sideprod_src-{}.txt'.format(save_name, set_name)), 'w') as f:
            for sideprod_src in sideprod_src_data:
                f.write('{}\n'.format(sideprod_src))

    if args.mode != 'test' or args.with_class:
        with open(os.path.join(save_dir, '{}sideprod_src-{}_with_class.txt'.format(save_name, set_name)), 'w') as f:
            for use_class_src in use_class_src_data:
                f.write('{}\n'.format(use_class_src))

    return src_data, tgt_data, sideprod_src_data


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
    sideprod = data['sideprod']
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
        "sideprod_src_data": [],
        "edit_distance": 0,
        "sideprod_edit_distance": 0,
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
        if sideprod not in ['<eos>', '<unk>']:
            sideprod_mol = Chem.MolFromSmiles(sideprod)
            for src, tgt in zip(return_status['src_data'], return_status['tgt_data']):
                sideprod_edit_distances = []
                min_dis = 10000
                min_sideprod = ''
                min_sideprod_list = []
                max_lcs_list = []
                if '.' in sideprod:
                    sideprod_list = sideprod.split('.')
                    lenth = len(sideprod_list)
                    sideprod_smi_list = [[] for _ in range(lenth)]
                    for n in range(lenth):
                        for n1 in range(Chem.MolFromSmiles(sideprod_list[n]).GetNumAtoms()):
                            sideprod_smi_list[n].append(
                                clear_map_canonical_smiles(sideprod_list[n], canonical=True, root=n1))
                    # sideprod_comb_list = combine(sideprod_smi_list)
                    sideprod_comb_list = list(itertools.product(*sideprod_smi_list))
                    sideprod_comb_list = [('.').join(list(sideprod_comb)) for sideprod_comb in sideprod_comb_list]
                    for sideprod_smi in list(set(sideprod_comb_list)):
                        sideprod_src = src.split()
                        sideprod_src.append('.')
                        sideprod_src.extend(smi_tokenizer(sideprod_smi).split())
                        distance = textdistance.levenshtein.distance(sideprod_src, tgt.split())
                        if distance < min_dis:
                            min_dis = distance
                            min_sideprod = sideprod_smi
                        elif distance == min_dis:
                            min_sideprod_list.append(sideprod_smi)
                    min_sideprod_list.append(min_sideprod)
                    for i in min_sideprod_list:
                        lcs_lenth = find_lcs_len(i, tgt.replace(' ', ''))
                        max_lcs_list.append(lcs_lenth)
                    assert len(max_lcs_list) == len(min_sideprod_list)
                    min_sideprod = min_sideprod_list[max_lcs_list.index(max(max_lcs_list))]
                    sideprod_src = src.replace(' ', '') + '.' + min_sideprod
                    return_status['sideprod_src_data'].append(smi_tokenizer(sideprod_src))
                    sideprod_edit_distances.append(min_dis)
                else:
                    for i in range(sideprod_mol.GetNumAtoms()):
                        sideprod_smi = clear_map_canonical_smiles(sideprod, canonical=True, root=i)
                        sideprod_src = src.split()
                        sideprod_src.append('.')
                        sideprod_src.extend(smi_tokenizer(sideprod_smi).split())
                        distance = textdistance.levenshtein.distance(sideprod_src, tgt.split())
                        if distance <= min_dis:
                            min_dis = distance
                            min_sideprod = sideprod_smi
                    min_sideprod_list.append(min_sideprod)
                    for i in min_sideprod_list:
                        lcs_lenth = find_lcs_len(i, tgt.replace(' ', ''))
                        max_lcs_list.append(lcs_lenth)
                    assert len(max_lcs_list) == len(min_sideprod_list)
                    min_sideprod = min_sideprod_list[max_lcs_list.index(max(max_lcs_list))]
                    sideprod_src = src.replace(' ', '') + '.' + min_sideprod
                    return_status['sideprod_src_data'].append(smi_tokenizer(sideprod_src))
                    sideprod_edit_distances.append(min_dis)
                assert min_dis < 10000
                return_status['sideprod_edit_distance'] = np.mean(sideprod_edit_distances)
        else:
            return_status['sideprod_src_data'] = return_status['src_data']
        edit_distances = []
        for src, tgt in zip(return_status['src_data'], return_status['tgt_data']):
            edit_distances.append(textdistance.levenshtein.distance(src.split(), tgt.split()))
        return_status['edit_distance'] = np.mean(edit_distances)
    return return_status


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='USPTO_50K')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--sample', type=str, default='prob', choices=['prob', 'random', 'cano'])
    parser.add_argument("--augmentation", type=int, default=20)
    parser.add_argument("--with_class", action="store_true")
    parser.add_argument("--seed", type=int, default=33)
    parser.add_argument("--processes", type=int, default=-1)
    parser.add_argument("--character", action="store_true")
    parser.add_argument("--canonical", action="store_true")
    parser.add_argument("--topn", type=int, default=10)
    parser.add_argument("--exp_id", type=str, default="")
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()
    print('preprocessing dataset {}...'.format(args.dataset))
    lcs_side_prob_dict_file = os.path.join(r'../%s' % args.exp_id, r'dataset/stage_two/lcs_side_prob.dict')
    datadir = os.path.join(r'../%s' % args.exp_id, r'dataset/stage_one')
    resultsdir = os.path.join(r'../%s' % args.exp_id, r'results/stage_one/with_class') if args.with_class else os.path.join(r'../%s' % args.exp_id, r'results/stage_one/without_class')

    if args.mode == 'test':
        datasets = ['test', 'eval']
    else:
        datasets = ['test', 'eval', 'train']
        args.topn = 1
    print(args)
    random.seed(args.seed)

    if args.sample == 'prob':
        if os.path.exists(lcs_side_prob_dict_file):
            lcs_side_prob_dict = joblib.load(lcs_side_prob_dict_file)
        else:
            print('Compute LCS dictionary from Train dataset')
            lcs_side = prob_sample_train(args)
            lcs_side_prob_dict = analyse_save(lcs_side)

    print('%s select side product SMILES' % args.sample)

    for i, data_set in enumerate(datasets):
        csv_path = '%s/%s/side_product_rxn_%s.csv' % (datadir, data_set, data_set)
        csv = pd.read_csv(csv_path)
        reaction_list = list(csv["reactants>reagents>production"])
        reaction_list = np.repeat(np.array(reaction_list), args.topn).tolist()
        rxns_class_list = list(csv["class"])
        rxns_class_list = np.repeat(np.array(rxns_class_list), args.topn).tolist()

        if data_set == 'train':
            side_topn_list = list(csv["sideprod"])
        else:
            if args.topn == 1:
                side_topn_list = list(csv["sideprod"])
            else:
                with open(os.path.join(resultsdir, '%s_top%s_smiles.txt' % (data_set, args.topn)), 'r+') as f:
                    side_topn_list = f.readlines()
                    side_topn_list = [side_topn.strip() for side_topn in side_topn_list]

        if data_set == 'test':
            assert len(side_topn_list) == 5007 * args.topn
        elif data_set == 'eval':
            assert len(side_topn_list) == 5001 * args.topn

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
        sideprod_smarts_list = side_topn_list

        print("Total Data Size", len(reaction_list))

        sub_react_list = reactant_smarts_list
        sub_prod_list = product_smarts_list
        sub_sideprod_list = sideprod_smarts_list

        src_data, tgt_data, sideprod_src_data = preprocess(
            sub_react_list,
            sub_prod_list,
            sub_sideprod_list,
            rxns_class_list,
            data_set,
        )
