from rdkit import Chem
import pandas as pd
import joblib

def read_lcs_side(data_path =r'../dataset/train/src_lcs-train.txt'):
    lcs_side = []
    with open(data_path,'r+') as r:
        for line in r.readlines():
            lcs_side.append(line.strip())
    return lcs_side


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

    joblib.dump(side_prob_dict, r'../dataset/lcs_side_prob.dict')







if __name__ == '__main__':
    lcs_side = read_lcs_side()
    analyse_save(lcs_side)