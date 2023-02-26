import os
import pickle
import copy
import random
import numpy as np
from tqdm import trange
from scipy.spatial.distance import pdist, squareform

import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler

from .utils import update_scope, update_tensors


class RetroLogitsDatasets(Dataset):
    def __init__(self, root, data_split, use_rxn_class=False):
        self.root = root
        self.data_split = data_split
        self.data_dir = os.path.join(root, self.data_split)
        if use_rxn_class:
            self.data_path = self.data_dir + '/side_rxn_datas_with_class.pkl'
        else:
            self.data_path = self.data_dir + '/side_rxn_datas.pkl'
        with open(self.data_path, 'rb') as f:
            self.rxn_datas = pickle.load(f)

    def __getitem__(self, index):
        return self.rxn_datas[index]

    def __len__(self):
        return len(self.rxn_datas)

class RetroTestDatasets(Dataset):
    def __init__(self, dataset):
        self.rxn_datas = dataset

    def __getitem__(self, index):
        return self.rxn_datas[index]

    def __len__(self):
        return len(self.rxn_datas)


class EmbDatasets(Dataset):
    def __init__(self, embed, labels):
        self.embed = embed
        self.labels = labels

    def __getitem__(self, index):
        return (self.embed[index], self.labels[index])

    def __len__(self):
        return len(self.labels)


class HTBatchSampler(BatchSampler):
    def __init__(self, dataset, H_idx, T_idx, batch_size, T_split_number=0.1):
        self.dataset = dataset
        self.batch_size = batch_size
        self.H_idx = H_idx if isinstance(H_idx, list) else np.array(H_idx).tolist()
        self.T_idx = T_idx if isinstance(T_idx, list) else np.array(T_idx).tolist()

        self.n_T = int(batch_size * T_split_number)
        self.n_T = self.n_T if self.n_T >= 1 else 1

        self.H_epoch = len(self.H_idx) // (batch_size - self.n_T)
        self.n_lack_T = len(self.T_idx) - self.n_T * self.H_epoch
        self.n_remain_H = len(self.H_idx) - (self.batch_size - self.n_T) * self.H_epoch
        assert self.n_remain_H >= 0

    def __iter__(self):
        random.seed(self.epoch)
        np.random.seed(self.epoch)
        random.shuffle(self.H_idx)
        random.shuffle(self.T_idx)
        if self.n_lack_T >= 0:
            T_idx = self.T_idx[:-self.n_lack_T]
        else:
            T_idx = copy.deepcopy(self.T_idx)
            T_idx.extend(np.random.choice(self.T_idx, -self.n_lack_T))
        H_idx = self.H_idx[:-self.n_remain_H]
        assert len(T_idx) == len(self) * self.n_T
        batch = []
        i = 0
        H_start, H_end, T_start, T_end = 0, 0, 0, 0
        while i < len(self):
            H_end += self.batch_size - self.n_T
            T_end += self.n_T
            batch.extend(H_idx[H_start:H_end])
            batch.extend(T_idx[T_start:T_end])
            assert len(batch) == self.batch_size
            random.shuffle(batch)
            yield batch
            batch = []
            i += 1
            H_start = H_end
            T_start = T_end

    def __len__(self):
        return self.H_epoch

    def set_epoch(self, epoch):
        self.epoch = epoch


def collate_pre(data):
    labels = torch.tensor([i['label'] for i in data])
    prod_tensors = [i['prod_inputs'][0] for i in data]
    prod_tensors = update_tensors(prod_tensors)
    prod_scopes = [i['prod_inputs'][1] for i in data]
    atom_scopes = [i[0] for i in prod_scopes]
    atom_scopes = update_scope(atom_scopes)

    return prod_tensors, torch.tensor(atom_scopes), labels


def collate_emb(data):
    embed = torch.cat([i[0].unsqueeze(0) for i in data], 0)
    labels = torch.tensor([i[1].item() for i in data])
    return embed, labels


class Upsample:
    def __init__(self, epoch, args):
        self.epoch = epoch
        self.args = args

    def upsample(self, embed, labels, portion=1.0, im_class_num=3):
        random.seed(self.epoch)
        c_largest = labels.max().item()
        idx = torch.tensor(list(range(len(labels))))

        for i in trange(im_class_num):
            chosen = idx[(labels == (c_largest - i))[idx]]
            num = int(chosen.shape[0] * portion)
            if portion == 0:
                avg_number = int(idx.shape[0] / (c_largest + 1))
                c_portion = int(avg_number / chosen.shape[0])
                num = chosen.shape[0]
            else:
                c_portion = 1

            if num == 1:
                pass
                '''
                new_embed = embed[chosen, :]
                new_labels = labels.new(torch.Size((chosen.shape[0], 1))).reshape(-1).fill_(c_largest - i)
                idx_new = np.arange(embed.shape[0], embed.shape[0] + chosen.shape[0])
                idx_append = idx.new(idx_new)
                embed = torch.cat((embed, new_embed), 0)
                labels = torch.cat((labels, new_labels), 0)
                idx = torch.cat((idx, idx_append), 0)
                '''
            else:
                for j in range(c_portion):
                    chosen = chosen[:num]

                    chosen_embed = embed[chosen, :]
                    distance = squareform(pdist(chosen_embed.cpu().detach()))
                    np.fill_diagonal(distance, distance.max() + 100)

                    idx_neighbor = distance.argmin(axis=-1)  # Equation 3

                    interp_place = random.random()
                    new_embed = embed[chosen, :] + (
                            chosen_embed[idx_neighbor, :] - embed[chosen, :]) * interp_place  # Equation 4

                    new_labels = labels.new(torch.Size((chosen.shape[0], 1))).reshape(-1).fill_(c_largest - i)
                    idx_new = np.arange(embed.shape[0], embed.shape[0] + chosen.shape[0])
                    idx_append = idx.new(idx_new)

                    embed = torch.cat((embed, new_embed), 0)
                    labels = torch.cat((labels, new_labels), 0)
                    idx = torch.cat((idx, idx_append), 0)

        return embed, labels

    def epoch_upsample(self, embed, labels, portion=1.0):
        random.seed(self.epoch)
        im_class_labels = torch.tensor(list(range(self.args.sep_point, self.args.nclass)))
        idx = torch.tensor(list(range(len(labels))))
        uni_list = list(np.unique(np.array(labels.cpu())))
        for i in uni_list:
            if i in im_class_labels:
                chosen = idx[(labels == i)[idx]]
                num = int(chosen.shape[0] * portion)
                if portion == 0:
                    avg_number = int(idx.shape[0] / len(list(set(labels))))
                    c_portion = int(avg_number / chosen.shape[0])
                    num = chosen.shape[0]
                else:
                    c_portion = 1

                if num == 1:
                    pass
                    '''
                    interp_place = random.random()
                    new_embed = embed[chosen, :] * interp_place
                    new_labels = labels.new(torch.Size((chosen.shape[0], 1))).reshape(-1).fill_(i)
                    idx_new = np.arange(embed.shape[0], embed.shape[0] + chosen.shape[0])
                    idx_append = idx.new(idx_new)
                    embed = torch.cat((embed, new_embed), 0)
                    labels = torch.cat((labels, new_labels), 0)
                    idx = torch.cat((idx, idx_append), 0)
                    '''
                else:
                    for j in range(c_portion):
                        chosen = chosen[:num]

                        chosen_embed = embed[chosen, :]
                        distance = squareform(pdist(chosen_embed.cpu().detach()))
                        np.fill_diagonal(distance, distance.max() + 100)

                        idx_neighbor = distance.argmin(axis=-1)  # Equation 3

                        interp_place = random.random()
                        new_embed = embed[chosen, :] + (
                                chosen_embed[idx_neighbor, :] - embed[chosen, :]) * interp_place  # Equation 4

                        new_labels = labels.new(torch.Size((chosen.shape[0], 1))).reshape(-1).fill_(i)
                        idx_new = np.arange(embed.shape[0], embed.shape[0] + chosen.shape[0])
                        idx_append = idx.new(idx_new)

                        embed = torch.cat((embed, new_embed), 0)
                        labels = torch.cat((labels, new_labels), 0)
                        idx = torch.cat((idx, idx_append), 0)

        return embed, labels
