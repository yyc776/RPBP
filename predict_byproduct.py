import os
import argparse

from copy import deepcopy

import pandas as pd

from Codes.utils import snapshot, str2bool, get_label_dict, check_model, compute_topn_acc, compute_class_topn_acc
from Codes.model import ByproductModel, Trainer
from Codes.data import RetroLogitsDatasets, HTBatchSampler, EmbDatasets, Upsample
from Codes.data import collate_pre, collate_emb
from Codes.embedder import Embedder
from Codes.mol_info import Label_Vocab

import torch
from torch.utils.data import DataLoader


def test_acc(test_dataloader, best_eval_model, args):
    trainer = Trainer(best_eval_model, args, device=args.device)
    trainer.test_step(test_dataloader, epoch='best_model')


def run_byproduct_model(args):
    self = Embedder(args)
    args.nclass = self.nclass
    args.sep_point = self.sep_point

    model = ByproductModel(args, device=self.args.device).to(self.args.device)
    trainer = Trainer(model, args, device=self.args.device)
    data_root = os.path.join(args.task_dataset, r'dataset/stage_one')

    print(f"Converting model to device: {self.args.device}")
    print("Param Count: ", sum([x.nelement() for x in model.parameters()]) / 10 ** 6, "M", flush=True)

    test_data = RetroLogitsDatasets(root=data_root, data_split='test', use_rxn_class=args.with_class)
    test_dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=0,
                                 collate_fn=collate_pre)

    eval_data = RetroLogitsDatasets(root=data_root, data_split='eval', use_rxn_class=args.with_class)
    eval_dataloader = DataLoader(eval_data, batch_size=args.batch_size, shuffle=False, num_workers=0,
                                 collate_fn=collate_pre)
    best_eval = 0.
    best_eval_epoch = 0
    best_eval_model = deepcopy(trainer.model)

    pre_model_state, model_path_list = check_model(os.path.join(args.save_root, 'pre_model'), obj='pre')
    if pre_model_state and args.load_pre:
        print('Load Pretrain Encoder !!!!', flush=True)
        trainer.load_state_dict(model_path_list[0], model_path_list[1])
        best_eval_model = deepcopy(trainer.model)
    else:
        print('Train Model !!!', flush=True)
        if args.epoch_sample:
            train_data = RetroLogitsDatasets(root=data_root, data_split='train',
                                             use_rxn_class=args.with_class)
            H_idx = self.idx_train_set_class['H']
            T_idx = self.idx_train_set_class['T']
            H_T_sample = HTBatchSampler(dataset=train_data, batch_size=args.batch_size, H_idx=H_idx, T_idx=T_idx,
                                        T_split_number=args.T_split_number)
            train_dataloader = DataLoader(train_data, batch_sampler=H_T_sample, num_workers=0, collate_fn=collate_pre)
            # train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_pre)

            for epoch in range(1, args.epochs + 1):
                H_T_sample.set_epoch(epoch)
                trainer.train_epochsample(train_dataloader, epoch)
                acc_test = trainer.test_step(test_dataloader, epoch=epoch, data_name='test')
                acc_eval = trainer.test_step(eval_dataloader, epoch=epoch, data_name='eval')
                trainer.scheduler.step(acc_eval)

                if acc_eval > best_eval:
                    best_eval = acc_eval
                    best_eval_epoch = epoch
                    best_eval_model = deepcopy(trainer.model)

            print('OverSample train best eval acc is %.5f' % best_eval)
            snapshot(best_eval_model.encoder, best_eval_epoch, acc=best_eval,
                     save_path=os.path.join(args.save_root, 'pre_model'))
            snapshot(best_eval_model.classifier, best_eval_epoch, acc=best_eval,
                     save_path=os.path.join(args.save_root, 'pre_model'))
            test_acc(test_dataloader, best_eval_model, args)
        else:
            train_data = RetroLogitsDatasets(root=data_root, data_split='train',
                                             use_rxn_class=args.with_class)
            train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0,
                                          collate_fn=collate_pre)

            for epoch in range(1, args.epochs + 1):
                trainer.train_step(train_dataloader, epoch)
                acc_test = trainer.test_step(test_dataloader, epoch=epoch, data_name='test')
                acc_eval = trainer.test_step(eval_dataloader, epoch=epoch, data_name='eval')
                trainer.scheduler.step(acc_eval)

                if acc_eval > best_eval:
                    best_eval = acc_eval
                    best_eval_epoch = epoch
                    best_eval_model = deepcopy(trainer.model)

            print('OverSample Pre-train best eval acc is %.5f' % best_eval)
            snapshot(best_eval_model.encoder, best_eval_epoch, acc=best_eval,
                     save_path=os.path.join(args.save_root, 'pre_model'))
            snapshot(best_eval_model.classifier, best_eval_epoch, acc=best_eval,
                     save_path=os.path.join(args.save_root, 'pre_model'))
            test_acc(test_dataloader, best_eval_model, args)

    if args.finetune:
        train_data = RetroLogitsDatasets(root=data_root, data_split='train',
                                         use_rxn_class=args.with_class)
        train_dataloader = DataLoader(train_data, batch_size=args.batch_size * 16, shuffle=True, num_workers=0,
                                      collate_fn=collate_pre)
        train_embed, train_labels = trainer.test_emb(train_dataloader)
        best_eval = trainer.test_step(eval_dataloader, epoch=0, data_name='eval')
        print('Start Finetune !!!!@', flush=True)
        for epoch in range(1, 1 + args.finetune_epochs):
            upsample = Upsample(epoch, args)
            up_train_embed, up_train_labels = upsample.upsample(train_embed, train_labels, portion=args.up_scale,
                                                                im_class_num=args.im_class_num)

            train_og_data = EmbDatasets(up_train_embed, up_train_labels)
            train_og_dataloader = DataLoader(train_og_data, batch_size=args.batch_size, shuffle=True,
                                             collate_fn=collate_emb)

            trainer.train_oversample(train_og_dataloader, epoch=epoch)
            acc_test = trainer.test_step(test_dataloader, epoch=epoch, data_name='test')
            acc_eval = trainer.test_step(eval_dataloader, epoch=epoch, data_name='eval')
            trainer.scheduler.step(acc_eval)

            if acc_eval > best_eval:
                best_eval = acc_eval
                best_eval_epoch = epoch
                best_eval_model = deepcopy(trainer.model)

        print('Finetune best eval acc is %.5f' % best_eval)
        snapshot(best_eval_model.encoder, best_eval_epoch, acc=best_eval,
                 save_path=os.path.join(args.save_root, 'pre_model'))
        snapshot(best_eval_model.classifier, best_eval_epoch, acc=best_eval,
                 save_path=os.path.join(args.save_root, 'pre_model'))
        test_acc(test_dataloader, best_eval_model, args)

    if args.infer is True:
        for data_name in ['test', 'eval']:
            data = RetroLogitsDatasets(root=data_root, data_split=data_name,
                                       use_rxn_class=args.with_class)
            dataloader = DataLoader(data, batch_size=args.batch_size * 2, shuffle=False, num_workers=0,
                                    collate_fn=collate_pre)
            trainer = Trainer(best_eval_model, args, device=args.device)
            log_outputs, labels = trainer.test_step(dataloader, epoch='infer_%s' % data_name, data_name=data_name)
            label_dict = get_label_dict(path=r'USPTO-50K/dataset/stage_one/byproduct_smiles_to_label.dict')
            Topn_Smiles = []
            Topn_Scores = []
            Topn_acc = [0 for _ in range(args.topn)]
            for idx, output in enumerate(log_outputs):
                value, index = torch.topk(output, k=args.topn)
                Topn_Scores.extend(value.cpu().numpy().tolist())
                Topn_Smiles.extend([label_dict.get_elem(i) for i in index.cpu().numpy().tolist()])
                Topn_acc = compute_topn_acc(index, labels[idx], Topn_acc)
            print('Stage One %s Top-%s Accuracy' % (data_name, args.topn))
            for i in range(args.topn):
                print('Top-{} accuracy: {:.5f}% '.format(i + 1, Topn_acc[i] / len(labels) * 100))
            save_name = 'with_class' if args.with_class else 'without_class'
            os.makedirs(os.path.join(args.task_dataset, r'results/stage_one', save_name), exist_ok=True)
            with open(os.path.join(args.task_dataset, r'results/stage_one', save_name,
                                   '%s_top%s_smiles.txt' % (data_name, args.topn)), 'w+') as f:
                for smiles in Topn_Smiles:
                    f.write('{}\n'.format(smiles))

            with open(os.path.join(args.task_dataset, r'results/stage_one', save_name,
                                   '%s_top%s_scores.txt' % (data_name, args.topn)), 'w+') as f:
                for scores in Topn_Scores:
                    f.write('{}\n'.format(scores))

        print('========================END===============================')

    if args.infer_each_class is True:
        for data_name in ['test', 'eval']:
            data = RetroLogitsDatasets(root=data_root, data_split=data_name,
                                       use_rxn_class=args.with_class)
            dataloader = DataLoader(data, batch_size=args.batch_size * 2, shuffle=False, num_workers=0,
                                    collate_fn=collate_pre)
            trainer = Trainer(best_eval_model, args, device=args.device)
            log_outputs, labels = trainer.test_step(dataloader, epoch='infer_%s' % data_name, data_name=data_name)

            csv = pd.read_csv(os.path.join(data_root, data_name, 'canonicalized_%s.csv' % data_name))
            rxn_class_list = csv['class'].tolist()
            rxn_class_list = [rxn_class - 1 for rxn_class in rxn_class_list]
            assert len(rxn_class_list) == len(labels)

            class_Topn_acc = [[0 for _ in range(args.topn)] for _ in range(10)]
            class_label = [0 for _ in range(10)]
            for idx, output in enumerate(log_outputs):
                value, index = torch.topk(output, k=args.topn)
                rxn_class = rxn_class_list[idx]
                class_label[rxn_class] += 1
                class_Topn_acc = compute_class_topn_acc(index, labels[idx], class_Topn_acc, rxn_class)

            print('Stage One %s Each Class Top-%s Accuracy' % (data_name, args.topn))
            for r in range(10):
                print('RXN_class %s acc:'% (r+1))
                for i in range(args.topn):
                    print('Top-{} accuracy: {:.5f}% '.format(i + 1, class_Topn_acc[r][i] / class_label[r] * 100))

        print('========================END===============================')


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
    parser.add_argument("--lr", default=0.003, type=float)
    parser.add_argument('--lr_cls', type=float, default=0.00003)

    parser.add_argument("--T_split_number", default=0.1, type=float)
    parser.add_argument("--num_workers", default=0)
    parser.add_argument("--seed", default=916)
    parser.add_argument("--sep_degree", default=150)
    parser.add_argument('--im_ratio', type=float, default=0.01)
    parser.add_argument('--sep_class', type=str, default='pareto_19')
    parser.add_argument("--im_class_num", default=150)
    parser.add_argument("--epochs", default=600, type=int)
    parser.add_argument("--finetune_epochs", default=10, type=int)
    parser.add_argument('--up_scale', type=float, default=1)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--gpu', type=str, default=0)
    parser.add_argument("--epoch_sample", default=False, type=str2bool)
    parser.add_argument("--finetune", default=False, type=str2bool)
    parser.add_argument("--load_pre", default=False, type=str2bool)
    parser.add_argument("--infer", default=True, type=str2bool)
    parser.add_argument("--infer_each_class", default=False, type=str2bool)
    parser.add_argument("--with_class", action="store_true")
    parser.add_argument('--topn', type=float, default=10)


    parser.add_argument("--task_dataset", type=str, default="USPTO-50K")
    parser.add_argument("--save_root", default=r'save_model/stage_one/without_class')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    if args.with_class:
        args.n_atom_feat += 10
        args.save_root = r'save_model/stage_one/with_class'
    args.save_root = os.path.join(args.task_dataset, args.save_root)
    print(args)
    run_byproduct_model(args)
