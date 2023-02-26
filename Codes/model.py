from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
import torch.optim.lr_scheduler as lr_scheduler

from .layers import GraphFeatEncoder, build_mlp
from .utils import get_input_from_data, get_acc, performance_measure
from .data import Upsample


class SideModel(nn.Module):
    def __init__(self, args, device):
        super(SideModel, self).__init__()
        self.args = args
        self.device = device
        self.encoder = GraphFeatEncoder(node_fdim=args.n_atom_feat,
                                        edge_fdim=args.n_bond_feat,
                                        hsize=args.nhid,
                                        depth=args.depth,
                                        dropout_p=args.dropout_mpn)
        self.classifier = build_mlp(in_dim=args.mlp_size,
                                    out_dim=args.nclass,
                                    h_dim=args.mlp_size,
                                    dropout_p=args.dropout_mlp)


class Trainer:
    def __init__(self, model, args=None, device='cpu'):
        self.model = model
        self.device = device
        self.args = args
        self.optimizer = Adam([{'params': model.parameters()}], lr=args.lr)
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max',
                                                        patience=args.patience,
                                                        factor=args.anneal_rate,
                                                        threshold=args.metric_thresh,
                                                        threshold_mode='abs')
        self.optimizer_cls = Adam(self.model.classifier.parameters(), args.lr_cls)

    def load_state_dict(self, encoder_path, classifier_path):
        self.model.encoder.load_state_dict(torch.load(encoder_path, map_location=self.device))
        self.model.classifier.load_state_dict(torch.load(classifier_path, map_location=self.device))

    def predict(self, data):
        embed, labels = data
        labels = labels.to(self.device)
        output = self.model.classifier(embed)
        ce_loss = -F.cross_entropy(output, labels)
        # loss_nodeclassfication = -ce_loss
        pt = torch.exp(-F.cross_entropy(output, labels))
        loss_nodeclassfication = -((1 - pt) ** self.args.gamma) * self.args.alpha * ce_loss
        acc = get_acc(output, labels)
        return loss_nodeclassfication, output, labels, acc

    def test_step(self, test_dataloader, epoch=None, data_name='test'):
        self.model.eval()
        total = 0.
        correct = 0.
        output_list = []
        labels_list = []
        with torch.no_grad():
            test_progress_bar = tqdm(test_dataloader)
            for i, data in enumerate(test_progress_bar):
                prod_tensors, atom_scopes, labels = get_input_from_data(data, obj='pre', device=self.device)
                hatom, hmol = self.model.encoder(prod_tensors, atom_scopes)
                _, output, labels, acc = self.predict((hmol, labels))
                output_list.append(output)
                labels_list.append(labels)

                cur_batch_size = len(labels)
                total += cur_batch_size
                correct += cur_batch_size * acc
                total_acc = '%.5f' % (correct / total)
                test_progress_bar.set_postfix(
                    acc=total_acc,
                    epoch=epoch)

            outputs = torch.cat(output_list, dim=0)
            labels = torch.cat(labels_list, dim=0)
        if 'infer' in str(epoch):
            return F.log_softmax(outputs, dim=-1), labels
        else:
            acc, macro_F, gmean, bacc = performance_measure(outputs, labels, pre=data_name)
            print('%s Epoch:%s acc:%.5f' % (data_name, epoch, acc), flush=True)
            return acc

    def test_emb(self, dataloader):
        self.model.eval()
        labels_list = []
        embed_list = []
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                prod_tensors, atom_scopes, labels = get_input_from_data(data, obj='pre', device=self.device)
                hatom, hmol = self.model.encoder(prod_tensors, atom_scopes)
                labels_list.append(labels)
                embed_list.append(hmol)
            labels = torch.cat(labels_list, dim=0)
            embeds = torch.cat(embed_list, dim=0)
        return embeds, labels

    def train_step(self, train_dataloader, epoch):
        self.model.train()
        total = 0.
        correct = 0.
        epoch_loss = 0.
        output_list = []
        labels_list = []
        progress_bar = tqdm(train_dataloader)
        for i, data in enumerate(progress_bar):
            prod_tensors, atom_scopes, labels = get_input_from_data(data, obj='pre', device=self.device)
            hatom, hmol = self.model.encoder(prod_tensors, atom_scopes)
            loss, output, labels, acc = self.predict((hmol, labels))
            loss.backward()
            epoch_loss += loss.item()

            cur_batch_size = len(labels)
            total += cur_batch_size

            self.optimizer.step()
            self.optimizer.zero_grad()

            correct += cur_batch_size * acc

            output_list.append(output.detach())
            labels_list.append(labels)

            progress_bar.set_postfix(
                loss='%.5f' % (epoch_loss / total),
                acc='%.5f' % (correct / total),
                epoch=epoch)
        outputs = torch.cat(output_list, dim=0)
        labels = torch.cat(labels_list, dim=0)

        acc_train, macro_F, gmean, bacc = performance_measure(outputs, labels, pre='train')
        print('Epoch:%s train_acc:%.5f' % (epoch, acc_train.item()), flush=True)

    def train_epochsample(self, train_dataloader, epoch):
        self.model.train()
        upsample = Upsample(epoch, self.args)
        output_list = []
        labels_list = []
        epoch_loss = 0.
        total = 0.
        correct = 0.
        progress_bar = tqdm(train_dataloader)
        for i, data in enumerate(progress_bar):
            prod_tensors, atom_scopes, labels = get_input_from_data(data, obj='pre', device=self.device)
            hatom, hmol = self.model.encoder(prod_tensors, atom_scopes)
            embed, labels = upsample.epoch_upsample(hmol, labels, portion=self.args.up_scale)

            loss, output, labels, acc = self.predict((embed, labels))
            loss.backward()
            epoch_loss += loss.item()

            cur_batch_size = len(labels)
            total += cur_batch_size

            self.optimizer.step()
            self.optimizer.zero_grad()

            correct += cur_batch_size * acc

            output_list.append(output.detach())
            labels_list.append(labels)

            progress_bar.set_postfix(
                loss='%.5f' % (epoch_loss / total),
                acc='%.5f' % (correct / total),
                epoch=epoch)
        outputs = torch.cat(output_list, dim=0)
        labels = torch.cat(labels_list, dim=0)
        acc, macro_F, gmean, bacc = performance_measure(outputs, labels)
        print('Train Epoch:%s acc:%.5f' % (epoch, acc), flush=True)
        return acc

    def train_oversample(self, train_og_dataloader, epoch):
        self.model.encoder.eval()
        self.model.classifier.train()
        total = 0.
        correct = 0.
        epoch_loss = 0.
        output_list = []
        labels_list = []
        progress_bar = tqdm(train_og_dataloader)
        for i, data in enumerate(progress_bar):
            loss, output, labels, acc = self.predict(data)
            loss.backward()
            epoch_loss += loss.item()

            cur_batch_size = len(labels)
            total += cur_batch_size

            self.optimizer_cls.step()
            self.optimizer_cls.zero_grad()

            correct += cur_batch_size * acc

            output_list.append(output.detach())
            labels_list.append(labels)

            progress_bar.set_postfix(
                loss='%.5f' % (epoch_loss / total),
                acc='%.5f' % (correct / total),
                epoch=epoch)
        outputs = torch.cat(output_list, dim=0)
        labels = torch.cat(labels_list, dim=0)

        acc_train, _, _, _ = performance_measure(outputs, labels, pre='train')
        print('Epoch:%s train_acc:%.5f' % (epoch, acc_train.item()), flush=True)
