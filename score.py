from rdkit import Chem
import os
import argparse
from tqdm import tqdm
import multiprocessing
import pandas as pd
from rdkit import RDLogger
import re
import numpy as np
from Codes.utils import is_number, str2bool

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


def smi_tokenizer(smi):
    pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    assert smi == ''.join(tokens)
    return ' '.join(tokens)


def canonicalize_smiles_clear_map(smiles, return_max_frag=True):
    mol = Chem.MolFromSmiles(smiles, sanitize=True)
    if mol is not None:
        [atom.ClearProp('molAtomMapNumber') for atom in mol.GetAtoms() if atom.HasProp('molAtomMapNumber')]
        try:
            smi = Chem.MolToSmiles(mol, isomericSmiles=True)
        except:
            if return_max_frag:
                return '', ''
            else:
                return ''
        if return_max_frag:
            sub_smi = smi.split(".")
            sub_mol = [Chem.MolFromSmiles(smiles, sanitize=True) for smiles in sub_smi]
            sub_mol_size = [(sub_smi[i], len(m.GetAtoms())) for i, m in enumerate(sub_mol) if m is not None]
            if len(sub_mol_size) > 0:
                return smi, canonicalize_smiles_clear_map(sorted(sub_mol_size, key=lambda x: x[1], reverse=True)[0][0],
                                                          return_max_frag=False)
            else:
                return smi, ''
        else:
            return smi
    else:
        if return_max_frag:
            return '', ''
        else:
            return ''


def compute_rank_stage_two(prediction, raw=False, alpha=1.0):
    valid_score = [[k for k in range(len(prediction[j]))] for j in range(len(prediction))]
    invalid_rates = [0 for k in range(len(prediction[0]))]
    rank = {}
    max_frag_rank = {}
    highest = {}
    if raw:
        # no test augmentation
        assert len(prediction) == 1
        for j in range(len(prediction)):
            for k in range(len(prediction[j])):
                if prediction[j][k][0] == "":
                    invalid_rates[k] += 1
            # error detection
            prediction[j] = [i for i in prediction[j] if i[0] != ""]
            for k, data in enumerate(prediction[j]):
                rank[data] = 1 / (alpha * k + 1)
    else:

        for j in range(len(prediction)):
            for k in range(len(prediction[j])):
                # predictions[i][j][k] = canonicalize_smiles_clear_map(predictions[i][j][k])
                if prediction[j][k][0] == "":
                    valid_score[j][k] = args.beam_size + 1
                    invalid_rates[k] += 1
            # error detection and deduplication
            de_error = [i[0] for i in sorted(list(zip(prediction[j], valid_score[j])), key=lambda x: x[1]) if
                        i[0][0] != ""]
            prediction[j] = list(set(de_error))
            prediction[j].sort(key=de_error.index)
            for k, data in enumerate(prediction[j]):
                if data in rank:
                    rank[data] += 1 / (alpha * k + 1)
                else:
                    rank[data] = 1 / (alpha * k + 1)
                if data in highest:
                    highest[data] = min(k, highest[data])
                else:
                    highest[data] = k
        for key in rank.keys():
            rank[key] += highest[key] * -1e8
    return rank, invalid_rates


def compute_rank(prediction, score, add=True, alpha=1):
    rank = {}
    # error detection and deduplication
    de_error = [(i[0], i[1]) for i in sorted(list(zip(prediction, score)), key=lambda x: x[1], reverse=True) if
                i[0][0] != ""]

    if add:
        for k, (pred, score) in enumerate(de_error):
            if pred in rank:
                rank[pred] += 1 / (k * alpha + 1)
            else:
                rank[pred] = 1 / (k * alpha + 1)
    else:
        pred, score = [], []
        for i in de_error:
            if i[0] not in pred:
                pred.append(i[0])
                score.append(i[1])
        for pred, score in zip(pred, score):
            rank[pred] = score
    return rank


def get_args():
    parser = argparse.ArgumentParser(
        description='score.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--score_stage_two', action="store_true", default=False)
    parser.add_argument('--infer_each_class', action="store_true", default=False)
    parser.add_argument('--beam_size', type=int, default=10, help='Beam size')
    parser.add_argument('--n_best', type=int, default=50, help='n best')
    parser.add_argument('--stage_one_topn', type=int, default=10)
    parser.add_argument('--predictions', type=str, required=True,
                        help="Path to file containing the predictions")
    parser.add_argument('--targets', type=str, default="", help="Path to file containing targets")
    parser.add_argument('--sources', type=str, default="", help="Path to file containing sources")
    parser.add_argument('--augmentation', type=int, default=20)
    parser.add_argument('--score_alpha', type=float, default=1.0)
    parser.add_argument('--score_beta', type=float, default=1.0)
    parser.add_argument('--length', type=int, default=-1)
    parser.add_argument('--process', type=int, default=multiprocessing.cpu_count())
    parser.add_argument('--synthon', action="store_true", default=False)
    parser.add_argument('--detailed', action="store_true", default=False)
    parser.add_argument('--raw', action="store_true", default=False)
    parser.add_argument('--add', type=str2bool, default=True)
    parser.add_argument('--save_file', type=str, default="")
    parser.add_argument('--save_accurate_indices', type=str, default="")

    if not parser.parse_known_args()[0].score_stage_two or parser.parse_known_args()[0].infer_each_class:
        parser.add_argument('--stage_one_scores', type=str, required=True,
                            help="Path to file containing the stage one scores")

    args = parser.parse_args()
    return args


def main(args):

    if args.score_stage_two:
        print('Reading predictions from file ...')
        with open(args.predictions, 'r') as f:
            lines = [''.join(line.strip().split(' ')) for line in f.readlines()]
        print(len(lines))
        data_size = len(lines) // (
                args.augmentation * args.beam_size) if args.length == -1 else args.length
        lines = lines[:data_size * (args.augmentation * args.beam_size)]
        Smiles, Scores = [], []
        for line in lines:
            smiles_score = line.split('\t')
            if len(smiles_score) == 1 and is_number(smiles_score[0]):
                Smiles.append('')
                Scores.append(eval(smiles_score[0]))
            elif len(smiles_score) == 1 and not is_number(smiles_score[0]):
                Smiles.append(smiles_score[0])
                Scores.append('')
            else:
                assert len(smiles_score) == 2
                Smiles.append(smiles_score[0])
                Scores.append(eval(smiles_score[1]))
        print("Canonicalizing predictions using Process Number ", args.process)

        with multiprocessing.Pool(processes=args.process) as pool:
            raw_predictions = list(tqdm(pool.imap(canonicalize_smiles_clear_map, Smiles), total=len(Smiles)))
        pool.close()
        pool.join()

        predictions = [[[] for j in range(args.augmentation)] for i in range(data_size)]  # data_len x augmentation x beam_size
        for i, line in enumerate(raw_predictions):
            predictions[i // (args.beam_size * args.augmentation)][i % (args.beam_size * args.augmentation) // args.beam_size].append(line)

        print("data size ", data_size)
        print('Reading targets from file ...')
        with open(args.targets, 'r') as f:
            lines = f.readlines()
            # lines = [''.join(line.strip().split(' ')) for line in f.readlines()]
        print("Origin File Length", len(lines))
        targets = [''.join(lines[i].strip().split(' ')) for i in
                   tqdm(range(0, data_size * args.augmentation, args.augmentation))]
        with multiprocessing.Pool(processes=args.process) as pool:
            targets = list(tqdm(pool.imap(canonicalize_smiles_clear_map, targets), total=len(targets)))
        pool.close()
        pool.join()
        ground_truth = targets
        print("Origin Target Lentgh, ", len(ground_truth))
        print("Cutted Length, ", data_size)
        accuracy = [0 for j in range(args.n_best)]
        max_frag_accuracy = [0 for j in range(args.n_best)]
        invalid_rates = [0 for j in range(args.beam_size)]
        accurate_indices = [[] for j in range(args.n_best)]
        ranked_results = []
        sorted_invalid_rates = [0 for j in range(args.beam_size)]
        for i in tqdm(range(len(predictions))):
            accurate_flag = False
            rank, invalid_rate = compute_rank_stage_two(predictions[i], raw=args.raw, alpha=args.score_alpha)
            for j in range(args.beam_size):
                invalid_rates[j] += invalid_rate[j]
            rank = list(zip(rank.keys(), rank.values()))
            rank.sort(key=lambda x: x[1], reverse=True)
            rank = rank[:args.n_best]
            ranked_results.append([item[0][0] for item in rank])
            for j, item in enumerate(rank):
                if item[0][0] == ground_truth[i][0]:
                    if not accurate_flag:
                        accurate_flag = True
                        accurate_indices[j].append(i)
                        for k in range(j, args.n_best):
                            accuracy[k] += 1
            for j, item in enumerate(rank):
                if item[0][1] == ground_truth[i][1]:
                    for k in range(j, args.n_best):
                        max_frag_accuracy[k] += 1
                    break
            for j in range(len(rank), args.beam_size):
                sorted_invalid_rates[j] += 1

        for i in range(args.n_best):
            # if i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 19, 49]:
            if i in range(10):
                print("Top-{} Acc:{:.3f}%, MaxFrag {:.3f}%,".format(i + 1, accuracy[i] / data_size * 100,
                                                                    max_frag_accuracy[i] / data_size * 100),
                      " Invalid SMILES:{:.3f}% Sorted Invalid SMILES:{:.3f}%".format(
                          invalid_rates[i] / data_size / args.augmentation * 100,
                          sorted_invalid_rates[i] / data_size / args.augmentation * 100))

    else:
        print('Reading stage one scores')
        with open(args.stage_one_scores, 'r') as f:
            first_scores = f.readlines()
        first_scores = [np.repeat(eval(score), args.augmentation * args.beam_size) for score in first_scores]
        print('Reading predictions from file ...')
        with open(args.predictions, 'r') as f:
            lines = [''.join(line.strip().split(' ')) for line in f.readlines()]
        print(len(lines))
        data_size = len(lines) // (
                    args.augmentation * args.beam_size * args.stage_one_topn) if args.length == -1 else args.length
        lines = lines[:data_size * (args.augmentation * args.beam_size * args.stage_one_topn)]
        first_scores = first_scores[:data_size * args.stage_one_topn]
        first_scores = [j for i in first_scores for j in i]
        Smiles, Scores = [], []
        for line in lines:
            smiles_score = line.split('\t')
            if len(smiles_score) == 1:
                Smiles.append('')
                Scores.append(eval(smiles_score[0]))
            else:
                assert len(smiles_score) == 2
                Smiles.append(smiles_score[0])
                Scores.append(eval(smiles_score[1]))
        assert len(Scores) == len(first_scores) == len(Smiles)
        Scores = [first_score * args.score_beta + second_score for first_score, second_score in zip(first_scores, Scores)]
        print("Canonicalizing predictions using Process Number ", args.process)

        with multiprocessing.Pool(processes=args.process) as pool:
            raw_predictions = list(tqdm(pool.imap(canonicalize_smiles_clear_map, Smiles), total=len(Smiles)))
        pool.close()
        pool.join()

        predictions = [[] for i in range(data_size)]  # data_len x augmentation x beam_size
        scores = [[] for i in range(data_size)]
        for i, (line, score) in enumerate(zip(raw_predictions, Scores)):
            predictions[i // (args.beam_size * args.augmentation * args.stage_one_topn)].append(line)
            scores[i // (args.beam_size * args.augmentation * args.stage_one_topn)].append(score)

        print("data size ", data_size)
        if args.targets != "":
            print('Reading targets from file ...')
            with open(args.targets, 'r') as f:
                lines = f.readlines()
                # lines = [''.join(line.strip().split(' ')) for line in f.readlines()]
            print("Origin File Length", len(lines))
            targets = [''.join(lines[i].strip().split(' ')) for i in
                       tqdm(range(0, data_size * args.augmentation * args.stage_one_topn, args.augmentation * args.stage_one_topn))]
            pool = multiprocessing.Pool(processes=args.process)
            targets = pool.map(func=canonicalize_smiles_clear_map, iterable=targets)
            pool.close()
            pool.join()
            ground_truth = targets
            print("Origin Target Lentgh, ", len(ground_truth))
            print("Cutted Length, ", data_size)

        if args.infer_each_class:
                csv = pd.read_csv(r'dataset/stage_one/test/canonicalized_test.csv')
                rxn_class_list = csv['class'].tolist()
                rxn_class_list = [rxn_class - 1 for rxn_class in rxn_class_list]
                assert len(rxn_class_list) == len(predictions)

                class_Topn_acc = [[0 for _ in range(args.n_best)] for _ in range(10)]
                class_max_frag_accuracy = [[0 for _ in range(args.n_best)] for _ in range(10)]
                class_label = [0 for _ in range(10)]

                for i in tqdm(range(len(predictions))):
                    accurate_flag = False
                    rxn_class = rxn_class_list[i]
                    class_label[rxn_class] += 1
                    ranked_results = []
                    rank = compute_rank(predictions[i], scores[i], add=args.add, alpha=args.score_alpha)
                    rank = list(zip(rank.keys(), rank.values()))
                    rank.sort(key=lambda x: x[1], reverse=True)
                    rank = rank[:args.n_best]
                    ranked_results.append([item[0][0] for item in rank])

                    for j, item in enumerate(rank):
                        if item[0][0] == ground_truth[i][0]:
                            if not accurate_flag:
                                accurate_flag = True
                                for k in range(j, args.n_best):
                                    class_Topn_acc[rxn_class][k] += 1

                    for j, item in enumerate(rank):
                        if item[0][1] == ground_truth[i][1]:
                            for k in range(j, args.n_best):
                                class_max_frag_accuracy[rxn_class][k] += 1
                            break

                print('Test Each Class Top-%s Accuracy' % args.n_best)
                for r in range(10):
                    print('RXN_class %s acc:' % (r + 1))
                    for i in range(args.n_best):
                        if i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 19, 49]:
                            # if i in range(10):
                            print("Top-{} Acc:{:.3f}%, MaxFrag {:.3f}%,".format(i + 1, class_Topn_acc[r][i] / class_label[r] * 100,
                                                                                class_max_frag_accuracy[r][i] / class_label[r] * 100),
                                  " Invalid SMILES:{:.3f}% Sorted Invalid SMILES:{:.3f}%".format(0, 0))

        else:
            accuracy = [0 for j in range(args.n_best)]
            accurate_indices = [[] for j in range(args.n_best)]
            max_frag_accuracy = [0 for j in range(args.n_best)]
            ranked_results = []

            for i in tqdm(range(len(predictions))):
                accurate_flag = False

                rank = compute_rank(predictions[i], scores[i], add=args.add, alpha=args.score_alpha)
                rank = list(zip(rank.keys(), rank.values()))
                rank.sort(key=lambda x: x[1], reverse=True)
                rank = rank[:args.n_best]
                ranked_results.append([item[0][0] for item in rank])
                if args.targets is "":
                    continue

                for j, item in enumerate(rank):
                    if item[0][0] == ground_truth[i][0]:
                        if not accurate_flag:
                            accurate_flag = True
                            accurate_indices[j].append(i)
                            for k in range(j, args.n_best):
                                accuracy[k] += 1

                for j, item in enumerate(rank):
                    if item[0][1] == ground_truth[i][1]:
                        for k in range(j, args.n_best):
                            max_frag_accuracy[k] += 1
                        break

            if args.targets is "":
                with open('stage_two_prediction.txt', 'w') as f:
                    for idx, results in enumerate(ranked_results):
                        f.write('id:%s\n' % idx)
                        for result in results:
                            f.write('{}\n'.format(result))
                exit(0)

            for i in range(args.n_best):
                if i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 19, 49]:
                    # if i in range(10):
                    print("Top-{} Acc:{:.3f}%, MaxFrag {:.3f}%,".format(i + 1, accuracy[i] / data_size * 100,
                                                                        max_frag_accuracy[i] / data_size * 100))


            if args.save_accurate_indices != "":
                with open(args.save_accurate_indices, "w") as f:
                    total_accurate_indices = []
                    for indices in accurate_indices:
                        total_accurate_indices.extend(indices)
                    total_accurate_indices.sort()

                    # for index in total_accurate_indices:
                    for index in accurate_indices[0]:
                        f.write(str(index))
                        f.write("\n")

            if args.save_file != "":
                with open(args.save_file, "w") as f:
                    for res in ranked_results:
                        for smi in res:
                            f.write(smi)
                            f.write("\n")
                        for i in range(len(res), args.n_best):
                            f.write("")
                            f.write("\n")


if __name__ == "__main__":
    args = get_args()
    if args.infer_each_class is True:
        args.score_stage_two = False
        assert args.targets != ''
    print(args)
    main(args)
