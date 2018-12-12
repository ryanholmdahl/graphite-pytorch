import pickle
from models import MultiGCNModelFeedback, MultiGCNModelNodeDirect
import torch
import torch.optim as optim
import sys
import os
import random
from collections import defaultdict
import argparse
import gc

from optimizer import get_loss, get_accuracy, get_precision, get_recall, get_sample_metrics, is_valid, adj_stack_to_tup
from preprocess import preprocess_graph


# Data and preprocessing
# dataset = 'all_molecules.pkl'
# num_train_examples = 10000
# sample_train_randomly = True
# identity_features = True
# num_bond_types = 5
# num_atom_types = 4
# num_max_nodes = 9
#
# # Model hyperparameters
# model_type = 'multi_gcn_feedback'
# lr = 0.001
# epochs = 2000
# autoregressive = 0.5
# encode_features = False
# gcn_batch_norm = False
# gcn_hiddens = [32, 32]
# gcn_aggs = ['mean', 'mean']
# gcn_relus = [False, True]
# graphite_relu = True
# graphite_layers = 3
# z_dim = 32
# z_agg = 'mean'
# dropout = 0.
# use_norm = False
# use_pos_weight = True
# load_path = None
#
# # Evaluation
# num_gen_samples = 100
# num_gen_conditions = 10
# num_images_per_condition = 1
# final_num_gen_samples = 1000
# final_num_gen_conditions = 1000
# final_num_images_per_condition = 0
# evals_to_stop = 100
# eval_only = False


# def get_params():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--graphite_layers', type=int)
#     parser.add_argument('--lr', type=float)
#     parser.add_argument('--zdim', type=int)
#     parser.add_argument('--gcn_layers', type=int)
#     parser.add_argument('--zagg', type=str)
#     parser.add_argument('--pos_weight', action='store_true')
#     parser.add_argument('--autoregressive', type=float)
#     args = parser.parse_args()
#     return args.graphite_layers, args.lr, args.zdim, args.gcn_layers, args.zagg, args.pos_weight, args.autoregressive


def set_path(dataset, num_train_examples, sample_train_randomly,  identity_features, num_bond_types,
                   num_atom_types, num_max_nodes,
                   model_type, lr, epochs, autoregressive, encode_features,
                   gcn_batch_norm, gcn_hiddens, gcn_aggs, gcn_relus,
                   graphite_relu, graphite_layers, z_dim, z_agg, dropout, num_gen_samples, num_gen_conditions,
                   evals_to_stop, eval_only, load_path, use_pos_weight):
    path = ''
    load_path = load_path is not None
    for metric in [dataset, num_train_examples, sample_train_randomly,  identity_features, num_bond_types,
                   num_atom_types, num_max_nodes,
                   model_type, lr, epochs, autoregressive, encode_features,
                   gcn_batch_norm, gcn_hiddens, gcn_aggs, gcn_relus,
                   graphite_relu, graphite_layers, z_dim, z_agg, dropout, num_gen_samples, num_gen_conditions,
                   evals_to_stop, eval_only, load_path, use_pos_weight]:
        path += '-' + str(metric)
    image_dir = os.path.join('images', path)
    path += '.txt'
    sys.stdout = open(os.path.join('results', path), 'wt')
    return image_dir


def train(dataset, num_train_examples, sample_train_randomly,  identity_features, num_bond_types,
                   num_atom_types, num_max_nodes,
                   model_type, lr, epochs, autoregressive, encode_features,
                   gcn_batch_norm, gcn_hiddens, gcn_aggs, gcn_relus,
                   graphite_relu, graphite_layers, z_dim, z_agg, dropout, num_gen_samples, num_gen_conditions,
                   evals_to_stop, eval_only, load_path, use_pos_weight,
          preprocessed_data, xa_mappings, num_images_per_condition, final_num_gen_samples, final_num_gen_conditions,
            final_num_images_per_condition):
    image_dir = set_path(dataset, num_train_examples, sample_train_randomly,  identity_features, num_bond_types,
                   num_atom_types, num_max_nodes,
                   model_type, lr, epochs, autoregressive, encode_features,
                   gcn_batch_norm, gcn_hiddens, gcn_aggs, gcn_relus,
                   graphite_relu, graphite_layers, z_dim, z_agg, dropout, num_gen_samples, num_gen_conditions,
                   evals_to_stop, eval_only, load_path, use_pos_weight)

    if model_type == 'multi_gcn_feedback':
        model = MultiGCNModelFeedback(num_bond_types,
                                      num_atom_types + num_max_nodes,
                                      encode_features,
                                      gcn_batch_norm,
                                      gcn_hiddens,
                                      gcn_aggs,
                                      gcn_relus,
                                      z_dim,
                                      z_agg,
                                      graphite_relu,
                                      graphite_layers,
                                      dropout,
                                      autoregressive)
    else:
        raise ValueError()
    opt = optim.Adam(model.parameters(), lr=lr)
    costs = []
    kls = []
    accuracies = []
    precisions = []
    recalls = []
    validities = []

    val_scores = []
    up_val_scores = []

    train_val = random.sample(preprocessed_data, min(100, len(preprocessed_data)))
    print(
        'epoch\tstep\tbatch recon\tbatch kl\tbatch acc\tbatch prec\tbatch recall\tbatch valid\tval recon\tval kl\tval '
        'acc\tval prec\tval recall\tval valid\tgen valid\tgen accurate\tgen unique\tgen novel\tgen overall')

    if load_path is not None:
        model.load_state_dict(torch.load(load_path))

    if eval_only:
        print(image_dir)
        print(get_sample_metrics(model, xa_mappings, final_num_gen_conditions, final_num_gen_samples,
                             final_num_images_per_condition, os.path.join(image_dir, 'final')))
        return

    # best_gen_overall = 0.
    # evals_left = evals_to_stop

    for epoch in range(epochs):
        # if evals_left <= 0:
        #     break
        done = False
        random.shuffle(preprocessed_data)
        for i in range(len(preprocessed_data)):
            norms, pos_weight, no_pos_weight, adj_norms, features, adj_labels = preprocessed_data[i]
            if not use_pos_weight:
                pos_weight = no_pos_weight
            preds = model(features, adj_norms)
            cost, kl = get_loss(preds, adj_labels, *model.get_z(features, adj_norms), adj_norms.shape[1], norms,
                                pos_weight)
            loss = cost + kl
            costs.append(cost)
            # validities.append(is_valid(features, preds.max(dim=0)[1]))
            validities.append(0.)
            kls.append(kl)
            accuracies.append(get_accuracy(preds, adj_labels))
            precisions.append(get_precision(preds, adj_labels))
            recalls.append(get_recall(preds, adj_labels))
            opt.zero_grad()
            loss.backward()
            opt.step()
            if i % 1000 == 0:
                val_costs = []
                val_kls = []
                val_accs = []
                val_precs = []
                val_recs = []
                val_valids = []
                for norms, pos_weight, no_pos_weight, adj_norms, features, adj_labels in train_val:
                    if not use_pos_weight:
                        pos_weight = no_pos_weight
                    preds = model(features, adj_norms)
                    cost, kl = get_loss(preds, adj_labels, *model.get_z(features, adj_norms), adj_norms.shape[1], norms,
                                        pos_weight)
                    val_costs.append(cost)
                    val_accs.append(get_accuracy(preds, adj_labels))
                    val_kls.append(kl)
                    val_precs.append(get_precision(preds, adj_labels))
                    val_recs.append(get_recall(preds, adj_labels))
                    val_valids.append(1 if is_valid(features, preds.max(dim=0)[1]) else 0)
                gen_valid, gen_acc, gen_unique, gen_novel, \
                    up_gen_valid, up_gen_acc, up_gen_unique, up_gen_novel = get_sample_metrics(
                        model, xa_mappings, num_gen_conditions, num_gen_samples, num_images_per_condition,
                        os.path.join(image_dir, str(epoch)))
                gen_overall = gen_valid * gen_acc * gen_unique * gen_novel
                up_gen_overall = up_gen_valid * up_gen_acc * up_gen_unique * up_gen_novel
                if any([v < 0 for v in [gen_valid, gen_acc, gen_unique, gen_novel]]):
                    gen_overall = 0.
                if any([v < 0 for v in [up_gen_valid, up_gen_acc, up_gen_unique, up_gen_novel]]):
                    up_gen_overall = 0.
                # if gen_overall > best_gen_overall:
                #     best_gen_overall = gen_overall
                #     evals_left = evals_to_stop
                val_scores.append(gen_overall)
                up_val_scores.append(up_gen_overall)
                print('\t'.join('{:.3f}'.format(e) for e in
                                [epoch, i, float(sum(costs) / len(costs)), float(sum(kls) / len(kls)),
                                 float(sum(accuracies) / len(accuracies)), float(sum(precisions) / len(precisions)),
                                 float(sum(recalls) / len(recalls)), float(sum(validities) / len(validities)),
                                 float(sum(val_costs) / len(val_costs)), float(sum(val_kls) / len(val_kls)),
                                 float(sum(val_accs) / len(val_accs)), float(sum(val_precs) / len(val_precs)),
                                 float(sum(val_recs) / len(val_recs)), float(sum(val_valids) / len(val_valids)),
                                 gen_valid, gen_acc, gen_unique, gen_novel, gen_overall,
                                 up_gen_valid, up_gen_acc, up_gen_unique, up_gen_novel, up_gen_overall]))
                costs = []
                kls = []
                accuracies = []
                precisions = []
                recalls = []
                validities = []
                sys.stdout.flush()
                gc.collect()

                old_count = 100
                new_count = 20

                if i % 10000 == 0:
                    torch.save(model.state_dict(), os.path.join(image_dir, 'model.pt'))

                # evals_left -= 1
                if (len(val_scores) > old_count and
                    sum(val_scores[-new_count:]) / new_count <= sum(val_scores[-old_count:]) / old_count) and \
                        (len(up_val_scores) > old_count and
                    sum(up_val_scores[-new_count:]) / new_count <= sum(up_val_scores[-old_count:]) / old_count):
                    done = True
                    break
        if done:
            break
    print(get_sample_metrics(model, xa_mappings, final_num_gen_conditions, final_num_gen_samples,
                             final_num_images_per_condition, os.path.join(image_dir, 'final')))
    torch.save(model.state_dict(), os.path.join(image_dir, 'model.pt'))


def read_data(dataset, num_train_examples, sample_train_randomly,  identity_features, num_bond_types,
                   num_atom_types, num_max_nodes):
    with open(dataset, 'rb') as infile:
        raw_data = pickle.load(infile, encoding='latin1')[10:]
        if sample_train_randomly:
            raw_data = [d for d in raw_data if random.random() < num_train_examples * 1.0 / len(raw_data)]
        else:
            raw_data = raw_data[:num_train_examples]

    # if FLAT:
    #     model = GCNModelFeedback(4, 32, 16, 32, 0., 1.0)
    # else:
    preprocessed_data = []
    invalids = 0

    xa_mappings = defaultdict(set)
    for molecule in raw_data:
        adjs = [torch.FloatTensor(molecule['adjs'][bond_type])
                for bond_type in ['single', 'double', 'triple', 'aromatic']]
        adjs.append(1. - torch.stack(adjs, dim=0).sum(dim=0))

        features = torch.FloatTensor(
            [[1 if molecule['atoms'][atom_idx] == i else 0 for i in range(num_atom_types)] +
             [1 if atom_idx == i else 0 for i in range(num_max_nodes)] if identity_features else []
             for atom_idx in range(len(molecule['atoms']))]
        )
        if not is_valid(features, torch.stack(adjs, dim=0).max(dim=0)[1]):
            invalids += 1
        # TODO: fix norms and pos weights
        # if use_norm:
        #     norms = sum(adj.shape[0] * adj.shape[0] for adj in adjs) / \
        #             sum(float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2) for adj in adjs)
        # else:
        #     norms = 1.
        norms = 1.
        adj_norms = torch.stack([torch.sparse.FloatTensor(*preprocess_graph(adj)).to_dense() for adj in adjs], dim=0)
        adj_stack = torch.stack([adj for adj in adjs])
        adj_labels = adj_stack.max(dim=0)[1]
        num_pos = adj_stack.sum(dim=1).sum(dim=1).clamp(min=1.)
        pos_weight = ((adj_stack.shape[1] * adj_stack.shape[1]) - num_pos) / num_pos
        no_pos_weight = torch.ones(num_bond_types)
        xa_mappings[tuple(features.reshape(-1).tolist())].add(adj_stack_to_tup(adj_labels))
        preprocessed_data.append((norms, pos_weight, no_pos_weight, adj_norms, features, adj_labels))

    print(invalids)
    return preprocessed_data, xa_mappings
