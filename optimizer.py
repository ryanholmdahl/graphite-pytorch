import torch.nn as nn
import torch.nn.functional as F
import torch
from rdkit import Chem
from collections import defaultdict
from rdkit.Chem import Draw
import random
import shutil
import os
import math
import sys
import numpy as np


def get_loss(preds, labels, z_mean, z_log_std, num_nodes, norms, pos_weight):
    cost = F.cross_entropy(preds.transpose(0, 2).reshape(num_nodes * num_nodes, -1),
                           labels.view(num_nodes * num_nodes), weight=pos_weight) * norms
    kl = (0.5 / num_nodes) * (1 + 2 * z_log_std - z_mean**2 - torch.exp(z_log_std)**2).sum()
    return cost, -kl


def get_accuracy(preds, labels):
    # preds: [5 x nodes x nodes]
    # labels [nodes x nodes]
    return (preds.max(dim=0)[1] == labels).float().sum() / (labels.shape[0] * labels.shape[0])
    # return ((preds > 0).float() == labels).float().sum() / labels.shape[0]


def get_recall(preds, labels):
    return ((preds.max(dim=0)[1] == labels).float() * (labels < 4).float()).sum() / (labels < 4).float().sum()


def get_precision(preds, labels):
    pred_inds = preds.max(dim=0)[1]
    return ((pred_inds == labels).float() * (pred_inds < 4).float()).sum() / max((pred_inds < 4).float().sum(), 1)


def x_to_node_list(x):
    x = x.tolist()
    node_list = [ind + 6 for i in range(len(x)) for ind in range(4) if x[i][ind] > 0]
    assert len(x) == len(node_list)
    return node_list


def is_valid(x, a):
    m = get_mol(x, a)
    try:
        Chem.rdmolops.SanitizeMol(m)
        s = Chem.MolToSmiles(m)
        m = Chem.MolFromSmiles(s)
    except:
        m = None
    return m is not None


def get_mol(x, a):
    return MolFromGraphs(x_to_node_list(x), a)


def get_sanitized(m):
    Chem.rdmolops.SanitizeMol(m)
    bond_types = {
        Chem.rdchem.BondType.SINGLE: 0,
        Chem.rdchem.BondType.DOUBLE: 1,
        Chem.rdchem.BondType.TRIPLE: 2,
        Chem.rdchem.BondType.AROMATIC: 3,
    }
    features = []
    adjs = np.zeros((4, m.GetNumAtoms(), m.GetNumAtoms()))
    for bond in m.GetBonds():
        adjs[bond_types[bond.GetBondType()], bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()] = 1
        adjs[bond_types[bond.GetBondType()], bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()] = 1
    for atom_idx in range(m.GetNumAtoms()):
        features.append([1 if i == m.GetAtomWithIdx(atom_idx).GetAtomicNum() - 6 else 0 for i in range(4)])
        adjs[:, atom_idx, atom_idx] += 1
    return np.array(features), adjs


def MolFromGraphs(node_list, adjacency_matrix):

    # create empty editable mol object
    mol = Chem.RWMol()

    # add atoms to mol and keep track of index
    node_to_idx = {}
    for i in range(len(node_list)):
        a = Chem.Atom(node_list[i])
        molIdx = mol.AddAtom(a)
        node_to_idx[i] = molIdx

    bond_idx = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE,
                Chem.rdchem.BondType.AROMATIC, None]

    # add bonds between adjacent atoms
    for ix, row in enumerate(adjacency_matrix):
        for iy, bond_type in enumerate(row):

            # only traverse half the matrix
            if iy <= ix:
                continue

            # add relevant bond type (there are many more of these)
            if bond_idx[bond_type] is not None:
                mol.AddBond(node_to_idx[ix], node_to_idx[iy], bond_idx[bond_type])

    # Convert RWMol to Mol object
    mol = mol.GetMol()

    return mol


def get_components(a):
    seen = set()

    def connected(i):
        found = set()
        to_check = [i]
        while len(to_check) > 0:
            i = to_check.pop()
            if i in found:
                continue
            found.add(i)
            seen.add(i)
            for j in range(len(a[i])):
                if a[i][j] < 4:
                    to_check.append(j)
            for j in range(len(a)):
                if a[j][i] < 4:
                    to_check.append(j)
        return found

    components = []
    for i in range(len(a)):
        if i not in seen:
            components.append(list(connected(i)))
    return components


def get_is_connected(x, a):
    largest = max(get_components(a), key=len)
    return len(largest) == len(x)


def adj_stack_to_tup(adj_stack):
    l = []
    for i in range(adj_stack.shape[0]):
        for j in range(adj_stack.shape[1]):
            if j <= i:
                continue
            l.append(int(adj_stack[i, j]))
    return tuple(l)


def upsample(a):
    assert a.shape[1] == a.shape[2]
    assert a.shape[0] == 5
    a_orig = a.max(dim=0)[1]
    # find all connected components
    components = [torch.LongTensor(t) for t in get_components(a.max(dim=0)[1])]
    prev_component = components[0]
    for j in range(1, len(components)):
        weights = a[:4, :, :].index_select(1, prev_component).index_select(2, components[j])
        assert weights.shape == (4, len(prev_component), len(components[j])), weights.shape
        edge_b = weights.max(dim=1)[0].max(dim=1)[0].argmax(dim=0)
        edge = torch.argmax(weights[edge_b, :, :])
        edge_i = prev_component[edge % len(prev_component)]
        edge_j = components[j][int(edge / len(prev_component))]
        a[edge_b, edge_i, edge_j] = np.infty
        a[edge_b, edge_j, edge_i] = np.infty
        prev_component = torch.cat([prev_component, components[j]], dim=0)
    a = a.max(dim=0)[1]
    assert len(components) == 1 or (a - a_orig).sum() != 0, (a, a_orig, a - a_orig)
    assert get_is_connected([1 for _ in range(a.shape[1])], a), a
    return a
    # add most likely edge between each component
    # for n in range(a.shape[1]):
    #     if a[:, n, :].max(dim=0)[1].sum() == 4 * a.shape[1]:
    #         a[4, n, :] = -np.infty
    #         a[4, :, n] = -np.infty
    # return a.max(dim=0)[1]


def do_images(xs, molecules, conditions, samples, images_per_condition, output_dir):
    os.makedirs(os.path.join(output_dir, 'valid'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'accurate'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'invalid'), exist_ok=True)
    for c in range(conditions):
        image_mols = random.sample(list(zip(xs[c * samples:(c+1) * samples],
                                            molecules[c * samples:(c+1) * samples])), images_per_condition)
        for i, (x, a) in enumerate(image_mols):
            m = get_mol(x, a[0])
            try:
                err = Chem.rdmolops.SanitizeMol(m, catchErrors=True)
                if err == 0 and is_valid(x, a[0]):
                    if get_is_connected(x, a[0]):
                        Draw.MolToImageFile(m, os.path.join(output_dir, 'accurate', 'c_{}_all_{}.png'.format(c, i)))
                    else:
                        Draw.MolToImageFile(m, os.path.join(output_dir, 'valid', 'c_{}_all_{}.png'.format(c, i)))
                else:
                    Draw.MolToImageFile(m, os.path.join(output_dir, 'c_{}_all_{}.png'.format(c, i)))
            except ValueError:
                continue


def get_sample_metrics(model, dataset, conditions=100, samples=1000, images_per_condition=1, output_dir='images'):
    # TODO: don't hardcode feature shape
    x_options = [torch.FloatTensor(np.array(x)).reshape(-1, 13) for x in dataset.keys() for _ in range(len(dataset[x]))]
    xs = random.sample(x_options, conditions)
    xs = [x for x in xs for _ in range(samples)]
    molecules = [model.sample(x, len(x)) for x in xs]
    upsampled_molecules = [(upsample(a[1]), ) for a in molecules]
    os.makedirs(os.path.join(output_dir, 'normal'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'upsampled'), exist_ok=True)

    do_images(xs, molecules, conditions, samples, images_per_condition, os.path.join(output_dir, 'normal'))
    do_images(xs, upsampled_molecules, conditions, samples, images_per_condition, os.path.join(output_dir, 'upsampled'))

    valid_molecules = [(x, a[0]) for x, a in zip(xs, molecules) if is_valid(x, a[0])]
    accurate_valid_molecules = []
    for x, a in valid_molecules:
        if get_is_connected(x, a):
            accurate_valid_molecules.append((x, a,
                                             tuple(x.reshape(-1).tolist()),
                                             adj_stack_to_tup(a)))
    unique_accurate_valid_molecules = []
    unique_accurate_valid_molecule_dict = defaultdict(set)
    for x, a, x_tup, a_tup in accurate_valid_molecules:
        if x_tup not in unique_accurate_valid_molecule_dict or a_tup not in unique_accurate_valid_molecule_dict[x_tup]:
            unique_accurate_valid_molecules.append((x, a, x_tup, a_tup))
            unique_accurate_valid_molecule_dict[x_tup].add(a_tup)

    novel_unique_accurate_valid_molecules = []
    for x, a, x_tup, a_tup in unique_accurate_valid_molecules:
        assert x_tup in dataset
        if a_tup not in dataset[x_tup]:
            novel_unique_accurate_valid_molecules.append((x, a))

    up_valid_molecules = [(x, a[0]) for x, a in zip(xs, upsampled_molecules) if is_valid(x, a[0])]
    up_accurate_valid_molecules = []
    for x, a in up_valid_molecules:
        if get_is_connected(x, a):
            up_accurate_valid_molecules.append((x, a,
                                                tuple(x.reshape(-1).tolist()),
                                                adj_stack_to_tup(a)))
    assert len(up_accurate_valid_molecules) == len(up_valid_molecules)
    up_unique_accurate_valid_molecules = []
    up_unique_accurate_valid_molecule_dict = defaultdict(set)
    for x, a, x_tup, a_tup in up_accurate_valid_molecules:
        if x_tup not in up_unique_accurate_valid_molecule_dict or \
                a_tup not in up_unique_accurate_valid_molecule_dict[x_tup]:
            up_unique_accurate_valid_molecules.append((x, a, x_tup, a_tup))
            up_unique_accurate_valid_molecule_dict[x_tup].add(a_tup)

    up_novel_unique_accurate_valid_molecules = []
    for x, a, x_tup, a_tup in up_unique_accurate_valid_molecules:
        assert x_tup in dataset
        if a_tup not in dataset[x_tup]:
            up_novel_unique_accurate_valid_molecules.append((x, a))

    return (
        len(valid_molecules) * 1.0 / len(molecules),
        len(accurate_valid_molecules) * 1.0 / len(valid_molecules)
            if len(valid_molecules) > 0 else -1,
        len(unique_accurate_valid_molecules) * 1.0 / len(accurate_valid_molecules)
            if len(accurate_valid_molecules) > 0 else -1,
        len(novel_unique_accurate_valid_molecules) * 1.0 / len(unique_accurate_valid_molecules)
            if len(unique_accurate_valid_molecules) > 0 else -1,
        len(up_valid_molecules) / len(molecules),
        len(up_accurate_valid_molecules) * 1.0 / len(up_valid_molecules)
            if len(up_valid_molecules) > 0 else -1,
        len(up_unique_accurate_valid_molecules) * 1.0 / len(up_accurate_valid_molecules)
            if len(up_accurate_valid_molecules) > 0 else -1,
        len(up_novel_unique_accurate_valid_molecules) * 1.0 / len(up_unique_accurate_valid_molecules)
            if len(up_unique_accurate_valid_molecules) > 0 else -1,
    )
