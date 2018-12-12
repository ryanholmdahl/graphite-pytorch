import pickle
import numpy as np
from rdkit import Chem

with open('data/data_qm9.pkl') as infile:
    d = pickle.load(infile)

molecules = []
for smiles in d['smiles_original']:
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        continue
    adjs = {
        bond_type: np.zeros((m.GetNumAtoms(), m.GetNumAtoms())) for bond_type in [
            Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE,
            Chem.rdchem.BondType.AROMATIC
        ]
    }
    atoms = []
    for bond in m.GetBonds():
        adjs[bond.GetBondType()][bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()] = 1
        adjs[bond.GetBondType()][bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()] = 1
    for atom_idx in range(m.GetNumAtoms()):
        atoms.append(m.GetAtomWithIdx(atom_idx).GetAtomicNum() - 6)
    molecules.append({
        'atoms': atoms,
        'adjs': {
            'single': adjs[Chem.rdchem.BondType.SINGLE],
            'double': adjs[Chem.rdchem.BondType.DOUBLE],
            'triple': adjs[Chem.rdchem.BondType.TRIPLE],
            'aromatic': adjs[Chem.rdchem.BondType.AROMATIC],
        }
    })
    if len(molecules) % 100 == 0:
        print(len(molecules))
    if len(molecules) == 100:
        with open('100_molecules.pkl', 'wb') as outfile:
            pickle.dump(molecules, outfile)
with open('all_molecules.pkl', 'wb') as outfile:
    pickle.dump(molecules, outfile)
