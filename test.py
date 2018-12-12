from optimizer import get_mol
import torch
from rdkit import Chem
from rdkit.Chem import Draw

x = torch.FloatTensor([[1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
        [1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
        [1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
        [1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.]])
a = torch.LongTensor([[0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 3],
        [0, 0, 3, 1]])
m = get_mol(x, a)

Chem.rdmolops.SanitizeMol(m)

s = Chem.MolToSmiles(m)
print(s)
m = Chem.MolFromSmiles(s)

for a in m.GetAtoms():
    print(a.GetSymbol(), a.GetExplicitValence())

for b in m.GetBonds():
    print(b)
Draw.MolToImageFile(m, 'images/kill_me.png')
