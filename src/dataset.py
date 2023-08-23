""" A collection of functions to help form the dataloaders for fixed representations/SMILES/Graph representation"""
import numpy as np

import torch
import torch.nn as nn

### for SMILES dataloaders
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence, pad_packed_sequence

### for molecular graph dataloaders
from torch.utils.data.sampler import SubsetRandomSampler
from torch_geometric.data import Data, Dataset

import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdchem import BondType as BT
from rdkit import RDLogger                                                                                                                                                               
RDLogger.DisableLog('rdApp.*')  

### generate fixed representations 
from rdkit import DataStructs
from rdkit.Chem import MACCSkeys
from descriptastorus.descriptors import rdDescriptors, rdNormalizedDescriptors
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors

### form fixed representations (TODO: save all fixed reps in advance?) 
MORGAN_RADIUS = 2
MORGAN_NUM_BITS = 2048

def generate_reps(smiles, mol_rep: str):
    """Generate fixed molecular representations 
    Inputs:
        smiles: a molecule in SMILES representation 
        mol_rep: representation name - options include morganBits, morganCounts, maccs, physchem, rdkit2d, atomPairs
    """
    mol = Chem.MolFromSmiles(smiles)

    if mol_rep == 'morganBits':
        features_vec = AllChem.GetMorganFingerprintAsBitVect(mol, radius=MORGAN_RADIUS, nBits=MORGAN_NUM_BITS)
        features = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(features_vec, features)
    elif mol_rep == 'morganCounts':
        features_vec = AllChem.GetHashedMorganFingerprint(mol, radius=MORGAN_RADIUS, nBits=MORGAN_NUM_BITS)
        features = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(features_vec, features)
    elif mol_rep == 'maccs':
        features_vec = MACCSkeys.GenMACCSKeys(mol)
        features = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(features_vec, features)
    elif mol_rep == 'physchem':
        # calculate physchem descriptors values 
        # Reference: https://github.com/molML/MoleculeACE/blob/main/MoleculeACE/benchmark/featurization.py
        weight = Descriptors.ExactMolWt(mol)
        logp = Descriptors.MolLogP(mol)
        h_bond_donor = Descriptors.NumHDonors(mol)
        h_bond_acceptors = Descriptors.NumHAcceptors(mol)
        rotatable_bonds = Descriptors.NumRotatableBonds(mol)
        atoms = Chem.rdchem.Mol.GetNumAtoms(mol)
        heavy_atoms = Chem.rdchem.Mol.GetNumHeavyAtoms(mol)
        molar_refractivity = Chem.Crippen.MolMR(mol)
        topological_polar_surface_area = Chem.QED.properties(mol).PSA
        formal_charge = Chem.rdmolops.GetFormalCharge(mol)
        rings = Chem.rdMolDescriptors.CalcNumRings(mol)
        # form features matrix
        features = np.array([weight, logp, h_bond_donor, h_bond_acceptors, rotatable_bonds, atoms, heavy_atoms, molar_refractivity, topological_polar_surface_area, formal_charge, rings])
    elif mol_rep == 'rdkit2d':
        # instantiate a descriptors generator
        generator = rdNormalizedDescriptors.RDKit2DNormalized()
        features = generator.process(smiles)[1:]
    elif mol_rep == 'atomPairs':
        features_vec = rdMolDescriptors.GetHashedAtomPairFingerprint(mol, nBits=2048, use2D=True)
        features = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(features_vec, features)
    else:
        raise ValueError('Not defined fingerprint!')

    return features


### form dataloaders for SMILES strings
def inputTensor(smiles, all_symbols, n_symbols):
    """Get tensor for input smiles"""
    smilesTensor = torch.zeros(len(smiles), n_symbols)
    for i, symbol in enumerate(smiles):
        smilesTensor[i][all_symbols.find(symbol)] = 1
    return smilesTensor 

class SMILES_Dataset(Dataset):
    """Form the dataset with inputTensor-label pairs"""
    def __init__(self, SMILES, labels, all_symbols, n_symbols, ):
        self.SMILES = SMILES
        self.labels = labels
        self.all_symbols = all_symbols
        self.n_symbols = n_symbols
        
    def __len__(self):
        return len(self.SMILES)

    def __getitem__(self, idx):
        #get the idx-th SMILES string
        smiles = self.SMILES[idx]
        input_tensor = inputTensor(smiles, self.all_symbols, self.n_symbols)

        return input_tensor.squeeze(), self.labels[idx], input_tensor.size(0)

def SMILES_collate_fn(batch):
    """Pad the SMILES sequences"""
    # get the local_feats, labels and lengths
    local_feats = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch])
    lengths = [item[2] for item in batch]
    
    seq_lengths = torch.LongTensor(lengths)
    padded_seqs = pad_sequence(local_feats)
    packed_seqs = pack_padded_sequence(padded_seqs, seq_lengths, enforce_sorted=False)
    
    return packed_seqs, labels.view(-1)   

### form dataloaders for molecular graphs (reference: moclr - dataset_test)
ATOM_LIST = list(range(1,119))
CHIRALITY_LIST = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER
]
BOND_LIST = [BT.SINGLE, BT.DOUBLE, BT.TRIPLE, BT.AROMATIC]
BONDDIR_LIST = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT
]

class GraphDataset(Dataset):
    """Form the dataset with graph-label pairs"""
    # def __init__(self, SMILES, labels, all_symbols, n_symbols, ):
    def __init__(self, SMILES, labels):
        super(Dataset, self).__init__()
        self.SMILES = SMILES
        self.labels = labels

    def __getitem__(self, idx):
        mol = Chem.MolFromSmiles(self.SMILES[idx])
        mol = Chem.AddHs(mol)

        N = mol.GetNumAtoms()
        M = mol.GetNumBonds()

        type_idx = []
        chirality_idx = []
        atomic_number = []
        for atom in mol.GetAtoms():
            type_idx.append(ATOM_LIST.index(atom.GetAtomicNum()))
            chirality_idx.append(CHIRALITY_LIST.index(atom.GetChiralTag()))
            atomic_number.append(atom.GetAtomicNum())

        x1 = torch.tensor(type_idx, dtype=torch.long).view(-1,1)
        x2 = torch.tensor(chirality_idx, dtype=torch.long).view(-1,1)
        x = torch.cat([x1, x2], dim=-1)

        row, col, edge_feat = [], [], []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            row += [start, end]
            col += [end, start]
            edge_feat.append([
                BOND_LIST.index(bond.GetBondType()),
                BONDDIR_LIST.index(bond.GetBondDir())
            ])
            edge_feat.append([
                BOND_LIST.index(bond.GetBondType()),
                BONDDIR_LIST.index(bond.GetBondDir())
            ])

        edge_index = torch.tensor([row, col], dtype=torch.long)
        edge_attr = torch.tensor(np.array(edge_feat), dtype=torch.long)

        y = torch.tensor(self.labels[idx], dtype=torch.long).view(1,-1)

        data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr)

        return data

    def __len__(self):
        return len(self.SMILES)
