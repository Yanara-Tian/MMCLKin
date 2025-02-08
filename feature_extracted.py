import os
import dgl
import numpy as np  
import pandas as pd
import networkx as nx
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix 
from rdkit.Chem.rdchem import HybridizationType, BondType 
import gvp
import torch
import gvp.data
import torch_cluster
from Bio.PDB import PDBParser
from torch_geometric.data import Data
from p2rank_pre_pock import three_to_one, get_clean_res_list
from rdkit.Chem import rdFMCS

 
def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        pass
    return list(map(lambda s: x == s, allowable_set))


def generate_smiles_nodes_edges_coords_graph_index_features(type, m):
    if type == 'smile':
        mol = Chem.MolFromSmiles(m)
    elif type == 'mol2':
        mola = Chem.MolFromMol2File(m)
        mol_smile = Chem.MolToSmiles(mola)
        mol = Chem.MolFromSmiles(mol_smile)
    elif type == 'sdf':
        mola = Chem.MolFromMolFile(m)
        mol_smile = Chem.MolToSmiles(mola)
        mol = Chem.MolFromSmiles(mol_smile)

    # nodes feature
    atom_type = []
    atom_number = []
    atom_hybridization =[]
    atomHs = []
    atom_charge = []
    atom_imvalence = []
    atom_aromatic = []
    atom_explicit = []
    atoms = mol.GetAtoms()
    num_atoms = mol.GetNumAtoms()
    for i, atom in enumerate(atoms):
        atom_type.append(atom.GetSymbol())  #C,H,O
        atom_number.append(atom.GetAtomicNum())  #获取原子符号
        atom_hybridization.append(atom.GetHybridization())  # SP, SP2
        atomHs.append(atom.GetTotalNumHs())  # 0,1,2,3
        atom_charge.append(atom.GetFormalCharge()) # 0, +1,-1, +2, -2, +3, -3
        atom_imvalence.append(atom.GetImplicitValence())  #获得原子的隐式化合价 0 1 2 3
        atom_aromatic.append(1 if atom.GetIsAromatic() else 0)
        atom_explicit.append(atom.GetExplicitValence())
        
    nodes_hybridization = [one_of_k_encoding(h, [HybridizationType.SP, HybridizationType.SP2, HybridizationType.SP3, HybridizationType.SP3D, HybridizationType.SP3D2]) for h in atom_hybridization]
    nodes_type = [one_of_k_encoding(t[0], ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) for t in atom_type]
    atom_Hss = [one_of_k_encoding(t, [0,1,2,3,4]) for t in atomHs]
    atom_imvalence = [one_of_k_encoding(t, [0,1,2,3,4,5,6]) for t in atom_imvalence]
    atom_charge = [one_of_k_encoding(t, [0,1,2,3,4,5,6]) for t in atom_charge]
    atom_explicit = [one_of_k_encoding(t, [0,1,2,3,4]) for t in atom_explicit]
    
    # bonds feature
    bond_type = []
    bond_conj = []
    bond_ring =[]
    row = []
    col = []
    bonds = mol.GetBonds()
    
    for bond in bonds:   
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        bond_type += 2 * [bond.GetBondType()]
        bond_ring += 2 * [bond.IsInRing()]
        bond_conj += 2 * [bond.GetIsConjugated()]
    
    edge_conj = [one_of_k_encoding(t, [True, False]) for t in bond_conj]
    edge_ring = [one_of_k_encoding(t, [True, False]) for t in bond_ring]   
    bond_index = torch.LongTensor([row, col])
    edge_type = [one_of_k_encoding(t, [BondType.SINGLE, BondType.DOUBLE, BondType.TRIPLE, BondType.AROMATIC]) for t in bond_type]
    # edge_attri = torch.FloatTensor(edge_type)
    perm = (bond_index[0] * num_atoms + bond_index[1]).argsort()  # argsort()函数返回的是数组从小到大的索引值
    edge_index = bond_index[:, perm]
    atom_f1 = torch.tensor([atom_number, atom_aromatic, atomHs], dtype=torch.float).t().contiguous()
    
    # concatence features
    mol_all_nodes_feature = torch.cat([torch.FloatTensor(nodes_hybridization), torch.FloatTensor(nodes_type), 
                          torch.FloatTensor(atom_Hss), torch.FloatTensor(atom_imvalence), torch.FloatTensor(atom_charge), torch.FloatTensor(atom_explicit),atom_f1], dim=-1)
    mol_all_edges_feature = torch.cat([torch.FloatTensor(edge_conj),torch.FloatTensor(edge_ring), torch.FloatTensor(edge_type)], dim=-1)
    
    # generate_atom_coordinate
    if type == 'smile':
        AllChem.EmbedMolecule(mol)
        AllChem.MMFFOptimizeMolecule(mol)
        mol_all_coords_feature = torch.FloatTensor(mol.GetConformer().GetPositions()) 
    elif type == 'mol2' or type == 'sdf':
        mcs = rdFMCS.FindMCS([mola,mol])
        mcs_mol = Chem.MolFromSmarts(mcs.smartsString)
        matcha = mola.GetSubstructMatch(mcs_mol)
        match = mol.GetSubstructMatch(mcs_mol)
        tuple_dict = dict(zip(match, matcha))
        sorted_tuple1 = sorted(match)
        sorted_tuple2 = [tuple_dict[key] for key in sorted_tuple1]
        mol_all_co_fea = torch.FloatTensor(mola.GetConformer().GetPositions())
        mol_all_coords_feature = torch.tensor([], dtype=torch.float32)
        for i in sorted_tuple2:
            mol_all_coords_feature = torch.cat((mol_all_coords_feature,mol_all_co_fea[i].view(1, -1)),0)
            
    # generate_graph
    atom_matrix = GetAdjacencyMatrix(mol)
    graph = nx.convert_matrix.from_numpy_matrix(atom_matrix)
    mol_dgl_graph = dgl.from_networkx(graph)
    mol_dgl_graph.ndata['h'] = torch.FloatTensor(mol_all_nodes_feature)
    
    return mol_all_nodes_feature, mol_all_edges_feature, mol_all_coords_feature, mol_dgl_graph, edge_index, mol_smile, atom_type


def generate_smiles_nodes_edges_coords_graph_features(type, m):
    if type == 'smile':
        mol = Chem.MolFromSmiles(m)
    elif type == 'mol2':
        mol = Chem.MolFromMol2File(m)
        smile = Chem.MolToSmiles(mol)
        mol = Chem.MolFromSmiles(smile)
    elif type == 'sdf':
        mol = Chem.MolFromMolFile(m)
        smi = Chem.MolToSmiles(mol)

    # nodes feature
    atom_type = []
    atom_number = []
    atom_hybridization =[]
    atomHs = []
    atom_charge = []
    atom_imvalence = []
    atom_aromatic = []
    atom_explicit = []
    atoms = mol.GetAtoms()
    num_atoms = mol.GetNumAtoms()
    for i, atom in enumerate(atoms):
        atom_type.append(atom.GetSymbol())  #C,H,O
        atom_number.append(atom.GetAtomicNum())  #获取原子符号
        atom_hybridization.append(atom.GetHybridization())  # SP, SP2
        atomHs.append(atom.GetTotalNumHs())  # 0,1,2,3
        atom_charge.append(atom.GetFormalCharge()) # 0, +1,-1, +2, -2, +3, -3
        atom_imvalence.append(atom.GetImplicitValence())  #获得原子的隐式化合价 0 1 2 3
        atom_aromatic.append(1 if atom.GetIsAromatic() else 0)
        atom_explicit.append(atom.GetExplicitValence())
        
    nodes_hybridization = [one_of_k_encoding(h, [HybridizationType.SP, HybridizationType.SP2, HybridizationType.SP3, HybridizationType.SP3D, HybridizationType.SP3D2]) for h in atom_hybridization]
    nodes_type = [one_of_k_encoding(t[0], ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) for t in atom_type]
    atom_Hss = [one_of_k_encoding(t, [0,1,2,3,4]) for t in atomHs]
    atom_imvalence = [one_of_k_encoding(t, [0,1,2,3,4,5,6]) for t in atom_imvalence]
    atom_charge = [one_of_k_encoding(t, [0,1,2,3,4,5,6]) for t in atom_charge]
    atom_explicit = [one_of_k_encoding(t, [0,1,2,3,4]) for t in atom_explicit]
    
    # bonds feature
    bond_type = []
    bond_conj = []
    bond_ring =[]
    row = []
    col = []
    bonds = mol.GetBonds()
    
    for bond in bonds:   
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        bond_type += 2 * [bond.GetBondType()]
        bond_ring += 2 * [bond.IsInRing()]
        bond_conj += 2 * [bond.GetIsConjugated()]
    
    edge_conj = [one_of_k_encoding(t, [True, False]) for t in bond_conj]
    edge_ring = [one_of_k_encoding(t, [True, False]) for t in bond_ring]   
    bond_index = torch.LongTensor([row, col])
    edge_type = [one_of_k_encoding(t, [BondType.SINGLE, BondType.DOUBLE, BondType.TRIPLE, BondType.AROMATIC]) for t in bond_type]
    # edge_attri = torch.FloatTensor(edge_type)
    perm = (bond_index[0] * num_atoms + bond_index[1]).argsort()  # argsort()函数返回的是数组从小到大的索引值
    edge_index = bond_index[:, perm]
    atom_f1 = torch.tensor([atom_number, atom_aromatic, atomHs], dtype=torch.float).t().contiguous()
    
    # concatence features
    mol_all_nodes_feature = torch.cat([torch.FloatTensor(nodes_hybridization), torch.FloatTensor(nodes_type), 
                          torch.FloatTensor(atom_Hss), torch.FloatTensor(atom_imvalence), torch.FloatTensor(atom_charge), torch.FloatTensor(atom_explicit),atom_f1], dim=-1)
    mol_all_edges_feature = torch.cat([torch.FloatTensor(edge_conj),torch.FloatTensor(edge_ring), torch.FloatTensor(edge_type)], dim=-1)
    
    # generate_atom_coordinate
    if type == 'smile':
        AllChem.EmbedMolecule(mol)
        AllChem.MMFFOptimizeMolecule(mol)
        mol_all_coords_feature = torch.FloatTensor(mol.GetConformer().GetPositions()) 
    elif type == 'mol2' or type == 'sdf':
        mol_all_coords_feature = torch.FloatTensor(mol.GetConformer().GetPositions())
    
    # generate_graph
    atom_matrix = GetAdjacencyMatrix(mol)
    graph = nx.convert_matrix.from_numpy_matrix(atom_matrix)
    mol_dgl_graph = dgl.from_networkx(graph)
    mol_dgl_graph.ndata['h'] = torch.FloatTensor(mol_all_nodes_feature)
    
    return mol_all_nodes_feature, mol_all_edges_feature, mol_all_coords_feature, mol_dgl_graph, edge_index

def generate_graph_feature(pdb):
    parser = PDBParser(QUIET=True)
    s = parser.get_structure(f'{pdb}', pdb)
    res_list = get_clean_res_list(s.get_residues(), verbose=False, ensure_ca_exist=True)
    res_list = [res for res in res_list if (('N' in res) and ('CA' in res) and ('C' in res) and ('O' in res))] # 215
    structure = {}
    structure['name'] = "placeholder"
    structure['seq'] = "".join([three_to_one.get(res.resname) for res in res_list])  
    coords = []
    for res in res_list:
        res_coords = []
        for atom in [res['N'], res['CA'], res['C'], res['O']]:
            res_coords.append(list(atom.coord))
        coords.append(res_coords) # 215*3
    structure['coords'] = coords 
    torch.set_num_threads(1)
    # print(len(structure['seq']))    
    dataset = gvp.data.ProteinGraphDataset([structure])
    protein = dataset[0]
    for i, edge_index in enumerate(protein.edge_index.tolist()):
        if i == 0:
            s_edge = torch.LongTensor(edge_index)
        elif i == 1:
            t_edge = torch.LongTensor(edge_index)
    
    protein_graph = dgl.graph((s_edge, t_edge))
    protein_graph.ndata['h'] = torch.FloatTensor(protein.node_s)
    
    return protein, protein_graph, coords

def _positional_embeddings(edge_index,  
                               num_embeddings=None,
                               period_range=[2, 1000]):
        # From https://github.com/jingraham/neurips19-graph-protein-design
        num_positional_embeddings = 16
        num_embeddings = num_embeddings or num_positional_embeddings
        d = edge_index[0] - edge_index[1]
     
        frequency = torch.exp(
            torch.arange(0, num_embeddings, 2, dtype=torch.float32)
            * -(np.log(10000.0) / num_embeddings)
        )
        angles = d.unsqueeze(-1) * frequency
        E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
        return E

def _rbf(D, D_min=0., D_max=20., D_count=16, device='cpu'):
    '''
    From https://github.com/jingraham/neurips19-graph-protein-design
    
    Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, if `D` has shape [...dims], then the returned tensor will have
    shape [...dims, D_count].
    '''
    D_mu = torch.linspace(D_min, D_max, D_count, device=device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF

def generate_pocket_features(pdb, csv, toFile):
    # pocket_df = pd.read_csv(csv)
    pocket_df = pd.read_csv(csv, usecols=['   center_x','   center_y','   center_z'])
    center_x = pocket_df.loc[0, '   center_x']
    center_y = pocket_df.loc[0, '   center_y']
    center_z = pocket_df.loc[0, '   center_z']
    x_min = center_x - 10
    x_max = center_x + 10
    y_min = center_y - 10
    y_max = center_y + 10
    z_min = center_z - 10
    z_max = center_z + 10
    protein, protein_graph = generate_graph_feature(pdb)
    pocket_res_coord = []
    pocket_node_s = []
    pocket_idx = []
    # protein_node = protein.node_s.tolist()
    for idx, res_coord in enumerate(protein.x.tolist()):
        if x_min < res_coord[0] < x_max and y_min < res_coord[1] < y_max and z_min < res_coord[2] < z_max:
            pocket_res_coord.append(res_coord)
            pocket_idx.append(idx)
            pocket_node_s.append(protein.node_s.tolist()[idx])

    pocket_coord_feature = torch.FloatTensor(pocket_res_coord)
    pocket_res_feature = torch.tensor(pocket_node_s, dtype=torch.float) 
    
    # generate the edge feature of pocket
    poc_X_ca = torch.as_tensor(pocket_coord_feature, dtype=torch.float32)
    pocket_edge_index = torch_cluster.knn_graph(poc_X_ca, k=30)
    pos_embeddings = _positional_embeddings(pocket_edge_index)
    E_vectors = poc_X_ca[pocket_edge_index[0]] - poc_X_ca[pocket_edge_index[1]]
    rbf = _rbf(E_vectors.norm(dim=-1), D_count=16)
    pocket_edge_feature = torch.cat([rbf, pos_embeddings],dim=-1)
    
    # generate graph
    for i, edge_index in enumerate(pocket_edge_index.tolist()):
        if i == 0:
            s_edge = torch.LongTensor(edge_index)
        elif i == 1:
            t_edge = torch.LongTensor(edge_index)
    
    pocket_graph = dgl.graph((s_edge, t_edge))
    pocket_graph.ndata['h'] = torch.FloatTensor(pocket_res_feature)
    protein_pocket_feats = Data(pro_graph=protein_graph,
                                pro_res_fea=protein.node_s,
                                pro_coords=protein.x,
                                pro_edge_fea=protein.edge_s,
                                pro_edge_index=protein.edge_index,
                                poc_graph=pocket_graph,
                                poc_coords=pocket_coord_feature,
                                poc_res_fea=pocket_res_feature,
                                poc_edge_fea=pocket_edge_feature,
                                poc_edge_index=pocket_edge_index )
    print(f'poc_graph:{protein_pocket_feats.poc_graph}')
    print(f'poc_coords:{protein_pocket_feats.poc_coords.shape}')
    print(f'poc_res_fea:{protein_pocket_feats.poc_res_fea.shape}')
    print(f'poc_edge_index:{protein_pocket_feats.poc_edge_index.shape}')
    print(f'poc_edge_fea:{protein_pocket_feats.poc_edge_fea.shape}')

    torch.save(protein_pocket_feats, toFile)
    return protein_pocket_feats


