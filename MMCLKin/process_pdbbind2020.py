import os
import pandas as pd
import numpy as np
import math
import torch
from feature_extracted import generate_graph_feature
from feature_extracted import generate_smiles_nodes_edges_coords_graph_index_features
from transformers import AutoModel, AutoTokenizer
from torch_geometric.data import Data
from torch_scatter import scatter_min
import esm
from Bio.PDB.PDBIO import PDBIO
from Bio.PDB.PDBIO import Select
import sys
from Bio.PDB import PDBParser

class Logger(object):
    def __init__(self, file_name="Default.log", stream=sys.stdout):
        self.terminal = stream
        self.log = open(file_name, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

sa_path = './pdbbind2020'
log_file_name = f"{sa_path}/pdbbind2020_process.log"
sys.stdout = Logger(log_file_name)
sys.stderr = Logger(log_file_name)

def generate_protein_sequence(pdb_path):
    aa_codes = {'ALA':'A', 'CYS':'C', 'ASP':'D', 'GLU':'E', 'PHE':'F', 'GLY':'G', 'HIS':'H', 'LYS':'K',
     'ILE':'I', 'LEU':'L', 'MET':'M', 'ASN':'N', 'PRO':'P', 'GLN':'Q', 'ARG':'R', 'SER':'S', 'THR':'T',
     'VAL':'V', 'TYR':'Y', 'TRP':'W'}
    amino_acid_sequence = ''
    chain_num = 0
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)
    for model in structure:
        for chain in model:
            chain_num = chain_num + 1
            for residue in chain:
                if residue.get_id()[0] == " " and residue.get_resname() != "HOH":
                    if (('N' in residue) and ('CA' in residue) and ('C' in residue) and ('O' in residue)):
                        amino_acid_sequence += aa_codes[residue.get_resname()]
    length = len(amino_acid_sequence)
    return amino_acid_sequence, chain_num

def generate_inner_coor(pos, atom_feats, edge_index):
    cutoff = 8.0
    num_nodes = atom_feats.size(0)
    j, i = edge_index
    vecs = pos[j] - pos[i]
    dist = vecs.norm(dim=-1)

    # Calculate distances.
    _, argmin0 = scatter_min(dist, i, dim_size=num_nodes)
    argmin0[argmin0 >= len(i)] = 0
    n0 = j[argmin0]
    add = torch.zeros_like(dist).to(dist.device)
    add[argmin0] = cutoff
    dist1 = dist + add

    _, argmin1 = scatter_min(dist1, i, dim_size=num_nodes)
    argmin1[argmin1 >= len(i)] = 0
    n1 = j[argmin1]
    # --------------------------------------------------------

    _, argmin0_j = scatter_min(dist, j, dim_size=num_nodes)
    argmin0_j[argmin0_j >= len(j)] = 0
    n0_j = i[argmin0_j]

    add_j = torch.zeros_like(dist).to(dist.device)
    add_j[argmin0_j] = cutoff
    dist1_j = dist + add_j

    # i[argmin] = range(0, num_nodes)
    _, argmin1_j = scatter_min(dist1_j, j, dim_size=num_nodes)
    argmin1_j[argmin1_j >= len(j)] = 0
    n1_j = i[argmin1_j]

    # n0, n1 for i
    n0 = n0[i]
    n1 = n1[i]

    # n0, n1 for j
    n0_j = n0_j[j]
    n1_j = n1_j[j]

    mask_iref = n0 == j
    iref = torch.clone(n0)
    iref[mask_iref] = n1[mask_iref]
    idx_iref = argmin0[i]
    idx_iref[mask_iref] = argmin1[i][mask_iref]

    mask_jref = n0_j == i
    jref = torch.clone(n0_j)
    jref[mask_jref] = n1_j[mask_jref]
    idx_jref = argmin0_j[j]
    idx_jref[mask_jref] = argmin1_j[j][mask_jref]

    pos_ji, pos_in0, pos_in1, pos_iref, pos_jref_j = (
        vecs,
        vecs[argmin0][i],
        vecs[argmin1][i],
        vecs[idx_iref],
        vecs[idx_jref]
    )

    # Calculate angles.
    a = ((-pos_ji) * pos_in0).sum(dim=-1)
    b = torch.cross(-pos_ji, pos_in0).norm(dim=-1)
    theta = torch.atan2(b, a)
    theta[theta < 0] = theta[theta < 0] + math.pi

    # Calculate torsions.
    dist_ji = pos_ji.pow(2).sum(dim=-1).sqrt()
    plane1 = torch.cross(-pos_ji, pos_in0)
    plane2 = torch.cross(-pos_ji, pos_in1)
    a = (plane1 * plane2).sum(dim=-1)  # cos_angle * |plane1| * |plane2|
    b = (torch.cross(plane1, plane2) * pos_ji).sum(dim=-1) / dist_ji
    phi = torch.atan2(b, a)
    phi[phi < 0] = phi[phi < 0] + math.pi

    # Calculate right torsions.
    plane1 = torch.cross(pos_ji, pos_jref_j)
    plane2 = torch.cross(pos_ji, pos_iref)
    a = (plane1 * plane2).sum(dim=-1)  # cos_angle * |plane1| * |plane2|
    b = (torch.cross(plane1, plane2) * pos_ji).sum(dim=-1) / dist_ji
    tau = torch.atan2(b, a)
    tau[tau < 0] = tau[tau < 0] + math.pi
        
    return dist, theta, phi, tau


three_to_one = {'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 
                'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 
                'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'}

def get_clean_res_list(res_list, verbose=False, ensure_ca_exist=False, bfactor_cutoff=None):
    clean_res_list = []
    for res in res_list:
        hetero, resid, insertion = res.full_id[-1]
        if hetero == ' ':
            a = res.resname
            if res.resname not in three_to_one:
                if verbose:
                    print(res, "has non-standard resname")
                continue
            if (not ensure_ca_exist) or ('CA' in res):
                if bfactor_cutoff is not None:
                    ca_bfactor = float(res['CA'].bfactor)
                    if ca_bfactor < bfactor_cutoff:
                        continue
                clean_res_list.append(res)
        else:
            if verbose:
                print(res, res.full_id, "is hetero")
    return clean_res_list

def gen_seq_list(pdb):
    parser = PDBParser(QUIET=True)
    s = parser.get_structure(pdb, pdb)
    res_list = get_clean_res_list(s.get_residues(), verbose=False, ensure_ca_exist=True)
    res_list = [res for res in res_list if (('N' in res) and ('CA' in res) and ('C' in res) and ('O' in res))]
    seq_list = [f'{res.parent.id}{res.id[1]}{res.id[2]}' for res in res_list]
    return seq_list
 
def clean_pock(poc_path, cle_ca_path):
    parser = PDBParser(QUIET=True)
    s = parser.get_structure("x", poc_path)
    all_res = get_clean_res_list(s.get_residues(), verbose=False, ensure_ca_exist=True) 
    all_atoms = [atom for res in all_res for atom in res.get_atoms()]
    chains = np.array([atom.full_id[2] for atom in all_atoms])
    
    class MySelect(Select):
        def accept_residue(self, residue, chains=chains):
            pdb, _, chain, (_, resid, insertion) = residue.full_id
            if chain in chains:
                return True
            else:
                return False

    io=PDBIO()
    io.set_structure(s)
    io.save(cle_ca_path, MySelect())
    
def generate_indexes(model, tokenizer, atom_type, mol_smiles):
    mo = ['Z','h','i','M','g','V','T','l', 'H', 'e', 'r', 'R', 'u', 't'] 
    mol_index = [id for id,i in enumerate(atom_type) if i[0] not in mo]
    tokenized_smiles = tokenizer(mol_smiles, padding=True, truncation=True, return_tensors="pt")
    subwords = tokenizer.tokenize(mol_smiles)
    with torch.no_grad():
        embeddings = model(**tokenized_smiles).last_hidden_state
    subinmol_index = [id for id,i in enumerate(subwords) if i.isalpha()]
    subinmol_words = [i for id,i in enumerate(subwords) if i.isalpha()]
    atom_ty = [i[0].upper() for i in atom_type]
    subinmol_words = [i.upper() for i in subinmol_words]  
    if atom_ty == subinmol_words:
        print("The lists are equal") 
    if atom_ty != subinmol_words:
        differing_elements = list(set(atom_ty) ^ set(subinmol_words))
        if len(atom_type) > len(subinmol_words):
            atom_ty = [i for i in atom_ty if i not in differing_elements]
            mol_index = [id for id,i in enumerate(atom_type) if i.isalpha() and i not in differing_elements and i not in mo]
        elif len(atom_type) < len(subinmol_words):
            subinmol_words = [i for i in subinmol_words if i not in differing_elements]
            subinmol_index = [id for id,i in enumerate(subwords) if i.isalpha() and i not in differing_elements]
        if len(mol_index) == len(subinmol_index):
            print('两者相等')
            if atom_ty == subinmol_words:
                print('两者一样')
            else:
                print(atom_ty)
                print(subinmol_words)
                
    return mol_index, subinmol_index, embeddings
                                                                                 
def generate_ifps_graphs_feature(kinase_pdb_path, kinase_clean_path, save_path):
    c_files = os.listdir(kinase_clean_path)
    kinase_csv = f'./pdbbind2020/pdbbind2020_dataset.csv'
    df_affinity = pd.read_csv(kinase_csv, usecols=['pdb', 'affinity'])
    m = 0
    n = 0
    i = 0
    error_pdbs = []
    error_num = []
    model1, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
    batch_converter = alphabet.get_batch_converter()
    
    model_name = "DeepChem/ChemBERTa-10M-MLM"
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    for c_fi in c_files:
        try:
            m = m + 1
            c_f = c_fi.split('_')[0]
            for row in df_affinity.iterrows():
                if str(row[1].pdb) == c_f:
                    affinity = torch.FloatTensor([row[1].affinity])
                    pdb_path = os.path.join(kinase_clean_path, c_fi)
                    pro_index = gen_seq_list(pdb_path)
                    protein, protein_graph, coords = generate_graph_feature(pdb_path)
                    protein_sequence, chain_num = generate_protein_sequence(pdb_path)
                    
                    pr_dist, pr_theta, pr_phi, pr_tau = generate_inner_coor(protein.x, protein.node_s, protein.edge_index)
                    
                    prot = [(0, protein_sequence[:1022])]
                    batch_labels, batch_strs, batch_tokens = batch_converter(prot)  #0（蛋白序号）；Fasta序列，蛋白表征
                    with torch.no_grad(): 
                        results = model1(batch_tokens, repr_layers=[33], return_contacts=True)
                    pro_token_repre = results["representations"][33]
                    
                    if pro_token_repre.shape[1] != protein.node_s.shape[0] + 2 :
                        n = n + 1
                        error_pdbs.append(c_fi)
                        error_num.append(pro_token_repre.shape[1] - protein.node_s.shape[0] - 2)
                        print(f'第{n}个 {c_fi} pdb has some problem!!!')
                    
                    k_files = os.listdir(f'{kinase_pdb_path}/{c_f}')
                    for k_fi in k_files:
                        suffix = k_fi.split('_')[-1]
                        if suffix == 'ligand.mol2':
                            mol_mol2 = os.path.join(f'{kinase_pdb_path}/{c_f}', k_fi)
                            mol_all_nodes_feature, mol_all_edges_feature, mol_all_coords_feature, mol_dgl_graph, edge_index, mol_smiles, atom_types = generate_smiles_nodes_edges_coords_graph_index_features('mol2', mol_mol2)
                            mol_dist, mol_theta, mol_phi, mol_tau = generate_inner_coor(mol_all_coords_feature, mol_all_nodes_feature, edge_index)
                            mol_index, subinmol_index, embeddings = generate_indexes(model, tokenizer, atom_types, mol_smiles)
                                
                        elif suffix == 'pocket.pdb':
                            pocket_path = os.path.join(f'{kinase_pdb_path}/{c_f}', k_fi)
                            poc_name = k_fi.split('_')[0]
                            poc_cle_path = os.path.join(f'{kinase_pdb_path}/{c_f}', f'{poc_name}_pocket_clean.pdb')
                            clean_pock(pocket_path, poc_cle_path)
                            poc_index = gen_seq_list(poc_cle_path)
                            pocket, pocket_graph, coords = generate_graph_feature(pocket_path)
                            pocket_sequence, chain_num = generate_protein_sequence(pocket_path)
                            
                            pock = [(0, pocket_sequence)]
                            batch_labels, batch_strs, pock_batch_tokens = batch_converter(pock)  #0（蛋白序号）；Fasta序列，蛋白表征
                            with torch.no_grad(): 
                                results = model1(pock_batch_tokens, repr_layers=[33], return_contacts=True)
                            poc_token_repre = results["representations"][33]
                            
                            if poc_token_repre.shape[1] != pocket.node_s.shape[0] + 2:
                                i = i + 1
                                error_pdbs.append(c_fi)
                                error_num.append(pro_token_repre.shape[1] - protein.node_s.shape[0] - 2) 
                                print(f'第{i}个 {c_fi} pocket has some problem!!!')

                            po_dist, po_theta, po_phi, po_tau = generate_inner_coor(pocket.x, pocket.node_s, pocket.edge_index)

                    indexes = [pro_index.index(x) for x in poc_index if x in pro_index]
                    mol_pocket_protein_fea = Data(com_affinity=affinity,
                                    pocinpro_index=indexes,
                                    pro_graphs=protein_graph,
                                    pro_atoms_feats_s=protein.node_s,
                                    pro_atoms_feats_v=protein.node_v,
                                    pro_coords_feats=protein.x,
                                    pro_edges_feats_s=protein.edge_s,
                                    pro_edges_feats_v=protein.edge_v,
                                    pro_edge_index=protein.edge_index,
                                    poc_graphs=pocket_graph,
                                    poc_coords_feats=pocket.x,
                                    poc_atoms_feats_s=pocket.node_s,
                                    poc_edges_feats_s=pocket.edge_s,
                                    poc_atoms_feats_v=pocket.node_v,
                                    poc_edges_feats_v=pocket.edge_v,
                                    poc_edge_index=pocket.edge_index,
                                    pro_token_repre=pro_token_repre,
                                    poc_token_repre=poc_token_repre,
                                    mol_atoms_feats=mol_all_nodes_feature, 
                                    mol_edges_feats=mol_all_edges_feature, 
                                    mol_coords_feats=mol_all_coords_feature,
                                    atoinmol_index = mol_index,
                                    subwinsmi_index = subinmol_index,
                                    mol_embedding = embeddings,
                                    mol_graphs=mol_dgl_graph, 
                                    mol_edge_index=edge_index,
                                    pro_fp=protein_sequence,
                                    poc_fp=pocket_sequence,
                                    mol_seq=mol_smiles,
                                    mol_dist=mol_dist,
                                    mol_theta=mol_theta,
                                    mol_phi=mol_phi,
                                    mol_tau=mol_tau,
                                    pr_dist=pr_dist,
                                    pr_theta=pr_theta,
                                    pr_phi=pr_phi,
                                    pr_tau=pr_tau,
                                    po_dist=po_dist,
                                    po_theta=po_theta,
                                    po_phi=po_phi,
                                    po_tau=po_tau)
                                    
                    torch.save(mol_pocket_protein_fea, f'{save_path}/{c_f}_plp.pt')
                    print(f'processing {m}_{c_fi}')
        except Exception as error:
            print(error)
            print(f'there is something wrong with {m}_{c_fi}')
            continue
           
    name  = ['pdb', 'num']
    leng = pd.DataFrame(columns=name, data=list(zip(error_pdbs,error_num)))
    leng.to_csv('./pdbbind2020/clean_pro_error.csv', encoding='utf-8')  
  
if __name__ == '__main__':
    pdbbind2020_path = './pdbbind2020/pdbbind_files'
    ifp_graphs_save_path = f'./pdbbind2020/pdbbind2020_3dgraph_esm_berta_features'
    pdbbind2020_clean_path = './pdbbind2020/clean_pdbbind2020'
    os.system(f"mkdir -p {ifp_graphs_save_path}")
    generate_ifps_graphs_feature(pdbbind2020_path, pdbbind2020_clean_path, ifp_graphs_save_path)

    