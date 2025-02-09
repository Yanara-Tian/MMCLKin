import pandas as pd
import os
import torch
import csv
import sys
import time
from Bio.PDB import PDBParser
from torch_geometric.data import Data
from feature_extracted import generate_graph_feature
from feature_extracted import generate_smiles_nodes_edges_coords_graph_index_features
import math
from torch_scatter import scatter_min
from transformers import AutoModel, AutoTokenizer
import esm

class Logger(object):
    def __init__(self, file_name="Default.log", stream=sys.stdout):
        self.terminal = stream
        self.log = open(file_name, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

MODEL_train_NAME = f"3dkkiba_process_{int(time.time())}"
sa_path = f'./3dkkiba/3dkkiba_process'
os.system('mkdir -p {}'.format(sa_path))
log_file_name = f"{sa_path}/{MODEL_train_NAME}.log"
sys.stdout = Logger(log_file_name)
sys.stderr = Logger(log_file_name)
    
#=============extracting the features of the drug target and pocket=======

def generate_protein_sequence(pdb_path):
    aa_codes = {'ALA':'A', 'CYS':'C', 'ASP':'D', 'GLU':'E', 'PHE':'F', 'GLY':'G', 'HIS':'H', 'LYS':'K',
     'ILE':'I', 'LEU':'L', 'MET':'M', 'ASN':'N', 'PRO':'P', 'GLN':'Q', 'ARG':'R', 'SER':'S', 'THR':'T',
     'VAL':'V', 'TYR':'Y', 'TRP':'W', 'HIE':'H'}
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

from Bio.PDB import PDBParser
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
    seq_list = [f'{res.id[1]}{res.id[2]}' for res in res_list]
    return seq_list

def generate_kin_fes(kinase_path, batch_converter, model1):
    error_pdbs = []
    pdbs_1022 = []
    error_num = []
    kinase_pdbs = os.listdir(kinase_path)
    kinase_feas = {}
    for ki in kinase_pdbs:
        try:
            dic = {}
            kina = ki.split('.')[0]
            pdb_path =os.path.join(kinase_path, ki)
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
                pdbs_1022.append(ki)
                error_num.append(pro_token_repre.shape[1] - protein.node_s.shape[0] - 2)
                if pro_token_repre.shape[1] > 1022:
                    print(f'{ki} pdb exceeds 1022!!!')
                else:
                    error_pdbs.append(ki)
                    print(f'{ki} pdb exceeds 1022!!!')
                
            kinase_fea = Data(pro_graphs=protein_graph,
                            pro_index=pro_index,
                            pro_atoms_feats_s=protein.node_s,
                            pro_atoms_feats_v=protein.node_v,
                            pro_coords_feats=protein.x,
                            pro_edges_feats_s=protein.edge_s,
                            pro_edges_feats_v=protein.edge_v,
                            pro_edge_index=protein.edge_index,
                            pro_token_repre=pro_token_repre,
                            pro_fp=protein_sequence,
                            pr_dist=pr_dist,
                            pr_theta=pr_theta,
                            pr_phi=pr_phi,
                            pr_tau=pr_tau)
            kinase_feas[kina] = kinase_fea
            
        except Exception as error:
            error_pdbs.append(ki)
            continue   
    return kinase_feas, pdbs_1022, error_pdbs

#kiba and davis
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
    if atom_ty != subinmol_words:
        differing_elements = list(set(atom_ty) ^ set(subinmol_words))
        if len(atom_type) > len(subinmol_words):
            atom_ty = [i for i in atom_ty if i not in differing_elements]
            mol_index = [id for id,i in enumerate(atom_type) if i.isalpha() and i not in differing_elements and i not in mo]

        elif len(atom_type) < len(subinmol_words):
            subinmol_words = [i for i in subinmol_words if i not in differing_elements]
            subinmol_index = [id for id,i in enumerate(subwords) if i.isalpha() and i not in differing_elements]
        print(atom_ty)
        print(subinmol_words)
        if len(mol_index) == len(subinmol_index):
            print('两者相等')
        else:
            print(f'{mol_smiles} has some problems')
    return mol_index, subinmol_index, embeddings

def gene_smi_fes(ligand_path, model, tokenizer):
    lig_mols = os.listdir(ligand_path)
    error_mol = []
    lig_feas = {}
    for sdf in lig_mols:
        try:
            mol_na = sdf.split('.')[0]
            mol_mol2 = os.path.join(ligand_path, sdf)
            mol_all_nodes_feature, mol_all_edges_feature, mol_all_coords_feature, mol_dgl_graph, edge_index, mol_smiles, atom_types = generate_smiles_nodes_edges_coords_graph_index_features('sdf', mol_mol2)
            mol_dist, mol_theta, mol_phi, mol_tau = generate_inner_coor(mol_all_coords_feature, mol_all_nodes_feature, edge_index)
            mol_index, subinmol_index, embeddings = generate_indexes(model, tokenizer, atom_types, mol_smiles)
            mol_fea = Data(mol_atoms_feats=mol_all_nodes_feature, 
                            mol_edges_feats=mol_all_edges_feature, 
                            mol_coords_feats=mol_all_coords_feature,
                            atoinmol_index = mol_index,
                            subwinsmi_index = subinmol_index,
                            mol_embedding = embeddings,
                            mol_graphs=mol_dgl_graph, 
                            mol_edge_index=edge_index,
                            mol_seq=mol_smiles,
                            mol_dist=mol_dist,
                            mol_theta=mol_theta,
                            mol_phi=mol_phi,
                            mol_tau=mol_tau) 
            lig_feas[mol_na] = mol_fea
        except Exception as error:
            error_mol.append(mol_na)
            continue
        
    return lig_feas, error_mol

def save_feas_pt(kinase_feature, pocket_feature, affinity, ligand_feature):
    pro_index = kinase_feature.pro_index
    poc_index = pocket_feature.pro_index 
    pocinpro_index = [pro_index.index(x) for x in poc_index if x in pro_index]
    mol_pocket_protein_fea = Data(com_affinity=affinity,
                            pocinpro_index=pocinpro_index,
                            pro_graphs=kinase_feature.pro_graphs,
                            pro_atoms_feats_s=kinase_feature.pro_atoms_feats_s,
                            pro_atoms_feats_v=kinase_feature.pro_atoms_feats_v,
                            pro_coords_feats=kinase_feature.pro_coords_feats,
                            pro_edges_feats_s=kinase_feature.pro_edges_feats_s,
                            pro_edges_feats_v=kinase_feature.pro_edges_feats_v,
                            pro_edge_index=kinase_feature.pro_edge_index,
                            pro_token_repre=kinase_feature.pro_token_repre,
                            pro_fp=kinase_feature.pro_fp,
                            pr_dist=kinase_feature.pr_dist,
                            pr_theta=kinase_feature.pr_theta,
                            pr_phi=kinase_feature.pr_phi,
                            pr_tau=kinase_feature.pr_tau,
                            
                            poc_graphs=pocket_feature.pro_graphs,
                            poc_coords_feats=pocket_feature.pro_coords_feats,
                            poc_atoms_feats_s=pocket_feature.pro_atoms_feats_s,
                            poc_edges_feats_s=pocket_feature.pro_edges_feats_s,
                            poc_atoms_feats_v=pocket_feature.pro_atoms_feats_v,
                            poc_edges_feats_v=pocket_feature.pro_edges_feats_v,
                            poc_edge_index=pocket_feature.pro_edge_index,
                            poc_token_repre=pocket_feature.pro_token_repre,
                            poc_fp=pocket_feature.pro_fp,
                            po_dist=pocket_feature.pr_dist,
                            po_theta=pocket_feature.pr_theta,
                            po_phi=pocket_feature.pr_phi,
                            po_tau=pocket_feature.pr_tau,
                            
                            mol_atoms_feats=ligand_feature.mol_atoms_feats, 
                            mol_edges_feats=ligand_feature.mol_edges_feats, 
                            mol_coords_feats=ligand_feature.mol_coords_feats,
                            atoinmol_index=ligand_feature.atoinmol_index,
                            subwinsmi_index=ligand_feature.subwinsmi_index,
                            mol_embedding=ligand_feature.mol_embedding,
                            mol_graphs=ligand_feature.mol_graphs, 
                            mol_edge_index=ligand_feature.mol_edge_index,
                            mol_seq=ligand_feature.mol_seq,
                            mol_dist=ligand_feature.mol_dist,
                            mol_theta=ligand_feature.mol_theta,
                            mol_phi=ligand_feature.mol_phi,
                            mol_tau=ligand_feature.mol_tau)

    return mol_pocket_protein_fea

def extract_3dgs_features(process_path, ligand_path, kinase_path, pocket_path, sa_path, save_path, da_set):
    # model1, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
    # batch_converter = alphabet.get_batch_converter()
    # kinase_fes, kina_1022, error_kina = generate_kin_fes(kinase_path, batch_converter, model1)
    # print(f'there are {len(kina_1022)} kinase exceeding 1022, they are {kina_1022}')
    # print(f'there are {len(error_kina)} error kinase, they are {error_kina}')
    # torch.save(kinase_fes, f'{sa_path}/3dkkiba_allkinase.pt')

    # pocket_fes, pock_1022, error_pock = generate_kin_fes(pocket_path, batch_converter, model1)
    # print(f'there are {len(pock_1022)} pockets exceeding 1022, they are {pock_1022}')
    # print(f'there are {len(error_pock)} error pockets, they are {error_pock}')
    # torch.save(pocket_fes, f'{sa_path}/3dkkiba_allpockets.pt')

    # model_name = "DeepChem/ChemBERTa-10M-MLM"
    # model = AutoModel.from_pretrained(model_name)
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # lig_fes, error_mol = gene_smi_fes(ligand_path, model, tokenizer)
    # print(f'there are {len(error_mol)} error ligands, they are {error_mol}')
    # torch.save(lig_fes, f'{sa_path}/3dkkiba_allligand.pt')
    kinase_fes = torch.load(f'{sa_path}/3dkkiba_allkinase.pt')
    pocket_fes = torch.load(f'{sa_path}/3dkkiba_allpockets.pt')
    lig_fes = torch.load(f'{sa_path}/3dkkiba_allligand.pt')
    data_path = open(process_path)
    df = pd.read_csv(process_path)
    error_com = []
    compounds = df['com_name']
    sdfs = list(lig_fes.keys())
    for i in compounds:
        if i not in sdfs:
            error_com.append(i)
            print(i)
    pros_uni = csv.reader(data_path)
    next(pros_uni)
    m = 0
    i = -1
    kinase_pdbs = os.listdir(kinase_path)
    for row in pros_uni:
        i = i + 1
        lig_name = row[1]
        drug_id = row[7]
        kinase_name = row[5]
        affinity = row[4]
        protein_id = row[6]
        affinity = torch.tensor(float(affinity))
        if i % 100 == 0:
            print(f'processing {i} system')
        if lig_name not in error_com:
            if f'{kinase_name}_kinase.pdb' in kinase_pdbs:
                kinase_feature = kinase_fes[f'{kinase_name}_kinase']
                pocket_feature = pocket_fes[f'{kinase_name}_kinase_pocket']
                ligand_feature = lig_fes[f'{lig_name}']
                mol_pocket_protein_fea = save_feas_pt(kinase_feature, pocket_feature, affinity, ligand_feature)
                torch.save(mol_pocket_protein_fea, f'{save_path}/{da_set}_{i}_{drug_id}_{lig_name}_{protein_id}_{kinase_name}_plp.pt')

if __name__=='__main__':
    save_path = './3dkkiba/3dkkiba_gra_seq_pts'
    os.makedirs(save_path, exist_ok = True)
    process_path = './3dkkiba/new_3dkkiba_overall.csv'
    da_set = '3dkkiba'
    sa_path = './3dkkiba'
    ligands_sdf_path = './3dkkiba/ligand_sdfs'
    kiba_kinase_path = './3dkkiba/3dkkiba_kinase_pdbs'
    pockets_path = './3dkkiba/pockets'
    extract_3dgs_features(process_path, ligands_sdf_path, kiba_kinase_path, pockets_path, sa_path, save_path, da_set)