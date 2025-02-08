import os
import esm
import torch
import pandas as pd
import math
from torch_scatter import scatter_min
from transformers import AutoModel, AutoTokenizer
from process_3dkdavis import gene_smi_fes, generate_kin_fes, save_feas_pt
from Bio.PDB import PDBParser
from feature_extracted import generate_graph_feature
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

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

def generate_kin_fes(kinase_path, batch_converter, model1):
    error_pdbs = []
    pdbs_1022 = []
    error_num = []
    pdb_path =kinase_path
    pro_index = gen_seq_list(pdb_path)
    protein, protein_graph, coords = generate_graph_feature(pdb_path)
    protein_sequence, chain_num = generate_protein_sequence(pdb_path)
    pr_dist, pr_theta, pr_phi, pr_tau = generate_inner_coor(protein.x, protein.node_s, protein.edge_index)
    prot = [(0, protein_sequence[:1022])]
    batch_labels, batch_strs, batch_tokens = batch_converter(prot)  #0（蛋白序号）；Fasta序列，蛋白表征
    with torch.no_grad(): 
        results = model1(batch_tokens, repr_layers=[33], return_contacts=True)
    pro_token_repre = results["representations"][33]
    
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
            
    return kinase_fea

# ligand_path = '/raid/source_tyn/pkidti/pdbfile/cases/mutant/wild_lrrk2_sdfs'
# model_name = "DeepChem/ChemBERTa-10M-MLM"
# model = AutoModel.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# lig_fes, error_mol = gene_smi_fes(ligand_path, model, tokenizer)
# torch.save(lig_fes, './cases/mutant/wild_lrrk2_2633_ligand_plp.pt')

# ligand_path = '/raid/source_tyn/pkidti/pdbfile/cases/mutant/gxq_8'
# model_name = "DeepChem/ChemBERTa-10M-MLM"
# model = AutoModel.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# lig_fes, error_mol = gene_smi_fes(ligand_path, model, tokenizer)
# torch.save(lig_fes, './cases/mutant/lrrk2_gxq_8_ligand_plp.pt')

# ligand_path = '/raid/source_tyn/pkidti/pdbfile/cases/mutant/g2019s_lrrk2_sdfs'
# model_name = "DeepChem/ChemBERTa-10M-MLM"
# model = AutoModel.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# lig_fes, error_mol = gene_smi_fes(ligand_path, model, tokenizer)
# torch.save(lig_fes, './cases/mutant/g2019s_lrrk2_979_ligand_plp.pt')

# kinase_path = './cases/mutant/8fo7_kinase.pdb'
# pocket_path = './cases/mutant/8fo7_pocket_12.pdb'
# model1, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
# batch_converter = alphabet.get_batch_converter()
# kinase_fes = generate_kin_fes(kinase_path, batch_converter, model1)
# pocket_fes = generate_kin_fes(pocket_path, batch_converter, model1)
# torch.save(kinase_fes, './cases/mutant/8fo7_kinase_plp.pt')
# torch.save(pocket_fes, './cases/mutant/8fo7_pocket_12_plp.pt')

# kinase2_path = './cases/mutant/8tzc_all_kinase.pdb'
# pocket2_path = './cases/mutant/8tzc_all_pocket_12.pdb'
# kinase_fes = generate_kin_fes(kinase2_path, batch_converter, model1)
# pocket_fes = generate_kin_fes(pocket2_path, batch_converter, model1)
# torch.save(kinase_fes, './cases/mutant/8tzc_kinase_plp.pt')
# torch.save(pocket_fes, './cases/mutant/8tzc_pocket_12_plp.pt')

# root_path = './cases/mutant'
# path1 = './cases/mutant/WILD_LRRK2_IC50.csv'
# path2 = './cases/mutant/G2019S_LRRK2_IC50.csv'
# lig_fes = {}
# lig_fes.update(torch.load(f'{root_path}/wild_lrrk2_2633_ligand_plp.pt'))
# lig_fes.update(torch.load(f'{root_path}/g2019s_lrrk2_979_ligand_plp.pt'))
# lig_keys = lig_fes.keys()
# for i, na in enumerate(lig_keys):
#     all_feas = {}
#     lig_feas = lig_fes[na]
#     if na.split('_')[0] == 'wild':
#         kin_fes = torch.load(f'{root_path}/8fo7_kinase_plp.pt')
#         poc_fes = torch.load(f'{root_path}/8fo7_pocket_12_plp.pt')
#         df1 = pd.read_csv(path1,header=0,index_col=0)
#         affinity = df1['IC50 (nM)'][na]
#         all_feas = save_feas_pt(kin_fes, poc_fes, affinity, lig_feas)
#         torch.save(all_feas, f'./cases/mutant/lrrk2_wild/{na}_plp.pt')
#     elif na.split('_')[0] == 'g2019s':
#         kin_fes = torch.load(f'{root_path}/8tzc_kinase_plp.pt')
#         poc_fes = torch.load(f'{root_path}/8tzc_pocket_12_plp.pt')
#         df2 = pd.read_csv(path2,header=0,index_col=0)
#         affinity = df2['IC50 (nM)'][na]
#         all_feas = save_feas_pt(kin_fes, poc_fes, affinity, lig_feas)
#         torch.save(all_feas, f'./cases/mutant/lrrk2_g2019s/{na}_plp.pt')

#### load finetune datasets extract features
# root_path = './cases/mutant'
# lig_fes = torch.load(f'{root_path}/lrrk2_gxq_8_ligand_plp.pt')
# path1 = './cases/mutant/gxq_8.csv'
# lig_keys = lig_fes.keys()
# for i, na in enumerate(lig_keys):
#     if '_' not in na:
#         all_feas = {}
#         lig_feas = lig_fes[na]
#         kin_fes = torch.load(f'{root_path}/8fo7_kinase_plp.pt')
#         poc_fes = torch.load(f'{root_path}/8fo7_pocket_12_plp.pt')
#         df1 = pd.read_csv(path1, sep=';')
#         index = df1['Compound ID'].tolist().index(na)
#         affinity = df1['WTIC50(nM)'][index]
#         all_feas = save_feas_pt(kin_fes, poc_fes, affinity, lig_feas)
#         torch.save(all_feas, f'./cases/mutant/lrrk2_gxq8/wild_{na}_plp.pt')
            
#         kin_fes = torch.load(f'{root_path}/8tzc_kinase_plp.pt')
#         poc_fes = torch.load(f'{root_path}/8tzc_pocket_12_plp.pt')
#         index = df1['Compound ID'].tolist().index(na)
#         affinity = df1['G2019SIC50(nM)'][index]
#         all_feas = save_feas_pt(kin_fes, poc_fes, affinity, lig_feas)
#         torch.save(all_feas, f'./cases/mutant/lrrk2_gxq8/g2019s_{na}_plp.pt')


def get_molecular_framework(smiles):
    mol = Chem.MolFromSmiles(smiles)
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold)

filename = './cases/mutant/lrrk2_wild_ic50_clean.csv'
odata = pd.read_csv(filename)

data = odata[odata['IC50 (nM)'] < 5500]
data['Scaffold'] = data['Ligand SMILES'].apply(get_molecular_framework)

# 按Scaffold分组
groups = data.groupby('Scaffold')

# 随机选取每个骨架的样本
selected_data = []

for name, group in groups:
    sample_size = max(1, int(len(group) * 979 / len(data)))
    sampled_group = group.sample(n=1, random_state=42)
    selected_data.append(sampled_group)

final_data = pd.concat(selected_data)

# 如果抽样的数据太多或太少，可以再次调整
if len(final_data) > 979:
    final_data = final_data.sample(n=979, random_state=42)
elif len(final_data) < 979:
    remaining_samples = data[~data.index.isin(final_data.index)]
    additional_needed = 979 - len(final_data)
    additional_samples = remaining_samples.sample(n=additional_needed, random_state=42)
    final_data = pd.concat([final_data, additional_samples])

# 验证结果大小
print(f"最终数据集的大小: {len(final_data)}")

# 保存新的数据集
final_data.to_csv('./cases/mutant/new_lrrk2_wild_ic50_clean.csv', index=False)

import shutil
filename = './cases/mutant/new_lrrk2_wild_ic50_clean.csv'
odata = pd.read_csv(filename)
data = odata['BindingDB MonomerID'].tolist()
pts = os.listdir('./cases/mutant/lrrk2_wild')
for pt in pts:
    if pt[:4] == 'wild':
        lig = int(pt.split('_')[1])
        if lig in data:
            ptp = f'./cases/mutant/lrrk2_wild/wild_{lig}_plp.pt'
            shutil.move(ptp, './cases/mutant/lrrk2_mw')


