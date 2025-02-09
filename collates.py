
import os
import time
import torch
import esm
import pandas as pd
from transformers import AutoModel, AutoTokenizer
from model import MMCLKin
from torch_geometric.data import Data
from process_3dkdavis import generate_inner_coor, generate_protein_sequence, generate_graph_feature, gen_seq_list, gene_smi_fes

def collate(lig, kin, poc):
    pocinpro_index = [kin.pro_index.index(x) for x in poc.pro_index if x in kin.pro_index]
    poc_batch = torch.tensor([[0] * poc.pro_atoms_feats_s.shape[0]], dtype=torch.int64)
    pocinpro_indexes = [torch.tensor([x for x in pocinpro_index if x <= 1022]).type(torch.long)]
    mol_batch = torch.tensor([[0] * lig.mol_atoms_feats.shape[0]], dtype=torch.int64)
    pro_batch = torch.tensor([[0] * kin.pro_atoms_feats_s.shape[0]], dtype=torch.int64)
    
    x_feats = Data(poc_batch = poc_batch,
                   mol_batch = mol_batch,
                   pro_batch = pro_batch,
                   pocinpro_indexes = pocinpro_indexes,
                   po_dist = poc.pr_dist,
                   po_theta = poc.pr_theta,
                   po_phi = poc.pr_phi,
                   po_tau = poc.pr_tau,
                   mol_dist = lig.mol_dist,
                   mol_theta = lig.mol_theta,
                   mol_phi = lig.mol_phi,
                   mol_tau = lig.mol_tau,
                   pr_dist = kin.pr_dist,
                   pr_theta = kin.pr_theta,
                   pr_phi = kin.pr_phi,
                   pr_tau = kin.pr_tau,
                   
                   mol_atoms_feats = lig.mol_atoms_feats,
                   mol_edges_feats = lig.mol_edges_feats,
                   mol_coords_feats = lig.mol_coords_feats,
                   mol_edge_index = lig.mol_edge_index.type(torch.long),
                   mol_embedding = lig.mol_embedding,
                   atominmol_indexes = [torch.tensor(lig.atoinmol_index).type(torch.long)],
                   subwinsmi_indexes = [torch.tensor(lig.subwinsmi_index).type(torch.long)],
                   mol_smiles = lig.mol_seq,
                   poc_coords_feats = poc.pro_coords_feats,
                   poc_atoms_feats_s = poc.pro_atoms_feats_s,
                   poc_edges_feats_s = poc.pro_edges_feats_s,
                   poc_atoms_feats_v = poc.pro_atoms_feats_v,
                   poc_edges_feats_v = poc.pro_edges_feats_v,
                   poc_edge_index = poc.pro_edge_index.type(torch.long),
                   poc_token_repre = poc.pro_token_repre,
                   pro_atoms_feats_s = kin.pro_atoms_feats_s,
                   pro_atoms_feats_v = kin.pro_atoms_feats_v,
                   pro_coords_feats = kin.pro_coords_feats,
                   pro_edges_feats_s = kin.pro_edges_feats_s,
                   pro_edges_feats_v = kin.pro_edges_feats_v,
                   pro_edge_index = kin.pro_edge_index.type(torch.long),
                   pro_token_repre = kin.pro_token_repre)
    
    return x_feats

def generate_kin_fes(ki, batch_converter, model1):
    error_pdbs = []
    pdbs_1022 = []
    error_num = []
    kinase_feas = {}
    kina = ki.split('.')[1]
    pdb_path = ki
    pro_index = gen_seq_list(pdb_path)
    protein, protein_graph, coords = generate_graph_feature(pdb_path)
    protein_sequence, chain_num = generate_protein_sequence(pdb_path)
    
    pr_dist, pr_theta, pr_phi, pr_tau = generate_inner_coor(protein.x, protein.node_s, protein.edge_index)
    prot = [(0, protein_sequence[:1022])]
    batch_labels, batch_strs, batch_tokens = batch_converter(prot)  #0（蛋白序号）；Fasta序列，蛋白表征
    with torch.no_grad():
        results = model1(batch_tokens, repr_layers=[33], return_contacts=True)
    pro_token_repre = results["representations"][33]
    n = 0
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
            
    return kinase_feas, pdbs_1022, error_pdbs