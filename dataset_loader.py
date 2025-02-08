import torch 
from torch_geometric.data import Data, Dataset
from torch.nn.utils.rnn import pad_sequence
 
class Pdbbindset(Dataset):
    def __init__(self, dataset, data_path):
        super(Pdbbindset, self).__init__()
        self.pts = []
        for i, data in enumerate(dataset):
            da = f'{data_path}/{data}'
            self.pts.append(da)
            
    def __len__(self):
        return len(self.pts)
           
    def __getitem__(self, idx):
        pt = self.pts[idx]
        return pt                                
                                    
def collate(args):
    pocinpro_index = torch.FloatTensor()
    atominmol_index = torch.FloatTensor()
    subwinsmi_index = torch.FloatTensor()
    mol_embedding = torch.FloatTensor()
    poc_token_repre = torch.FloatTensor()
    pro_token_repre = torch.FloatTensor()
    po_dist = torch.FloatTensor()
    po_theta = torch.FloatTensor()
    po_phi = torch.FloatTensor()
    po_tau = torch.FloatTensor()
    mol_dist = torch.FloatTensor()
    mol_theta = torch.FloatTensor()
    mol_phi = torch.FloatTensor()
    mol_tau = torch.FloatTensor()
    pr_dist = torch.FloatTensor()
    pr_theta = torch.FloatTensor()
    pr_phi = torch.FloatTensor()
    pr_tau = torch.FloatTensor()
    mol_atoms_feats = torch.FloatTensor()
    mol_edges_feats = torch.FloatTensor()
    mol_coords_feats = torch.FloatTensor()
    poc_coords_feats = torch.FloatTensor()
    poc_atoms_feats_s = torch.FloatTensor()
    poc_edges_feats_s = torch.FloatTensor()
    poc_atoms_feats_v = torch.FloatTensor()
    poc_edges_feats_v = torch.FloatTensor()
    pro_atoms_feats_s = torch.FloatTensor()
    pro_atoms_feats_v = torch.FloatTensor()
    pro_coords_feats = torch.FloatTensor()
    pro_edges_feats_s = torch.FloatTensor()
    pro_edges_feats_v = torch.FloatTensor()
    pro_edge_index = torch.FloatTensor() 
    poc_edge_index = torch.FloatTensor()
    mol_edge_index = torch.FloatTensor()
    
    smiles_len = []
    poc_token_len = []
    pro_token_len = []
    
    mol_size = []
    poc_size = []
    pro_size = []
    y = []
    
    subw_len = 0
    pro_aa_len = 0
    poc_aa_len = 0
    atom_len = 0
    all_batch = []
    atominmol_indexes = []
    pocinpro_indexes = []
    subwinsmi_indexes = []
    
    for i, pts in enumerate(args):
        pt = torch.load(pts)
        atominmol_indexes.append(torch.tensor(pt.atoinmol_index).type(torch.long))
        pocinpro_indexes.append(torch.tensor([x for x in pt.pocinpro_index if x <= 1022]).type(torch.long))
        subwinsmi_indexes.append(torch.tensor(pt.subwinsmi_index).type(torch.long))
        po_dist = torch.cat((po_dist, pt.po_dist), 0)
        po_theta = torch.cat((po_theta, pt.po_theta), 0)
        po_phi = torch.cat((po_phi, pt.po_phi), 0)
        po_tau = torch.cat((po_tau, pt.po_tau), 0)
        mol_dist = torch.cat((mol_dist, pt.mol_dist), 0)
        mol_theta = torch.cat((mol_theta, pt.mol_theta), 0)
        mol_phi = torch.cat((mol_phi, pt.mol_phi), 0)
        mol_tau = torch.cat((mol_tau, pt.mol_tau), 0)
        pr_dist = torch.cat((pr_dist, pt.pr_dist), 0)
        pr_theta = torch.cat((pr_theta, pt.pr_theta), 0)
        pr_phi = torch.cat((pr_phi, pt.pr_phi), 0)
        pr_tau = torch.cat((pr_tau, pt.pr_tau), 0)
        mol_atoms_feats = torch.cat((mol_atoms_feats, pt.mol_atoms_feats), 0)
        mol_edges_feats = torch.cat((mol_edges_feats, pt.mol_edges_feats), 0)
        mol_coords_feats = torch.cat((mol_coords_feats, pt.mol_coords_feats), 0)
        poc_coords_feats = torch.cat((poc_coords_feats, pt.poc_coords_feats), 0)
        poc_atoms_feats_s = torch.cat((poc_atoms_feats_s, pt.poc_atoms_feats_s), 0)
        poc_edges_feats_s = torch.cat((poc_edges_feats_s, pt.poc_edges_feats_s), 0)
        poc_atoms_feats_v = torch.cat((poc_atoms_feats_v, pt.poc_atoms_feats_v), 0)
        poc_edges_feats_v = torch.cat((poc_edges_feats_v, pt.poc_edges_feats_v), 0)
        pro_atoms_feats_s = torch.cat((pro_atoms_feats_s, pt.pro_atoms_feats_s), 0)
        pro_atoms_feats_v = torch.cat((pro_atoms_feats_v, pt.pro_atoms_feats_v), 0)
        pro_coords_feats = torch.cat((pro_coords_feats, pt.pro_coords_feats), 0)
        pro_edges_feats_s = torch.cat((pro_edges_feats_s, pt.pro_edges_feats_s), 0)
        pro_edges_feats_v = torch.cat((pro_edges_feats_v, pt.pro_edges_feats_v), 0)
        
        pro_edge_index = torch.cat((pro_edge_index, pt.pro_edge_index + pro_aa_len), 1)
        pocinpro_index = torch.cat((pocinpro_index, torch.tensor(pt.pocinpro_index).unsqueeze(0) + pro_aa_len), 1)
        pro_aa_len = pro_aa_len + pt.pro_atoms_feats_s.shape[0]
        poc_edge_index = torch.cat((poc_edge_index, pt.poc_edge_index + poc_aa_len), 1)
        poc_aa_len = poc_aa_len + pt.poc_atoms_feats_s.shape[0]
        mol_edge_index = torch.cat((mol_edge_index, pt.mol_edge_index + atom_len), 1) 
        atominmol_index = torch.cat((atominmol_index, torch.tensor(pt.atoinmol_index).unsqueeze(0) + atom_len), 1)
        atom_len = atom_len + pt.mol_atoms_feats.shape[0]
        mol_size.extend([i] * pt.mol_atoms_feats.shape[0])
        poc_size.extend([i] * pt.poc_atoms_feats_s.shape[0])
        pro_size.extend([i] * pt.pro_atoms_feats_s.shape[0])
        
        subwinsmi_index = torch.cat((subwinsmi_index, torch.tensor(pt.subwinsmi_index).unsqueeze(0) + subw_len), 1)
        subw_len = subw_len + pt.mol_embedding.shape[1] - 2
        y.append(pt.com_affinity.unsqueeze(0))
        # y.append(pt.com_affinity)
        smiles_len.append(pt.mol_embedding.squeeze(0))
        poc_token_len.append(pt.poc_token_repre.squeeze(0))
        pro_token_len.append(pt.pro_token_repre.squeeze(0))
        
    mol_batch = torch.tensor(mol_size, dtype=torch.int64)
    poc_batch = torch.tensor(poc_size, dtype=torch.int64)
    pro_batch = torch.tensor(pro_size, dtype=torch.int64)
    mol_embedding = pad_sequence(smiles_len).permute(1,0,2)
    poc_token_repre = pad_sequence(poc_token_len).permute(1,0,2)
    pro_token_repre = pad_sequence(pro_token_len).permute(1,0,2)
    y = torch.cat(y, dim=0)
    x_feats = Data(y = y,
                po_dist = po_dist,
                po_theta = po_theta,
                po_phi = po_phi,
                po_tau = po_tau,
                mol_dist = mol_dist,
                mol_theta = mol_theta,
                mol_phi = mol_phi,
                mol_tau = mol_tau,
                pr_dist = pr_dist,
                pr_theta = pr_theta,
                pr_phi = pr_phi,
                pr_tau = pr_tau,
                mol_atoms_feats = mol_atoms_feats,
                mol_edges_feats = mol_edges_feats,
                mol_coords_feats = mol_coords_feats,
                poc_coords_feats = poc_coords_feats,
                poc_atoms_feats_s = poc_atoms_feats_s,
                poc_edges_feats_s = poc_edges_feats_s,
                poc_atoms_feats_v = poc_atoms_feats_v,
                poc_edges_feats_v = poc_edges_feats_v,
                pro_atoms_feats_s = pro_atoms_feats_s,
                pro_atoms_feats_v = pro_atoms_feats_v,
                pro_coords_feats = pro_coords_feats,
                pro_edges_feats_s = pro_edges_feats_s,
                pro_edges_feats_v = pro_edges_feats_v,
                pro_edge_index = pro_edge_index.type(torch.long),
                poc_edge_index = poc_edge_index.type(torch.long),
                mol_edge_index = mol_edge_index.type(torch.long),
                mol_embedding = mol_embedding,
                poc_token_repre = poc_token_repre,
                pro_token_repre = pro_token_repre,
                mol_batch = mol_batch,
                poc_batch = poc_batch,
                pro_batch = pro_batch,
                atominmol_indexes = atominmol_indexes,
                pocinpro_indexes = pocinpro_indexes,
                subwinsmi_indexes = subwinsmi_indexes,
                pocinpro_index = pocinpro_index.type(torch.long),
                atominmol_index = atominmol_index.type(torch.long),
                subwinsmi_index = subwinsmi_index.type(torch.long))
    
    return x_feats
