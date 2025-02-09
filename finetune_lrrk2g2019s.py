
import os
import gc
import sys
import torch
import time
from tqdm import *
import random
import warnings
import argparse
import numpy as np
from model import MMCLKin
from loss import get_loss
from metric import regression_metrics
from torch.optim import  lr_scheduler, AdamW
from torch_geometric.data import Data
import math
import pandas as pd
from dataset_loader import Pdbbindset
from torch.nn.utils.rnn import pad_sequence

warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--EPOCHS', default=150, type=int, help='number of epoch')
parser.add_argument('--early_stop_patience', default=50, type=int, help='number of epoch')
parser.add_argument('--model_save_path', default='lrrk2_g2019s', type=str, help='number of epoch')

parser.add_argument('--node_in_dim', default=6, type=int, help='output size of model')
parser.add_argument('--node_h_dim', default=6, type=int, help='the number of num_heads')
parser.add_argument('--edge_in_dim', default=32, type=int, help='output size of model')
parser.add_argument('--edge_h_dim', default=32, type=int, help='the number of num_heads')

parser.add_argument('--model', default='MMCLKin', type=str, help='LGBIFP_CADTI, LGB_CADTI')
parser.add_argument('--lr_reduce_patience', default=50, type=int, help='the rate of dropout')
parser.add_argument('--hidden_dim', default=64, type=int, help='output size of model')
parser.add_argument('--num_heads', default=2, type=int, help='the number of num_heads')
parser.add_argument('--weight_decay', default=1e-8, type=float, help='the nomalization parameter')

parser.add_argument('--scheduler', default='CosineAnnealingWarmRestarts', type=str, help='ReduceLROnPlateau, ExponentialLR...')
parser.add_argument('--optimizer', default='AdamW', type=str, help='Adam, SGD...')
parser.add_argument('--loss', default='mse', type=str, help='ce,wce,focal,bfocal...')
parser.add_argument('--lstm_dropout', default=0.2, type=float, help='lstm_dropout')
parser.add_argument('--dropout_rate', default=0.2, type=float, help='the rate of nn dropout')
parser.add_argument('--alpha', default=0.2, type=float, help='leakyrelu')
args = parser.parse_args()

class Logger(object):
    def __init__(self, file_name="Default.log", stream=sys.stdout):
        self.terminal = stream
        self.log = open(file_name, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

#--------------------------load model and gpu-----------------------------------
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
criterion = get_loss(args.loss)

def valid_test(beta1, beta2, beta3, beta4, model, dataloader, batch_size, path, epoch, length_v, dataset):
    model.eval()
    total_loss = 0
    total_loss1 = 0
    total_loss2 = 0
    total_loss3 = 0
    total_loss4 = 0
    total_loss5 = 0
    total_y_true = torch.Tensor()
    total_y_pred = torch.Tensor()
    
    with torch.no_grad():
        c = 0
        for i, x_feats in enumerate(dataloader):
            x_feats = x_feats.to(device)
            y_pred, g_spo_att, g_spr_att, s_spo_att, s_spr_att, pg_spo_att, pg_spr_att, ps_spo_att, ps_spr_att = model(x_feats)
            
            c = c + batch_size
            y_pred = y_pred.reshape(-1)
            y_true = x_feats.y.reshape(-1)
            loss1 = criterion(y_pred, y_true)
            loss2 = gcl_loss(g_spo_att, g_spr_att)
            loss3 = gcl_loss(s_spo_att, s_spr_att)
            loss4 = gcl_loss(pg_spo_att, ps_spo_att)
            loss5 = gcl_loss(pg_spr_att, ps_spr_att)
       
            loss = loss1 + loss2*beta1 + loss3*beta2 + loss4*beta3 + loss5*beta4
            loss = loss.detach()
            total_loss += loss
            total_loss1 += loss1
            total_loss2 += loss2
            total_loss3 += loss3
            total_loss4 += loss4
            total_loss5 += loss5
            
            total_y_true = torch.cat((total_y_true, y_true.cpu()), 0)
            total_y_pred = torch.cat((total_y_pred, y_pred.cpu()), 0) 
            
            if i % 50 == 0:
                print('# [{}/{}] {}: {:.1%} total_loss={:.5f} loss1={:.5f} loss2={:.5f} loss3={:.5f} loss4={:.5f} loss5={:.5f}\n'.format(
                epoch, args.EPOCHS, dataset, c/length_v, loss, loss1, loss2, loss3, loss4, loss5))
            
    mean_loss = total_loss/len(dataloader)
    mean_loss1 = total_loss1/len(dataloader)
    mean_loss2 = total_loss2/len(dataloader)
    mean_loss3 = total_loss3/len(dataloader)
    mean_loss4 = total_loss4/len(dataloader)
    mean_loss5 = total_loss5/len(dataloader)
    
    results = regression_metrics(total_y_true, total_y_pred)
    rmse = results['mse']

    return mean_loss, mean_loss1, mean_loss2, mean_loss3, mean_loss4, mean_loss5, results, rmse, total_y_true.tolist(), total_y_pred.tolist()

def setup_seed(seed):
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True

def sim(zi, zj):
    zi = zi/torch.sqrt((zi*zi).sum(1)).unsqueeze(1)
    zj = zj/torch.sqrt((zj*zj).sum(1)).unsqueeze(1)
    return torch.mm(zi, zj.t())

def gcl_loss(z_1d, z_3d):
    gcl_tau = 0.07
    f = lambda x: torch.exp(x / gcl_tau)
    between_sim = f(sim(z_1d, z_3d))
    g_loss = -torch.log(torch.diag(between_sim) / between_sim.sum(1))-torch.log(torch.diag(between_sim) / between_sim.sum(0))
    loss = g_loss.sum()
    return loss

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
        y.append(torch.tensor(-math.log10(pt.com_affinity),dtype=torch.float32).unsqueeze(0))
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
         
def main(params):
    MODEL_train_NAME = f"{args.model}_lrrk2_g2019s_{int(time.time())}"
    sa_path = f'./output/lrrk2_g2019s/{args.model_save_path}/{MODEL_train_NAME}'
    os.system('mkdir -p {}'.format(sa_path))
    log_file_name = f"{sa_path}/{MODEL_train_NAME}.log"
    sys.stdout = Logger(log_file_name)
    sys.stderr = Logger(log_file_name)
    
    setup_seed(1234)
    train_path = './mutant/lrrk2_mw'
    train_dataset = os.listdir(train_path)
    test_path = './mutant/lrrk2_g4'
    test_dataset = os.listdir(test_path)

    print('train_dataset:', len(train_dataset))
    print('test_dataset:', len(test_dataset))
    
    train_length = len(train_dataset)
    test_length = len(test_dataset)
    train_dataset = Pdbbindset(train_dataset, train_path)
    test_dataset = Pdbbindset(test_dataset, test_path)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True, num_workers=4, pin_memory=False, collate_fn=collate)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=True, num_workers=4, pin_memory=False,collate_fn=collate)
    del train_dataset
    del test_dataset

    path = './pkls/finetune_lrrk2g2019s/MMCLKin_DTI_pearson_best.pkl'
    lstm_dropout = 0.2
    alpha = 0.2
    num_heads = 2
    hidden_dim = 512
    dropout_rate = 0.4

    model = MMCLKin(lstm_dropout, alpha, num_heads, hidden_dim, dropout_rate, n_head=8, smile_vocab=63, local_rank=device)  
    model.load_state_dict(torch.load(path)['model'], strict=True)
    model.to(device)
    #----------------------Setting the loss, optimizer, scheduler--------------------------
    param = [p for p in model.parameters() if p.requires_grad] 
    optimizer = AdamW(param, lr=params['learning_rate'], weight_decay=args.weight_decay)
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    
    print("init done")
    best_mse = 1000
    val_loss = []
    early_stop_cnt = 0
    best_epoch = 0
    best_pearson = 0          
    print(str(args))
    print('learning_rate:{0}, batch_size:{1},hidden_dim:{2}, dropout_rate:{3}, lr_reduce_rate:{4}, \
        beta1:{5},  beta2:{6},  beta3:{7}, beta4:{8}'\
        .format(params['learning_rate'],params['batch_size'], params['hidden_dim'],params['dropout_rate'],\
            params['lr_reduce_rate'],params['beta1'], params['beta2'],params['beta3'],params['beta4']))
    
    for epoch in range(args.EPOCHS):
        print(f'\n=======第 {epoch} epoch==========\n')
        start_t = time.time()
        model.train()
        losses = 0
        losses1 = 0
        losses2 = 0
        losses3 = 0
        losses4 = 0
        losses5 = 0
        
        for i, x_feats in enumerate(train_dataloader):
            x_feats = x_feats.to(device)
            output, g_spo_att, g_spr_att, s_spo_att, s_spr_att, pg_spo_att, pg_spr_att, ps_spo_att, ps_spr_att = model(x_feats)
            out = output.reshape(-1)
            y = x_feats.y.reshape(-1)
            loss1 = criterion(out, y)
            loss2 = gcl_loss(g_spo_att, g_spr_att)
            loss3 = gcl_loss(s_spo_att, s_spr_att)
            loss4 = gcl_loss(pg_spo_att, ps_spo_att)
            loss5 = gcl_loss(pg_spr_att, ps_spr_att)
            
            loss = loss1 + loss2*params['beta1'] + loss3*params['beta2'] + loss4*params['beta3'] + loss5*params['beta4'] 
        
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            loss = loss.detach()
            losses += loss.item()
            losses1 += loss1.item()
            losses2 += loss2.item()
            losses3 += loss3.item()
            losses4 += loss4.item()
            losses5 += loss5.item()
            b = (i+1) * params['batch_size']
            
            if i % 50 == 0:
                print('# [{}/{}] training {:.1%} total_loss={:.5f} loss1={:.5f} loss2={:.5f} loss3={:.5f} loss4={:.5f} loss5={:.5f}\n'.format(\
                    epoch, args.EPOCHS, b/train_length, loss, loss1, loss2, loss3, loss4, loss5), end='\r')
        
        train_loss = losses/len(train_dataloader)
        train_loss1 = losses1/len(train_dataloader)
        train_loss2 = losses2/len(train_dataloader)
        train_loss3 = losses3/len(train_dataloader)
        train_loss4 = losses4/len(train_dataloader)
        train_loss5 = losses5/len(train_dataloader)
        end_t = time.time()
        spent_t = end_t - start_t
        m, s = divmod(spent_t, 60)
        h, m = divmod(m, 60)
        print(f'{epoch} epoch is trained by taking: {"%02d:%02d:%02d" % (h, m, s)}\n')
        print('total_train_loss={:.5f} train_loss1={:.5f} train_loss2={:.5f} \
            train_loss3={:.5f} train_loss4={:.5f} train_loss5={:.5f}\n'.format(train_loss,\
            train_loss1, train_loss2, train_loss3, train_loss4, train_loss5))
                                            
        valid_loss, valid_loss1, valid_loss2, valid_loss3, valid_loss4, valid_loss5, \
            test_results, test_mse, total_y_true, total_y_pred = valid_test(params['beta1'],\
            params['beta2'], params['beta3'], params['beta4'], model, test_dataloader, \
            params['batch_size'], test_path, epoch, test_length, dataset='valid')
        
        print('total_valid_loss={:.5f} valid_loss1={:.5f} valid_loss2={:.5f} \
            valid_loss3={:.5f} valid_loss4={:.5f} valid_loss5={:.5f}\n'.format(valid_loss, \
                valid_loss1, valid_loss2, valid_loss3, valid_loss4, valid_loss5))
        print(f'valid_data_metric:{test_results}\n')
        
        val_loss.append(valid_loss.cpu())

        save_root = sa_path
        save_path = os.path.join(save_root, f'{args.model}_mse_best.pkl' ) 
        s_path = os.path.join(save_root, f'{args.model}_pearson_best.pkl' )

        if test_mse < best_mse:
            best_mse = test_mse
            best_mse_results = test_results
            save_dict = {'model':model.state_dict(), 'optim':optimizer.state_dict(), 'mse':best_mse}
            best_yt = total_y_true
            best_yp = total_y_pred
            print(f"when the mse is minimum, the true value is:{best_yt}\n")
            print(f"when the mse is minimum, the pred_value is:{best_yp}\n")
            torch.save(save_dict, save_path) 

        if test_results['pearson_value'] > best_pearson:
            early_stop_cnt = 0
            best_pearson = test_results['pearson_value']
            best_epoch = epoch
            best_results = test_results
            best_yt = total_y_true
            best_yp = total_y_pred
            save_dict = {'model':model.state_dict(), 'optim':optimizer.state_dict(), 'pearson':best_pearson}
            torch.save(save_dict, s_path)
            print(f"when the pearson is maximum, the true value is:{best_yt}\n")
            print(f"when the pearson is maximum, the pred_value is:{best_yp}\n")
        else:
            early_stop_cnt += 1
            print("early_stop_cnt", early_stop_cnt)
            
        gc.collect()
        torch.cuda.empty_cache()

        if 0 < args.early_stop_patience < early_stop_cnt:
            print(f'Early stop hitted after the epoch {epoch} training!\n')
            break
        
    print(f'using {args.model} to evaluate the {args.dataset} dataset\n')
    print(f'the valid pearson is maximum in epoch {best_epoch}\n')
    print(f"when the test mse is minimum, the best test result is:{best_mse_results}\n")
    print(f"when the pearson is maximum, the best test results is:{best_results}\n")

params1 = {"batch_size": 32,
        "learning_rate": 0.00001,
        "lr_reduce_rate": 0.1,
        "dropout_rate": 0.2,
        "hidden_dim": 512,
        "beta1": 0.3,
        "beta2": 0.1,
        "beta3": 0.6,
        "beta4": 0.1}

paramses = [params1]
for params in paramses:
    main(params)

