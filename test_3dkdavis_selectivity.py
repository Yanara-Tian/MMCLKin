import torch
from model import MMCLKin
import os
from sklearn.metrics import mean_squared_error
from metric import spearman
from metric import regression_metrics
import pandas as pd
import numpy as np

def comput_sds(real_results, pred_results):
    n = 0
    for i in real_results:
        if i > 6:
            n = n + 1
    real_sds = n/len(real_results)
    m = 0
    for d in pred_results:
        if d >= 6:
            m = m + 1
    pred_sds = m/len(pred_results)
    
    return n, len(real_results), real_sds, m, len(pred_results), pred_sds

d_path = './test_datasets/3dkdavis_new_drug_selectivity'
valid_datasets = os.listdir(d_path)
valid_dataset = {}
for drug in valid_datasets:
    key = drug.split('_')[2]
    if key in valid_dataset:
        valid_dataset[key].append(drug)
    else:
        valid_dataset[key] = [drug]
            
valid_dataset = list(valid_dataset.values())
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
        
path = './pkls/3dkdavis_selectivity/0.6056MMCLKin_DTI_pearson_best.pkl'

lstm_dropout = 0.2
alpha = 0.2
num_heads = 2
hidden_dim = 512
dropout_rate = 0.4
model = MMCLKin(lstm_dropout, alpha, num_heads, hidden_dim, dropout_rate, n_head=8, smile_vocab=63, local_rank=device)  
model.load_state_dict(torch.load(path)['model'], strict=True)
print('testing the selectivity performance of MMCLKin model')
model.to(device)
model.eval()
all_ytrue = torch.Tensor()
all_ypred = torch.Tensor()
valid_results = []
real_sdses = []
pred_sdses = []
sele_coif = 0
with torch.no_grad():
    for i, x_feats in enumerate(valid_dataset):
        print(f'processing the {i}th kinase inhibitor')
        total_y_true = torch.Tensor()
        total_y_pred = torch.Tensor()
        for x_fe in x_feats:
            x_feat = torch.load(f'{d_path}/{x_fe}')
            x_feat.mol_batch = torch.tensor([[0] * x_feat.mol_atoms_feats.shape[0]], dtype=torch.int64).squeeze(0)
            x_feat.poc_batch = torch.tensor([[0] * x_feat.poc_atoms_feats_s.shape[0]], dtype=torch.int64).squeeze(0)
            x_feat.pro_batch = torch.tensor([[0] * x_feat.pro_atoms_feats_s.shape[0]], dtype=torch.int64).squeeze(0)
            x_feat.pocinpro_indexes = [torch.tensor([x for x in x_feat.pocinpro_index if x <= 1022]).type(torch.long)]
            x_feat.atominmol_indexes = [torch.tensor(x_feat.atoinmol_index).type(torch.long)]
            x_feat.subwinsmi_indexes = [torch.tensor(x_feat.subwinsmi_index).type(torch.long)]
            x_feat = x_feat.to(device)
            y_pred, g_spo_att, g_spr_att, s_spo_att, s_spr_att, pg_spo_att, pg_spr_att, ps_spo_att, ps_spr_att = model(x_feat)
            y_pred = y_pred.reshape(-1)
            y_true = x_feat.com_affinity.reshape(-1)
            total_y_true = torch.cat((total_y_true, y_true.cpu()), 0)
            total_y_pred = torch.cat((total_y_pred, y_pred.cpu()), 0)
            all_ytrue = torch.cat((all_ytrue, y_true.cpu()), 0)
            all_ypred = torch.cat((all_ypred, y_pred.cpu()), 0)
            
        n, len_real_results, real_sds, m, len_pred_results, pred_sds= comput_sds(total_y_true, total_y_pred)
        real_sdses.append(real_sds)
        pred_sdses.append(pred_sds)
        s_mse = mean_squared_error(total_y_true.tolist(), total_y_pred.tolist())
        sele_coif = pred_sds*s_mse + sele_coif
        valid_results.append([x_fe, n, len_real_results, real_sds, m, len_pred_results, pred_sds, s_mse, pred_sds*s_mse])
    
    rp_pearson = np.corrcoef(real_sdses, pred_sdses)[0, 1]
    mean_sc = sele_coif/(i+1)
    results = regression_metrics(all_ytrue.tolist(), all_ypred.tolist())
    print('rp_pearson', rp_pearson, 'mean_selectivity', mean_sc, 'results', results)
    name = ['drug', 'real_posi', 'all_real', 'real_sds', 'pred_posi','all_pred','pred_sds', 'single_mse', 'selectivity_coiff']
    dfse = pd.DataFrame(columns=name, data=valid_results)
    os.makedirs('./output/3dkdavis/selectivity', exist_ok=True)
    s_path = os.path.join('./output/3dkdavis/selectivity', f'MMCLKin_DTI_mse_best44.csv')
    dfse.to_csv(s_path, encoding='utf-8')