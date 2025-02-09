
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
from torch.optim import Adam, SGD, lr_scheduler, AdamW
from dataset_loader import Pdbbindset, collate

warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--dataset', type=str, default='pdbbind2020', help='tox21, lipophilicity')
parser.add_argument('--EPOCHS', default=300, type=int, help='number of epoch')
parser.add_argument('--early_stop_patience', default=100, type=int, help='number of epoch')
parser.add_argument('--model_save_path', default='pdbbind_0000', type=str, help='number of epoch')
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
                print('# [{}/{}] {}: {:.1%} total_loss={:.5f} loss1={:.5f} loss2={:.5f} \
                    loss3={:.5f} loss4={:.5f} loss5={:.5f}\n'.format(
                    epoch, args.EPOCHS, dataset, c/length_v, loss, loss1, loss2, \
                    loss3, loss4, loss5))
            
    mean_loss = total_loss/len(dataloader)
    mean_loss1 = total_loss1/len(dataloader)
    mean_loss2 = total_loss2/len(dataloader)
    mean_loss3 = total_loss3/len(dataloader)
    mean_loss4 = total_loss4/len(dataloader)
    mean_loss5 = total_loss5/len(dataloader)
    
    results = regression_metrics(total_y_true, total_y_pred)
    rmse = results['rmse']

    return mean_loss, mean_loss1, mean_loss2, mean_loss3, mean_loss4, mean_loss5, results, rmse

def read_ref(file):
    ref = open(file, 'r')
    ref_cons = ref.readlines()
    ref_set = []
    for con in ref_cons:
        if con.split()[0] != "#":
            ref_set.append(f'{con.split()[0]}_plp.pt')
    return ref_set

def check_exist(ref_sets, all_sets):
    for fi in ref_sets:
        if fi not in all_sets:
            ref_sets.remove(fi)

    return ref_sets

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

def cosine_similarity_loss(u, v, batch):
    loss = 1 - np.sum(np.dot(u, v) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1))) / batch
    return loss

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

def main(params):
    MODEL_train_NAME = f"{args.model}_{args.dataset}_{int(time.time())}"
    sa_path = f'./output/{args.dataset}/{args.model_save_path}{MODEL_train_NAME}'
    os.system('mkdir -p {}'.format(sa_path))
    log_file_name = f"{sa_path}/{MODEL_train_NAME}.log"
    sys.stdout = Logger(log_file_name)
    sys.stderr = Logger(log_file_name)
    
    print(f'using {args.model} to evaluate the {args.dataset} dataset\n')
    path = './pdbbind2020/pdbbind2020_3dgraph_esm_berta_features'
    datasets = os.listdir(path)
    print('datasets:', len(datasets))
    setup_seed(1234)
    
    refine_txt = './pdbbind2020/INDEX_refined_set.2020'
    core_txt = './pdbbind2020/CoreSet.dat'
    ref_sets = read_ref(refine_txt)

    ref_sets = check_exist(ref_sets, datasets)
    ref_sets = check_exist(ref_sets, datasets)
    setup_seed(1234)
    random.seed(1234)
    valid_dataset = random.sample(ref_sets, 500)
    test_dataset = read_ref(core_txt)
    test_dataset = check_exist(test_dataset, datasets)
    train_dataset = [i for i in datasets if i not in valid_dataset and i not in test_dataset]
    
    train_path = valid_path = test_path = path
    train_length = len(train_dataset)
    valid_length = len(valid_dataset)
    test_length = len(test_dataset)
    train_dataset = Pdbbindset(train_dataset, train_path)
    valid_dataset = Pdbbindset(valid_dataset, valid_path)
    test_dataset = Pdbbindset(test_dataset, test_path)
    
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True, num_workers=4, collate_fn=collate)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=params['batch_size'], shuffle=True, num_workers=4, collate_fn=collate)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=True, num_workers=4, collate_fn=collate)

    print('train_dataset:', len(train_dataset))
    print('valid_dataset:', len(valid_dataset))
    print('test_dataset:', len(test_dataset))
    del train_dataset
    del valid_dataset
    
    model = MMCLKin(args.lstm_dropout, args.alpha, args.num_heads, hidden_dim=params['hidden_dim'], dropout_rate=params['dropout_rate'], n_head=8, smile_vocab=63, local_rank=device)
    model.to(device)
    
    #----------------------Setting the loss, optimizer, scheduler--------------------------
    param = [p for p in model.parameters() if p.requires_grad] 
    if args.optimizer == 'Adam':
        optimizer = Adam(param, lr=params['learning_rate'], weight_decay=args.weight_decay)
    elif args.optimizer == 'SGD':
        optimizer = SGD(param, lr=params['learning_rate'], weight_decay=args.weight_decay)
    elif args.optimizer == 'AdamW':
        optimizer = AdamW(param, lr=params['learning_rate'], weight_decay=args.weight_decay)

    if args.scheduler == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=params['lr_reduce_rate'], patience=args.lr_reduce_patience, min_lr=1e-6)
    elif args.scheduler == 'ExponentialLR':
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.90)
    elif args.scheduler == 'CosineAnnealingWarmRestarts':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
        
    best_rmse = 10
    best_pearson=0
    val_loss = []
    early_stop_cnt = 0
    best_epoch = 0
                   
    print(str(args))
    print('learning_rate:{0}, batch_size:{1},hidden_dim:{2}, dropout_rate:{3}, lr_reduce_rate:{4}, beta1:{5},  beta2:{6},  beta3:{7}, beta4:{8}'\
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
                print('# [{}/{}] training {:.1%} total_loss={:.5f} loss1={:.5f} loss2={:.5f} loss3={:.5f} loss4={:.5f} loss5={:.5f}\n'.format(epoch, args.EPOCHS, \
                        b/train_length, loss, loss1, loss2, loss3, loss4, loss5), end='\r')
        
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
        print('total_train_loss={:.5f} train_loss1={:.5f} train_loss2={:.5f} train_loss3={:.5f} train_loss4={:.5f} \
              train_loss5={:.5f}\n'.format(train_loss,train_loss1, train_loss2, train_loss3, train_loss4, train_loss5))
                                            
        valid_loss, valid_loss1, valid_loss2, valid_loss3, valid_loss4, valid_loss5, valid_results, valid_rmse = valid_test(params['beta1'],\
            params['beta2'], params['beta3'], params['beta4'], model, valid_dataloader, \
            params['batch_size'], valid_path, epoch, valid_length, dataset='valid')
            
        print('total_valid_loss={:.5f} valid_loss1={:.5f} valid_loss2={:.5f} valid_loss3={:.5f} valid_loss4={:.5f} \
            valid_loss5={:.5f}\n'.format(valid_loss, valid_loss1, valid_loss2, valid_loss3, valid_loss4, valid_loss5))
        print(f'valid_data_metric:{valid_results}\n')

        test_loss, test_loss1, test_loss2, test_loss3, test_loss4, test_loss5, test_results, \
            test_rmse = valid_test(params['beta1'],params['beta2'], params['beta3'], params['beta4'],\
                model, test_dataloader, params['batch_size'], test_path, epoch, test_length,\
                    dataset='test')   
        
        print(f'test_loss\n:{round(float(test_loss), 4)}')
        print(f'test_data_metric\n:{test_results}')
        val_loss.append(valid_loss.cpu())

        save_root = sa_path
        save_path = os.path.join(save_root, f'{args.model}_pearson_best.pkl' ) 
        s_path = os.path.join(save_root, f'{args.model}_rmse_best.pkl' )
        
        if valid_results['pearson_value'] > best_pearson:
            early_stop_cnt = 0
            best_pearson = valid_results['pearson_value']
            best_vrm_tres = test_results
            best_epoch = epoch
            save_dict = {'model':model.state_dict(), 'optim':optimizer.state_dict(), 'rmse':best_pearson}
            torch.save(save_dict, save_path) 
        else:
            early_stop_cnt += 1
            print("early_stop_cnt", early_stop_cnt)
             
        if valid_rmse < best_rmse:
            best_rmse = valid_rmse
            best_vrm_tr = test_results
            save_dict = {'model':model.state_dict(), 'optim':optimizer.state_dict(), 'rmse':best_rmse}
            torch.save(save_dict, s_path)   
        
        gc.collect()
        torch.cuda.empty_cache()

        if 0 < args.early_stop_patience < early_stop_cnt:
            print(f'Early stop hitted after the epoch {epoch} training!\n')
            break
          
    print(f'using {args.model} to evaluate the {args.dataset} dataset\n')
    print(f"the test results when valid_rmse is minimum:{best_vrm_tr}\n")
    print(f'the valid_pearson is maximum in epoch {best_epoch}\n')
    print(f"the test results when valid_pearson is maximum:{best_vrm_tres}\n")
    return best_vrm_tres['rmse']


params = {"batch_size": 32,
        "learning_rate": 0.0001,
        "lr_reduce_rate": 0.1,
        "dropout_rate": 0.2,
        "hidden_dim": 256,
        "beta1": 1.9,
        "beta2": 0.1,
        "beta3": 0.9,
        "beta4": 0.8}

main(params)