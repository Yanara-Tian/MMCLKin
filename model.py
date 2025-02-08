import dgl
import math
import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_batch
import torch.nn.functional as F                        
from torch.autograd import Variable 
from torch_geometric.nn import  GCNConv
from torch_geometric.nn import SAGEConv 
from torch.nn.utils.rnn import pad_sequence
from comnet import EComENet
from torch_geometric.utils import to_dense_batch

class A_MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, embedding_dim, num_heads):
        super(A_MultiHeadAttention, self).__init__()
        assert embedding_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embedding_dim  
        self.num_heads = num_heads 
        self.head_dim = embedding_dim // num_heads 
        self.qkv_proj = nn.Linear(input_dim, 3 * embedding_dim)   
        self.o_proj = nn.Linear(embedding_dim, embedding_dim)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def scaled_dot_product(self, q, k, v, mask=None):
        d_k = q.size()[-1]  # 64
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) #1,2,68,68
        attn_logits_a = attn_logits / math.sqrt(d_k) # 缩放，以防止梯度消失或梯度爆炸，防止softmax函数梯度国小
        if mask is not None:
            attn_logits = attn_logits_a.masked_fill(mask == 0, -9e15)
        attention = F.softmax(attn_logits, dim=-1)# 内积过大
        values = torch.matmul(attention, v)
        return values, attention, attn_logits_a

    def forward(self, x, mask=None, return_attention=False): 
        batch_size, seq_length, embed_dim = x.size() # 1,68,128
        qkv = self.qkv_proj(x) # 1,68,384
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim)  #1,68,2,192
        qkv = qkv.permute(0, 2, 1, 3)   # 1,2,68,192
        q, k, v = qkv.chunk(3, dim=-1) # 1,2,68,64

        values, attention, attn_logits_a = self.scaled_dot_product(q, k, v, mask=mask)
        values = values.permute(0, 2, 1, 3)  # 1,68,2,64
        values = values.reshape(batch_size, seq_length, embed_dim)   # 1,68,128
        o = self.o_proj(values) # 1,68,128
        if return_attention:
            return o, attention, attn_logits_a
        else:
            return o

class LinkAttention(nn.Module):
    def __init__(self, input_dim, n_heads):
        super(LinkAttention, self).__init__()
        self.query = nn.Linear(input_dim, n_heads)  #输入是input_dim，输出是8
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x, masks):  # smiles_out, smiles_mask
        query = self.query(x).transpose(1,2)   # (1,8,21)
        value = x  # x (1,21,128)
        
        minus_inf = -9e15*torch.ones_like(query)  # 生成与input形状相同,元素全为1的张量 (256,8,81)
        e = torch.where(masks>0.5, query, minus_inf) 
        a = self.softmax(e)  #（256，8，81）
        out = torch.matmul(a, value) #a（256，8，81）; value (256,81,256); out: (256, 8，256)
        b = torch.sum(out, dim=1)
        out = torch.sum(out, dim=1).squeeze()  #在行上对维度进行加和 （256，256）
        return out, a
            
class MMCLKin(nn.Module):  
    def __init__(self, lstm_dropout, alpha, num_heads, hidden_dim, dropout_rate, n_head, smile_vocab, local_rank):
        super(MMCLKin, self).__init__() 
        self.local_rank=local_rank
        self.smile_vocab = smile_vocab
        self.hidden_dim = hidden_dim
        self.n_heads = num_heads
        self.lstm_dropout = lstm_dropout
        self.dropout = nn.Dropout(dropout_rate)
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.relu = nn.ReLU()
        self.prelu = nn.PReLU()
        self.elu = nn.ELU()
        self.conv = GCNConv(hidden_dim + 3, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.layer_norm1 = nn.LayerNorm(hidden_dim*2)
        self.bilstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=1, bidirectional=True, dropout=self.lstm_dropout)
        self.self_attention = A_MultiHeadAttention(hidden_dim*2, hidden_dim*2, num_heads)
        self.out_fc = nn.Linear(hidden_dim*2, hidden_dim)
        self.out_m_fc = nn.Linear(hidden_dim, int(hidden_dim/2))
        self.output_fc = nn.Linear(int(hidden_dim/2), 1)
        self.h_0 = Variable(torch.zeros(2, 1, self.hidden_dim))
        self.c_0 = Variable(torch.zeros(2, 1, self.hidden_dim))
        self.smi_egnn = dgl.nn.EGNNConv(24, self.hidden_dim, self.hidden_dim, 8)
        self.poc_egnn = dgl.nn.EGNNConv(6, self.hidden_dim, self.hidden_dim, 32)
        self.pro_egnn = dgl.nn.EGNNConv(6, self.hidden_dim, self.hidden_dim, 32)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.conv1 = SAGEConv(self.hidden_dim, self.hidden_dim)
        self.conv2 = SAGEConv(self.hidden_dim, self.hidden_dim)
        self.comnent = EComENet(device=self.local_rank, cutoff=8.0,num_layers=4,node_dim=75,edge_dim=8,hidden_channels=self.hidden_dim,\
            out_channels=self.hidden_dim, num_radial=3,num_spherical=2,num_output_layers=3, mol_edge_in_dim=8)
        self.pcomnent = EComENet(device=self.local_rank, cutoff=8.0,num_layers=4,node_dim=6,edge_dim=32,hidden_channels=self.hidden_dim,\
            out_channels=self.hidden_dim, num_radial=3,num_spherical=2,num_output_layers=3, mol_edge_in_dim=32)
        self.lstm0 = nn.LSTM(1280, hidden_dim, num_layers=1, bidirectional=True, dropout=self.lstm_dropout)
        self.lstm1 = nn.LSTM(hidden_dim *2, hidden_dim, num_layers=1, bidirectional=True, dropout=self.lstm_dropout)
        self.lstm2 = nn.LSTM(1280, hidden_dim , num_layers=1, bidirectional=True, dropout=self.lstm_dropout)
        self.lstm3 = nn.LSTM(hidden_dim *2, hidden_dim , num_layers=1, bidirectional=True, dropout=self.lstm_dropout)
        
        self.smiles_embed = nn.Embedding(self.smile_vocab+1, 256, padding_idx=self.smile_vocab)
        
        self.gru0 = nn.GRU(1280, self.hidden_dim * 5, batch_first=True)
        self.gru1 = nn.GRU(self.hidden_dim * 5, self.hidden_dim*2, batch_first=True)
        self.gru2 = nn.GRU(1280, self.hidden_dim*5, batch_first=True)
        self.gru3 = nn.GRU(self.hidden_dim * 5, self.hidden_dim*2, batch_first=True)
        
        self.smiles_input_fc = nn.Sequential(
            nn.Linear(384, self.hidden_dim*4),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim*4, self.hidden_dim))
        self.smiles_rnn = nn.LSTM(self.hidden_dim, self.hidden_dim, batch_first=True, bidirectional=True, dropout=dropout_rate)
        self.out_attention = LinkAttention(hidden_dim, n_heads=8)  # in：256；out:8
        self.number_heads = n_head
          
    def forward(self, x_feats):
        #  graph
        # smiles_feature
        e_smi_feats = self.comnent(x_feats.mol_dist, x_feats.mol_theta, x_feats.mol_phi, x_feats.mol_tau, x_feats.mol_atoms_feats, x_feats.mol_edge_index, x_feats.mol_edges_feats)
        smi_feats = self.elu(self.conv2(self.elu(self.conv1(e_smi_feats, x_feats.mol_edge_index)), x_feats.mol_edge_index)).permute(1, 0) #512,33
        smi_fts = torch.cat([to_dense_batch(smi_feats[i], x_feats.mol_batch)[0].unsqueeze(0) for i in range(smi_feats.shape[0])], dim=0) # 64,32,122 
        smi_feats = smi_fts.permute(1, 2, 0).to(self.local_rank)  
        
        # pockets_feature
        e_poc_feats = self.pcomnent(x_feats.po_dist, x_feats.po_theta, x_feats.po_phi, x_feats.po_tau, x_feats.poc_atoms_feats_s, x_feats.poc_edge_index, x_feats.poc_edges_feats_s)
        e_poc_feats = self.dropout(e_poc_feats)
        poc_feats = self.elu(self.conv2(self.elu(self.conv1(e_poc_feats.to(self.local_rank), x_feats.poc_edge_index.to(self.local_rank))), x_feats.poc_edge_index.to(self.local_rank))).permute(1, 0)  # 41,64
        poc_fts = torch.cat([to_dense_batch(poc_feats[i], x_feats.poc_batch)[0].unsqueeze(0) for i in range(poc_feats.shape[0])], dim=0) # 64,32,122 
        poc_feats = poc_fts.permute(1, 2, 0).to(self.local_rank)  
      
        # proteins_feature
        e_pro_feats = self.pcomnent(x_feats.pr_dist, x_feats.pr_theta, x_feats.pr_phi, x_feats.pr_tau, x_feats.pro_atoms_feats_s,x_feats.pro_edge_index, x_feats.pro_edges_feats_s)
        e_pro_feats = self.dropout(e_pro_feats)
        pro_feats = self.elu(self.conv2(self.elu(self.conv1(e_pro_feats.to(self.local_rank), x_feats.pro_edge_index.to(self.local_rank))), x_feats.pro_edge_index.to(self.local_rank))).permute(1, 0)  # 910, 64
        pro_fts = torch.cat([to_dense_batch(pro_feats[i], x_feats.pro_batch)[0].unsqueeze(0) for i in range(pro_feats.shape[0])], dim=0) # 64,32,122 
        pro_feats = pro_fts.permute(1, 2, 0).to(self.local_rank)   
    
        # local_global_feats
        # smiles_pockets
        smiles_pockets_feats = torch.cat((smi_feats, poc_feats), dim=1).view(smi_feats.shape[0], -1, self.hidden_dim)   # torch.Size([32, 196, 512])
        re_smiles_pockets_feats = self.layer_norm(smiles_pockets_feats).permute(1, 0, 2)  #torch.Size([196, 32, 512])
        h_0 = Variable(torch.zeros(2, smi_feats.shape[0], self.hidden_dim).to(self.local_rank))  # torch.Size([2, 32, 512])
        c_0 = Variable(torch.zeros(2, smi_feats.shape[0], self.hidden_dim).to(self.local_rank))  # torch.Size([2, 32, 512])
        bi_smiles_pockets_feats, _ = self.bilstm(re_smiles_pockets_feats, (h_0, c_0))  # 68,1,128 
        outbi_smiles_pockets_feats = bi_smiles_pockets_feats.permute(1, 0, 2).to(self.local_rank)   # torch.Size([2, 130, 1024])

        smi_poc_max_dim = outbi_smiles_pockets_feats.shape[1]  # 68
        mask = self.self_mask(outbi_smiles_pockets_feats, smi_poc_max_dim)  # 1,68,68
        output_smiles_pockets, _, spo_attention= self.self_attention(outbi_smiles_pockets_feats, mask=mask, return_attention=True)  # torch.Size([2, 130, 1024])
   
        # smiles_protein
        smiles_proteins_feats = torch.cat((smi_feats, pro_feats), dim=1).view(smi_feats.shape[0], -1, self.hidden_dim)  # 1,295,64
        re_smiles_proteins_feats = self.layer_norm(smiles_proteins_feats).permute(1, 0, 2).to(self.local_rank)  # 295,1,64
        h_0 = Variable(torch.zeros(2, smi_feats.shape[0], self.hidden_dim).to(self.local_rank))
        c_0 = Variable(torch.zeros(2, smi_feats.shape[0], self.hidden_dim).to(self.local_rank))
        bi_smiles_proteins_feats, _ = self.bilstm(re_smiles_proteins_feats, (h_0, c_0))  # 295, 1, 128
        outbi_smiles_proteins_feats = bi_smiles_proteins_feats.permute(1, 0, 2).to(self.local_rank)  # torch.Size([2, 978, 1024])
        
        smi_pro_max_dim = outbi_smiles_proteins_feats.shape[1]  # 295
        mask = self.self_mask(outbi_smiles_proteins_feats, smi_pro_max_dim)  # 1,295,295
        output_smiles_proteins, _, spr_attention = self.self_attention(outbi_smiles_proteins_feats, mask=mask, return_attention=True)  # torch.Size([2, 978, 1024])
      
        ###sequence
        #smiles
        smiles = self.smiles_input_fc(x_feats.mol_embedding)  #torch.Size([2, 156, 512])
        smiles_out, _ = self.smiles_rnn(smiles) 
        # protein
        prot_seq_embedding, _ = self.lstm0(x_feats.pro_token_repre)
        prot_seq_embedding, _ = self.lstm1(prot_seq_embedding)  # torch.Size([2, 899, 1024])
        prot_seq_embedding = self.dropout(prot_seq_embedding)
        
        pock_seq_embedding, _ = self.lstm2(x_feats.poc_token_repre)
        pock_seq_embedding, _ = self.lstm3(pock_seq_embedding)  # torch.Size([2, 51, 1024])
        pock_seq_embedding = self.dropout(pock_seq_embedding)
        
        #smi_protein
        spr_cat = torch.cat((smiles_out, prot_seq_embedding), dim=1)
        spr_cat_max_dim = spr_cat.shape[1] 
        spr_masks = self.self_mask(spr_cat, spr_cat_max_dim)  # 1,68,68
        output_spr, _, spr_seq_attention= self.self_attention(spr_cat, mask=spr_masks, return_attention=True)  # torch.Size([2, 1055, 1024])
        
        #smi_pockets
        spo_cat = torch.cat((smiles_out, pock_seq_embedding), dim=1) 
        spo_cat_max_dim = spo_cat.shape[1] 
        spo_masks = self.self_mask(spo_cat, spo_cat_max_dim)  # 1,68,68
        output_spo, _, spo_seq_attention= self.self_attention(spo_cat, mask=spo_masks, return_attention=True)  #torch.Size([2, 207, 1024])
       
        # all
        # smiles_pocket_protein
        all_feats = torch.cat((output_smiles_pockets, output_smiles_proteins, output_spr, output_spo), dim=1)  # torch.Size([16, 2334, 512])
        # pool_feats = global_add_pool(self.layer_norm1(all_feats), x_feats.all_batch)
        pool_feats = self.pool(self.layer_norm1(all_feats).permute(0, 2, 1)).reshape(smi_feats.shape[0], self.hidden_dim*2)  # torch.Size([16, 512])
        s_out = self.dropout(self.elu(self.out_fc(pool_feats)))
        m_out = self.dropout(self.elu(self.out_m_fc(s_out)))
        out = self.output_fc(m_out)  # torch.Size([2, 1])
        
        # smi_poc graph attention and smi_pocinpro graph attention contrast learning
        # graph-based smiles and pocket attention
        smi_poc_gra_att = torch.sum(torch.sum(spo_attention, dim=1), dim=2)  # torch.Size([2, 130])
        layer_norm2 = nn.LayerNorm(smi_poc_gra_att.size(1), elementwise_affine=False).to(self.local_rank)
        n_spo_gra_att = layer_norm2(smi_poc_gra_att)  
        # pure graph-based smiles and pocket attention (mol atom align with smile sequence)
        pu_smi_poc_gra_att = []
        for i, pu_smi_poc_gra in enumerate(smi_poc_gra_att):
            smi_length = smi_feats.shape[1]
            s_gra_att = pu_smi_poc_gra[x_feats.atominmol_indexes[i]].squeeze(0)  # 20
            spo_gra_att = torch.cat((s_gra_att, pu_smi_poc_gra[smi_length:]), dim=0)  # torch.Size([130])
            layer_norm6 = nn.LayerNorm(spo_gra_att.size(0), elementwise_affine=False).to(self.local_rank)
            pu_spo_gra_att = layer_norm6(spo_gra_att)  # torch.Size([130])
            pu_smi_poc_gra_att.append(pu_spo_gra_att)
        pu_spo_gra_att = pad_sequence(pu_smi_poc_gra_att).permute(1,0)
        
        # graph-based smiles and (pocket in protein) attention
        smi_pro_gra_sum = torch.sum(torch.sum(spr_attention, dim=1), dim=2)  # torch.Size([2, 978])
        smi_pro_gra_su = []
        for i, smi_pro_gra in enumerate(smi_pro_gra_sum):
            pro_att = smi_pro_gra[smi_length:]
            pocinpro_gra_att = pro_att[x_feats.pocinpro_indexes[i]].squeeze(0)  # torch.Size([49])
            smi_pro_gra_att = torch.cat((smi_pro_gra[:smi_length], pocinpro_gra_att), dim=0) #torch.Size([130])
            layer_norm3 = nn.LayerNorm(smi_pro_gra_att.size(0), elementwise_affine=False).to(self.local_rank)
            n_spr_gra_att = layer_norm3(smi_pro_gra_att) 
            smi_pro_gra_su.append(n_spr_gra_att)
        n_spr_gra_att = pad_sequence(smi_pro_gra_su).permute(1,0)  
        #pure graph-based smiles and protein attention(mol atom align with smile sequence)
        pu_smi_pro_gra_su = []
        for i, smi_pro_gra in enumerate(smi_pro_gra_sum):
            smi_pro_gra_att = torch.cat((smi_pro_gra[x_feats.atominmol_indexes[i]].squeeze(0), smi_pro_gra[smi_length:1022+smi_length]), dim=0)  # torch.Size([978])
            layer_norm7 = nn.LayerNorm(smi_pro_gra_att.size(0), elementwise_affine=False).to(self.local_rank)
            pu_spr_gra_att = layer_norm7(smi_pro_gra_att)  # torch.Size([978])
            pu_smi_pro_gra_su.append(pu_spr_gra_att)
        pu_spr_gra_att = pad_sequence(pu_smi_pro_gra_su).permute(1,0)
        
        # smi_poc sequence attention and smi_pocinpro sequence attention contrast learning(with start and end token)
        # sequence-based smile and pockets attention (with start token and end token)
        smi_poc_seq_att = torch.sum(torch.sum(spo_seq_attention, dim=1), dim=2) # torch.Size([2, 207])
        layer_norm4 = nn.LayerNorm(smi_poc_seq_att.size(1), elementwise_affine=False).to(self.local_rank)
        n_spo_seq_att = layer_norm4(smi_poc_seq_att)  # torch.Size([2, 207])
        #pure sequence-based smile and pocket attention(without start end end token, smile sequence aligns with the mol atom)
        pu_smi_poc_seq_att = []
        for i, pu_smi_poc_seq in enumerate(smi_poc_seq_att):
            sms_length = x_feats.mol_embedding.shape[1]
            poc_aa = pu_smi_poc_seq[sms_length+1:-1]
            smi_poc_seq_at = torch.cat((pu_smi_poc_seq[x_feats.subwinsmi_indexes[i] + 1].squeeze(0), poc_aa), dim=0)  # torch.Size([130])
            layer_norm8 = nn.LayerNorm(smi_poc_seq_at.size(0), elementwise_affine=False).to(self.local_rank)
            pu_spo_seq_att = layer_norm8(smi_poc_seq_at)  # torch.Size([130])
            pu_smi_poc_seq_att.append(pu_spo_seq_att)
        pu_spo_seq_att = pad_sequence(pu_smi_poc_seq_att).permute(1,0)  # torch.Size([2, 130])
        
        # sequence-based smile and (pocket in protein) attention (with start token and end token)
        smi_pro_seq_sum = torch.sum(torch.sum(spr_seq_attention, dim=1), dim=2) # torch.Size([2, 1055])
        smi_pro_seq_su = []
        for i, smi_pro_seq in enumerate(smi_pro_seq_sum):
            index = x_feats.pocinpro_indexes[i] + 1
            pro_seq_att = smi_pro_seq[sms_length:]
            pocinpro_seq_att = torch.cat((pro_seq_att[[0]], pro_seq_att[index].squeeze(0), pro_seq_att[[-1]]), dim=0)
            smi_pro_seq_att = torch.cat((smi_pro_seq[:sms_length], pocinpro_seq_att), dim=0)  # torch.Size([207])
            layer_norm5 = nn.LayerNorm(smi_pro_seq_att.size(0), elementwise_affine=False).to(self.local_rank)
            n_spr_seq_att = layer_norm5(smi_pro_seq_att)  # torch.Size([207])
            smi_pro_seq_su.append(n_spr_seq_att)
        n_spr_seq_att = pad_sequence(smi_pro_seq_su).permute(1,0)
        # pure sequence-based smile and (pocket in protein) attention (without start token and end token, smile sequence aligns with the mol atom)
        pu_smi_pro_seq_su = []
        for i, smi_pro_seq in enumerate(smi_pro_seq_sum):
            pro_seq_aa = smi_pro_seq[sms_length+1:]
            if pro_seq_aa.shape[0] > 1022:
                pro_seq_aa = pro_seq_aa[:1022]
            else:
                pro_seq_aa = pro_seq_aa[:-1]
            smi_pro_seq_att = torch.cat((smi_pro_seq[x_feats.subwinsmi_indexes[i] + 1].squeeze(0), pro_seq_aa), dim=0)
            layer_norm9 = nn.LayerNorm(smi_pro_seq_att.size(0), elementwise_affine=False).to(self.local_rank)
            pu_spr_seq_att = layer_norm9(smi_pro_seq_att)  # 482
            pu_smi_pro_seq_su.append(pu_spr_seq_att)
        pu_spr_seq_att = pad_sequence(pu_smi_pro_seq_su).permute(1,0)
        
        return out, n_spo_gra_att, n_spr_gra_att, n_spo_seq_att, n_spr_seq_att, pu_spo_gra_att, pu_spr_gra_att, pu_spo_seq_att, pu_spr_seq_att
        
        
    def generate_masks(self, adj,n_heads):  #定义了8个head，同时将所有的smiles用1进行掩码
        out = torch.ones(adj.shape[0], adj.shape[1])  # 256*81，选取smiles_out的第0位的维度以及第一位的维度，并形成相应的由1组成的矩阵
        max_size = adj.shape[1]  #81
        out = out.unsqueeze(1).expand(-1, n_heads, -1)  # 256，8，81  unsqueeze()是在第二维增加一个维度，
        return out.cuda(device=adj.device)
    
    def self_mask(self, input, max_dim):
        mask = torch.eye(max_dim, dtype=torch.uint8).view(1, max_dim, max_dim).to(self.local_rank)
        mask[0, input.size()[1]:max_dim, :] = 0
        mask[0, :, input.size()[1]:max_dim] = 0
        mask[0, :, input.size()[1] - 1] = 1
        mask[0, input.size()[1] - 1, :] = 1
        mask[0,  input.size()[1] - 1,  input.size()[1] - 1] = 0
        return mask 
 