import torch.nn as nn
import torch.nn.functional as F
import torch
from layers import GraphConvolution
from gat import GraphAttentionLayer

class AnomalyDAE(nn.Module):
    def __init__(self,                 
                 in_dim,
                 num_nodes,
                 hidden_dim=64,
                 dropout=0.):
        super(AnomalyDAE, self).__init__()
        
        self.fea_transform = torch.nn.Linear(in_dim, hidden_dim)
        
        self.att_transform2 = torch.nn.Linear(hidden_dim, in_dim)
        self.att_transform1 = torch.nn.Linear(num_nodes, hidden_dim)
        # self.transform = torch.nn.Linear(num_nodes, in_dim)
        
        self.gat = GraphAttentionLayer(hidden_dim, in_dim)
        self.softmax = F.softmax
        self.relu = torch.nn.ReLU()

    def forward(self, x, adj):
        
        structure = self.fea_transform(x)
        structure = self.relu(structure)
        
        emb = self.gat(structure, adj)
        # emb = self.relu(emb)
        
        att_transform = self.att_transform1(x.T)
        att_transform = self.relu(att_transform)
        att_transform = self.att_transform2(att_transform)
        att_transform = self.relu(att_transform)
        
        con_adj = emb @ emb.T
        
        con_att =  emb @ att_transform.T
        
        return con_adj, con_att 
    

# if __name__ == "__main__":
    
        
# class Encoder(nn.Module):
#     def __init__(self, nfeat, nhid, dropout):
#         super(Encoder, self).__init__()
#         self.gc1 = GraphAttentionLayer(nfeat, nhid)
#         self.gc2 = GraphAttentionLayer(nhid, nhid)
#         self.dropout = dropout

#     def forward(self, x, adj):
#         x = F.relu(self.gc1(x, adj))
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = F.relu(self.gc2(x, adj))

#         return x

# class Attribute_Decoder(nn.Module):
#     def __init__(self, nfeat, nhid, dropout):
#         super(Attribute_Decoder, self).__init__()

#         self.gc1 = GraphConvolution(nhid, nhid)
#         self.gc2 = GraphConvolution(nhid, nfeat)
#         self.dropout = dropout

#     def forward(self, x, adj):
#         x = F.relu(self.gc1(x, adj))
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = F.relu(self.gc2(x, adj))

#         return x

# class Structure_Decoder(nn.Module):
#     def __init__(self, nhid, dropout):
#         super(Structure_Decoder, self).__init__()

#         self.gc1 = GraphConvolution(nhid, nhid)
#         self.dropout = dropout

#     def forward(self, x, adj):
#         x = F.relu(self.gc1(x, adj))
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = x @ x.T

#         return x

# class Dominant(nn.Module):
#     def __init__(self, feat_size, hidden_size, dropout):
#         super(Dominant, self).__init__()
        
#         self.shared_encoder = Encoder(feat_size, hidden_size, dropout)
#         self.attr_decoder = Attribute_Decoder(feat_size, hidden_size, dropout)
#         self.struct_decoder = Structure_Decoder(hidden_size, dropout)
    
#     def forward(self, x, adj):
#         # encode
#         x = self.shared_encoder(x, adj)
#         # decode feature matrix
#         x_hat = self.attr_decoder(x, adj)
#         # decode adjacency matrix
#         struct_reconstructed = self.struct_decoder(x, adj)
#         # return reconstructed matrices
#         return struct_reconstructed, x_hat