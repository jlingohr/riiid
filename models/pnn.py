import numpy as np
import torch
import torch.nn as nn


class PNN(nn.Module):
    def __init__(self, embed_size, hidden_dim, keep_prob=0.5):
        super(PNN, self).__init__()
        self.embed_size = embed_size
        self.hidden_dim = hidden_dim
        self.keep_prob = keep_prob
        
        self.W = nn.Linear(195, hidden_dim)
        self.relu = nn.ReLU()
        self.do = nn.Dropout(keep_prob)
        
        self.D = nn.Linear(hidden_dim, 1)
        
    def forward(self, inputs):
        num_inputs = len(inputs)
        num_pairs = int(num_inputs * (num_inputs-1) / 2)

        xw = torch.cat(inputs, axis=1)
        xw3d = torch.reshape(xw, [-1, num_inputs, self.embed_size]).permute(1,0,2)  # [batch_size, 3, embedding_size]

        row = [0, 0, 1]
        col = [1, 2, 2]

        # batch * pair * k
        p = xw3d.index_select(0, torch.LongTensor(row).to(xw.device)).permute(1, 0, 2)
        q = xw3d.index_select(0, torch.LongTensor(col).to(xw.device)).permute(1, 0, 2)
        
        p = torch.reshape(p, [-1, num_pairs, self.embed_size])
        q = torch.reshape(q, [-1, num_pairs, self.embed_size])
        ip = torch.reshape(torch.sum(p * q, [-1]), [-1, num_pairs])
        l = torch.cat([xw, ip], 1)
        
        # TODO double check this is correct
        h = self.W(l)
        h = self.relu(h)
        h = self.do(h)
        p = self.D(h)
        
        return h, p