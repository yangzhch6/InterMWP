from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np

class LayerNormalization(nn.Module):
    def __init__(self, d_hid, eps=1e-6):
        super(LayerNormalization, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_hid))
        self.beta = nn.Parameter(torch.zeros(d_hid))
        self.eps = eps

    def forward(self, z):
        mean = z.mean(dim=-1, keepdim=True,)
        std = z.std(dim=-1, keepdim=True,)
        ln_out = (z - mean) / (std + self.eps)
        ln_out = self.gamma * ln_out + self.beta

        return ln_out

class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x

class LogicScaledDotProductAttention(nn.Module):
    def __init__(self, hidden_size, dropout=.1):
        super(LogicScaledDotProductAttention, self).__init__()
        self.scale_factor = np.sqrt(hidden_size)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.w_q = nn.Linear(hidden_size, hidden_size)
        self.w_k = nn.Linear(hidden_size, hidden_size)
        self.w_v = nn.Linear(hidden_size, hidden_size)

        self.layer_norm = LayerNormalization(hidden_size)

    def forward(self, problem, logic, attn_mask=None):
        # problem: [batch_size x logic_size x len_p x h]
        # logic:   [batch_size x logic_size x len_l x h]
        q = self.w_q(problem)
        k = self.w_k(logic)
        v = self.w_v(logic)
        # attn: [batch_size x logic_size x len_p x len_l]
        scores = torch.matmul(q, k.transpose(-1, -2)) / self.scale_factor
        if attn_mask is not None:
            assert attn_mask.size() == scores.size()
            scores.masked_fill_(attn_mask==0, -1e12)
        attn = self.dropout(self.softmax(scores))

        # [batch_size x logic_size x len_p x h]
        context = torch.matmul(attn, v)

        output = self.dropout(context + problem)
        output = self.layer_norm(output)
        return output, attn


class LogicTransformerLayer(nn.Module):
    def __init__(self, hidden_size, dropout=0.1):
        super(LogicTransformerLayer, self).__init__()
        self.hidden_size = hidden_size
        self.logic_attn = LogicScaledDotProductAttention(hidden_size, dropout)
        self.pos_ffn = PositionwiseFeedForward(hidden_size, hidden_size, dropout)

    def forward(self, problem, logic, attn_mask_p, attn_mask_l):
        '''
        problem: [batch_size x len_p x h]
        logic:   [logic_size x len_l x h]
        attn_mask_p: [batch_size x len_p]
        attn_mask_l: [logic_size x len_l]
        '''
        batch_size = problem.shape[0]
        logic_size = logic.shape[0]
        len_p = attn_mask_p.shape[-1]
        len_l = attn_mask_l.shape[-1]

        # [batch_size x logic_size x len_p x h]
        problem = problem.unsqueeze(1).repeat(1, logic_size, 1, 1) 
        # [batch_size x logic_size x len_l x h]
        logic = logic.unsqueeze(0).repeat(batch_size, 1, 1, 1)     
        # [batch_size x logic_size x len_p x len_l]
        attn_mask_p = attn_mask_p.reshape(batch_size, 1, -1, 1).repeat(1, logic_size, 1, len_l) 
        # [batch_size x logic_size x len_p x len_l]
        attn_mask_l = attn_mask_l.reshape(1, logic_size, 1, -1).repeat(batch_size, 1, len_p, 1) 
        
        # [batch_size x logic_size x len_p x len_l]
        attention_mask = torch.mul(attn_mask_p, attn_mask_l)

        output, attn_score = self.logic_attn(problem, logic, attention_mask)
        output = self.pos_ffn(output)

        return output, attn_score


class Retriever(nn.Module):
    def __init__(self, hidden_size, dropout=0.1):
        super(Retriever, self).__init__()
        self.hidden_size = hidden_size
        self.logic_transformer = LogicTransformerLayer(hidden_size, dropout)
        self.score = nn.Linear(hidden_size, 1)
    
    def avg_pooling(self, outputs, attn_mask_p):
        # outputs: # [batch_size x logic_size x len_p x h]
        # attn_mask_p: [batch_size x len_p]

        # [batch_size x logic_size x len_p x h]
        batch_size = outputs.shape[0]
        logic_size = outputs.shape[1]
        mask_for_avg = attn_mask_p.reshape(batch_size, 1, -1, 1)
        mask_for_avg = mask_for_avg.repeat(1, logic_size, 1, self.hidden_size)
        outputs = outputs * mask_for_avg
        outputs = torch.sum(outputs, dim=-2) # [batch_size x logic_size x h]
        sentence_len = torch.sum(mask_for_avg, dim=-2) 
        outputs = outputs / sentence_len
        return outputs
    
    def forward(self, problem, logic, attn_mask_p, attn_mask_l):
        outputs, _ = self.logic_transformer(
            problem, logic, attn_mask_p, attn_mask_l
        )
        outputs = self.avg_pooling(outputs, attn_mask_p)
        score = self.score(outputs) # [batch_size x logic_size x 1]
        return score.squeeze(-1)
