import math
import torch
import copy
import torch.nn as nn

class self_atten(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, h, length, hidden, dropout):
        super(self_atten, self).__init__()
        self.multi_headed_attention = MultiHeadedAttention(h, hidden)
        self.feed_forward = PositionwiseFeedForward(hidden, hidden * 4)
        self.sublayer = clones(SublayerConnection(hidden, dropout), 2)

    def forward(self, x, y, mask=None):
        "Follow Figure 1 (left) for connections."
        y = self.sublayer[0](y, lambda y: self.multi_headed_attention(x, y, y,  mask))
        return self.sublayer[1](y, self.feed_forward)

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, hidden, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert hidden % h == 0
        # We assume d_v always equals d_k
        self.d_k = hidden // h
        self.h = h
        self.linears = clones(nn.Linear(hidden, hidden), 4)  # (3 + 1)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, query, key, value,  mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from hidden => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x = self.attention(query, key, value,causality=True, mask=mask,
                                 dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

    def attention(self, query, key, value, causality=True, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        if causality:
            scores = torch.tril(scores, diagonal=0, out=None)
            x_cc = torch.Tensor([]).cuda()
            # for i in range(self.h):
            #     x_cc = torch.cat([x_cc, x_co.unsqueeze(1)], dim=1)
            # scores = x_cc
            inf = torch.full_like(scores, float('-inf'))
            scores = torch.where(scores.eq(0), inf, scores)
        p_attn = self.softmax(scores)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value)

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, hidden, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(hidden, d_ff)
        self.w_2 = nn.Linear(d_ff, hidden)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.w_2(self.dropout(self.relu(self.w_1(x))))

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, hidden, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(hidden)  # 用一个维度
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
