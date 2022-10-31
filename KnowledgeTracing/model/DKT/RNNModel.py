import torch
import torch.nn as nn
from torch.autograd import Variable
from KnowledgeTracing.Constant import Constants as C


class DKT(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(DKT, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.output_dim = output_dim
        self.rnn = nn.GRU(self.hidden_dim * 4, hidden_dim, layer_dim, batch_first=True)

        self.fc = nn.Linear(hidden_dim, 1)
        self.fc1 = nn.Linear(768, 1)
        self.sig = nn.Sigmoid()
        emb_dim = C.INITIAL
        emb = nn.Embedding(2 * C.NUM_S, emb_dim)
        self.ques = emb(torch.LongTensor([i for i in range(2 * C.NUM_S)])).cuda()

        self.emb_q = nn.Embedding(C.NUM_Q + 1, hidden_dim)
        self.emb_s = nn.Embedding(C.NUM_S + 1, hidden_dim)
        self.emb_q.weight.requires_grad = False
        self.emb_s.weight.requires_grad = False

    def forward(self, x_53):  # shape of input: [batch_size, length, 2q ]
        # x = x_53[:, :, :3]
        # x_d = x.matmul(self.ques)  # x_d [64,50,128]
        ques = x_53[:, :, 0].long()  # [64,50]
        skill = x_53[:, :, 1].long()  # [64,50]
        ans = x_53[:, :, 2].unsqueeze(-1)  # [64,50,1]
        x_co = x_53[:, :, 3:]
        """emb init"""
        ini_q = self.emb_q(ques)  # [64,50,256]
        ini_s = self.emb_s(skill)
        x_2 = torch.cat([ini_q,ini_s],-1)  # [64,50,512]
        left = ans.repeat(1, 1, self.hidden_dim * 2).float()  # [64,50,hid]
        right = (1 - ans).repeat(1, 1, self.hidden_dim * 2).float()
        x = torch.cat([x_2.mul(left), x_2.mul(right)], dim=-1)  # [64,50,4hid]

        '''rnn'''
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).cuda()
        out, hn = self.rnn(x, h0)
        # logit = self.fc(out)
        '''拼上下一题信息'''

        e = x_2[:,1:]
        logit = self.fc1(torch.cat([out[:,:C.MAX_STEP-1],e],-1))
        return logit
