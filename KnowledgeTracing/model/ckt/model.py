# -*- coding: utf-8 -*-
# @Time : 2022/5/30 11:07
# @Author : Yumo
# @File : model.py
# @Project: NewKT
# @Comment :

import torch
import torch.nn as nn
from KnowledgeTracing.Constant import Constants as C
from KnowledgeTracing.model.ckt.sakt import sakt
from KnowledgeTracing.model.ckt.saint import saint


class DKT(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(DKT, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.output_dim = output_dim

        # self.embedding = Embedding(C.skill, C.length, C.hidden)
        # self.embedding = Embedding(C.questions, C.length, C.hidden)

        # self.sakt_s = sakt(in_dim=C.hidden, seq_len=C.MAX_STEP, dim=hidden_dim, heads=8, dout=0.2)
        # self.sakt_q = sakt(in_dim=C.hidden, seq_len=C.MAX_STEP, dim=hidden_dim, heads=8, dout=0.2)

        self.saint_s = saint(dim_model=C.hidden,
                             heads_en=16,
                             heads_de=16,
                             total_ex=C.hidden,
                             total_in=C.hidden,
                             seq_len=C.MAX_STEP)
        self.saint_q = saint(dim_model=C.hidden,
                             heads_en=16,
                             heads_de=16,
                             total_ex=C.hidden,
                             total_in=C.hidden,
                             seq_len=C.MAX_STEP)

        self.dkt1 = nn.GRU(self.hidden_dim * 2, hidden_dim, layer_dim, batch_first=True)
        self.dkt2 = nn.GRU(self.hidden_dim * 2, hidden_dim, layer_dim, batch_first=True)

        self.fc_S = nn.Linear(hidden_dim, C.NUM_S)
        self.fc_Q = nn.Linear(hidden_dim * 2, C.NUM_Q)

        self.emb_q = nn.Embedding(C.NUM_Q + 1, hidden_dim)
        self.emb_s = nn.Embedding(C.NUM_S + 1, hidden_dim)
        self.emb_q.weight.requires_grad = False
        self.emb_s.weight.requires_grad = False

        self.emb_a = nn.Embedding(2 + 1, hidden_dim)
        self.emb_a.weight.requires_grad = False

        self.fc1 = nn.Linear(hidden_dim * 2, C.NUM_S)
        self.fc2 = nn.Linear(hidden_dim * 3, C.NUM_Q)

    def forward(self, x_53):  # shape of input: [batch_size, length, 3+50]
        """q,s,a,co"""
        ques = x_53[:, :, 0].long()  # [64,50]
        skill = x_53[:, :, 1].long()  # [64,50]
        anss = x_53[:, :, 2].long()
        ans = x_53[:, :, 2].unsqueeze(-1)  # [64,50,1]
        x_co = x_53[:, :, 3:]
        """emb init"""
        ini_q = self.emb_q(ques)
        ini_s = self.emb_s(skill)  # x_s [64,50,hid]
        """new emb"""
        left = ans.repeat(1, 1, self.hidden_dim).float()  # [64,50,hid]
        right = (1 - ans).repeat(1, 1, self.hidden_dim).float()
        emb_q = torch.cat([ini_q.mul(left), ini_q.mul(right)], dim=-1)  # [64,50,2hid]
        emb_s = torch.cat([ini_s.mul(left), ini_s.mul(right)], dim=-1)  # [64,50,2hid]
        '''dual SAKT'''
        # out_s = self.sakt_s(emb_s, ini_s, x_co)
        # out_q = self.sakt_q(emb_q, ini_q, x_co)
        '''dual SAINT'''
        ans = self.emb_a(anss)
        out_s = self.saint_s(ini_s, emb_s, ans)
        out_q = self.saint_q(ini_q, emb_q, ans)

        '''dual DKT'''
        # out_s, hn = self.dkt1(emb_s)
        # out_q, hn = self.dkt2(emb_q)
        out_q = torch.cat([out_s, out_q], -1)

        # '''拼上下一题信息'''
        # es = ini_s[:,1:]  # [64,49,h]
        # eq = ini_q[:,1:]
        # logit_s = self.fc1(torch.cat([out_s[:, :C.MAX_STEP-1], es], -1))  # [64,49,2h]
        # logit_q = self.fc2(torch.cat([out_q[:, :C.MAX_STEP-1], eq], -1))  # [64,49,3h]

        logit_s = self.fc_S(out_s)
        logit_q = self.fc_Q(out_q)

        return logit_s, logit_q
    #
    # def attention(self, lstm_out,x_co):
    #     """
    #     :param lstm_out: output of LSTM size [batchSize, length, dim]
    #     :return: att_lstm_out [batchSize, length, dim]
    #     """
    #     att_score = torch.bmm(lstm_out, lstm_out.permute(0, 2, 1))  # [batchSize, length, length]
    #     att_score = torch.tril(att_score, diagonal=0)
    #     # att_score = x_co
    #     # x = lstm_out.detach().cpu().numpy()
    #     # a = att_score.detach().cpu().numpy()
    #     """把0置位负无穷大！！！"""
    #     inf = torch.full_like(att_score, float('-inf'))
    #     att_scores = torch.where(att_score.eq(0), inf, att_score)
    #
    #     att_weight = torch.softmax(att_scores, -1)
    #     # b = att_weight.detach().cpu().numpy()
    #     att_lstm_out = torch.bmm(att_weight, lstm_out)  # [batchSize, length, dim]
    #     return att_lstm_out
