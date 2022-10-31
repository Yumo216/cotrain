import torch
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self, n_questions, length, embedding_dim):
        super().__init__()
        self.n_questions = n_questions
        self.q_emb = nn.Linear(embedding_dim * 2, embedding_dim, bias=False)  # 问题的 embedding
        self.k_emb = nn.Linear(embedding_dim * 2, embedding_dim, bias=False)  # 问题和结果的 embedding
        self.pos_emb = nn.Embedding(length, embedding_dim)  # 位置编码
        self.length = length

    def forward(self, x):  # shape of input: [batch_size, length, questions * 2]
        n_batch = x.shape[0]
        p = torch.LongTensor([[i for i in range(self.length)] for j in range(n_batch)]).cuda()
        pos = self.pos_emb(p)

        q = self.q_emb(x)  # shape: [batch_size, length, embedding_dim]
        k = self.k_emb(x)  # shape: [batch_size, length, embedding_dim]

        return q, k+pos
