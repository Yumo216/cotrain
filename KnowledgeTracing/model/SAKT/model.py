import torch.nn as nn

from KnowledgeTracing.model.SAKT.attention import Encoder
from KnowledgeTracing.model.SAKT.embedding import Embedding

class SAKTModel(nn.Module):
    def __init__(self, head, length, hidden, n_question, dropout):
        super(SAKTModel, self).__init__()
        self.embedding = Embedding(n_question, length, hidden)
        self.encoder = Encoder(head, length, hidden, dropout)

    def forward(self, y, x_co, QorS):  # shape of input: [batch_size, length, questions * 2]
        x, y = self.embedding(y)  # shape: [batch_size, length, d_model]
        logist = self.encoder(x, y, x_co)  # shape: [batch_size, length, d_model]
        return logist
