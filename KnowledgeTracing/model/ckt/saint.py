
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy


class Feed_Forward_block(nn.Module):
    """
    out =  Relu( M_out*w1 + b1) *w2 + b2
    """

    def __init__(self, dim_ff):
        super().__init__()
        self.layer1 = nn.Linear(in_features=dim_ff, out_features=dim_ff)
        self.layer2 = nn.Linear(in_features=dim_ff, out_features=dim_ff)

    def forward(self, ffn_in):
        return self.layer2(F.relu(self.layer1(ffn_in)))


class Encoder_block(nn.Module):
    """
    M = SkipConct(Multihead(LayerNorm(Qin;Kin;Vin)))
    O = SkipConct(FFN(LayerNorm(M)))
    """

    def __init__(self, dim_model, heads_en, total_ex, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.embd_ex = nn.Linear(total_ex, dim_model, bias=False)
        self.embd_in = nn.Linear(total_ex * 2, dim_model, bias=False)
        # embedings
        self.embd_pos = nn.Embedding(seq_len, embedding_dim=dim_model)  # positional embedding

        self.multi_en = nn.MultiheadAttention(embed_dim=dim_model,
                                              num_heads=heads_en)  # multihead attention    ## todo add dropout, LayerNORM
        self.ffn_en = Feed_Forward_block(dim_model)  # feedforward block     ## todo dropout, LayerNorm
        self.layer_norm1 = nn.LayerNorm(dim_model)
        self.layer_norm2 = nn.LayerNorm(dim_model)

    def forward(self, in_ex,in_in):

        ## todo create a positional encoding ( two options numeric, sine)

        in_ex = self.embd_ex(in_ex)
        in_in = self.embd_in(in_in)
        # in_pos = self.embd_pos( in_pos )
        # combining the embedings
        out = in_ex  # + in_pos                      # (b,n,d)

        in_pos = get_pos(self.seq_len)
        in_pos = self.embd_pos(in_pos.cuda())
        out = out + in_pos  # Applying positional embedding
        in_in = in_in + in_pos

        out = out.permute(1, 0, 2)  # (n,b,d)  # print('pre multi', out.shape )
        in_in = in_in.permute(1, 0, 2)

        # Multihead attention
        n, _, _ = out.shape
        out = self.layer_norm1(out)  # Layer norm
        skip_out = out
        out, attn_wt = self.multi_en(out, out, in_in,
                                     attn_mask=get_mask(seq_len=n))  # attention mask upper triangular
        out = out + skip_out  # skip connection

        # feed forward
        out = out.permute(1, 0, 2)  # (b,n,d)
        out = self.layer_norm2(out)  # Layer norm
        skip_out = out
        out = self.ffn_en(out)
        out = out + skip_out  # skip connection

        return out


class Decoder_block(nn.Module):
    """
    M1 = SkipConct(Multihead(LayerNorm(Qin;Kin;Vin)))
    M2 = SkipConct(Multihead(LayerNorm(M1;O;O)))
    L = SkipConct(FFN(LayerNorm(M2)))
    """

    def __init__(self, dim_model, total_in, heads_de, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.embd_in = nn.Linear(total_in, dim_model, bias=False)# interaction embedding
        self.embd_pos = nn.Embedding(seq_len, embedding_dim=dim_model)  # positional embedding
        self.multi_de1 = nn.MultiheadAttention(embed_dim=dim_model,
                                               num_heads=heads_de)  # M1 multihead for interaction embedding as q k v
        self.multi_de2 = nn.MultiheadAttention(embed_dim=dim_model,
                                               num_heads=heads_de)  # M2 multihead for M1 out, encoder out, encoder out as q k v
        self.ffn_en = Feed_Forward_block(dim_model)  # feed forward layer

        self.layer_norm1 = nn.LayerNorm(dim_model)
        self.layer_norm2 = nn.LayerNorm(dim_model)
        self.layer_norm3 = nn.LayerNorm(dim_model)

    def forward(self, in_in, en_out):

        ## todo create a positional encoding ( two options numeric, sine)

        in_in = self.embd_in(in_in)
        # combining the embedings
        out = in_in  # + in_cat #+ in_pos                         # (b,n,d)


        in_pos = get_pos(self.seq_len)
        in_pos = self.embd_pos(in_pos.cuda())
        out = out + in_pos  # Applying positional embedding

        out = out.permute(1, 0, 2)  # (n,b,d)# print('pre multi', out.shape )
        n, _, _ = out.shape

        # Multihead attention M1                                     ## todo verify if E to passed as q,k,v
        out = self.layer_norm1(out)
        skip_out = out
        out, attn_wt = self.multi_de1(out, out, out,
                                      attn_mask=get_mask(seq_len=n))  # attention mask upper triangular
        out = skip_out + out  # skip connection

        # Multihead attention M2                                     ## todo verify if E to passed as q,k,v
        en_out = en_out.permute(1, 0, 2)  # (b,n,d)-->(n,b,d)
        en_out = self.layer_norm2(en_out)
        skip_out = out
        out, attn_wt = self.multi_de2(out, en_out, en_out,
                                      attn_mask=get_mask(seq_len=n))  # attention mask upper triangular
        out = out + skip_out

        # feed forward
        out = out.permute(1, 0, 2)  # (b,n,d)
        out = self.layer_norm3(out)  # Layer norm
        skip_out = out
        out = self.ffn_en(out)
        out = out + skip_out  # skip connection

        return out


# def get_clones(module, N):
#     return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def get_mask(seq_len):
    ##todo add this to device
    return torch.from_numpy(np.triu(np.ones((seq_len, seq_len)), k=1).astype('bool')).cuda()


def get_pos(seq_len):
    # use sine positional embeddinds
    return torch.arange(seq_len).unsqueeze(0)


class saint(nn.Module):
    def __init__(self, dim_model, heads_en, total_ex, total_in, heads_de, seq_len):
        super().__init__()


        self.encoder = Encoder_block(dim_model, heads_en, total_ex, seq_len)
        self.decoder = Decoder_block(dim_model, total_in, heads_de, seq_len)

        self.out = nn.Linear(in_features=dim_model, out_features=1)

    def forward(self, in_ex, in_in,ans):

        ## pass through each of the encoder blocks in sequence
        in_ex = self.encoder(in_ex,in_in)
        # in_cat = in_ex  # passing same output as q,k,v to next encoder block

        ## pass through each decoder blocks in sequence
        in_in = self.decoder(ans, en_out=in_ex)

        ## Output layer
        # in_in = torch.sigmoid(self.out(in_in))
        return in_in

