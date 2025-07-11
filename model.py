from torch import nn
import torch
import matplotlib.pyplot as plt

"""
    input shape:[batch, seq_len, d_model]
"""

class PositionEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=512):
        super().__init__()
        # shape: [max_seq_len, 1]
        position = torch.arange(0, max_seq_len).unsqueeze(1)
        item = 1/10000 ** (torch.arange(0, d_model, 2)/d_model)
        tmp_pos = position * item
        pe = torch.zeros(max_seq_len, d_model)
        pe[:, 0::2] = torch.sin(tmp_pos)
        pe[:, 1::2] = torch.cos(tmp_pos)
        # plt.matshow(pe)
        # plt.show()  这两行用于可视化位置编码的图像
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe, False)



    def forward(self, x):
        batch, seq_len,_ = x.shape
        pe = self.pe
        return x + pe[:,:seq_len,:]




def attention(query, key, value, mask=None):
    d_model = key.shape[-1]
    # query, key, value shape:[batch, seq_len, d_model]
    att_ = torch.matmul(query, key.transpose(-2, -1)) / d_model ** 0.5
    if mask is not None:
        att_ = att_.masked_fill(mask, -1e9)

    att_score = torch.softmax(att_, dim=-1)
    return torch.matmul(att_score, value)





class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()
        assert d_model % heads == 0   # 这里的做法是将不同的注意力头分治不同的qkv部分
        self.q_linear = nn.Linear(d_model, d_model, bias=False)
        self.k_linear = nn.Linear(d_model, d_model, bias=False)
        self.v_linear = nn.Linear(d_model, d_model, bias=False)
        self.linear = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.heads = heads
        self.d_k = d_model // heads
        self.d_model = d_model


    def forward(self, q, k, v, mask=None):
        # [n, seq_len, d_model] -> [n, heads, seq_len, d_k]
        # 这一步中，将输入x分布在三个linear中计算得到qkv，隐含了“w”矩阵
        q = self.q_linear(q).reshape(q.shape[0], -1, self.heads, self.d_k).transpose(1, 2)
        k = self.k_linear(k).reshape(q.shape[0], -1, self.heads, self.d_k).transpose(1, 2)
        v = self.v_linear(v).reshape(q.shape[0], -1, self.heads, self.d_k).transpose(1, 2)
        out = attention(q, k, v, mask)
        out = out.transpose(1,2).reshape(out.shape[0], -1, self.d_model)
        out = self.linear(out)
        out = self.dropout(out)
        return out




class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.ffn(x)




class EncoderLayer(nn.Module):
    def __init__(self, heads, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.self_multi_head_att = MultiHeadAttention(heads, d_model, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for i in range(2)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        multi_head_att_out = self.self_multi_head_att(x, x, x, mask)
        multi_head_att_out = self.norms[0](x + multi_head_att_out)
        ffn_out = self.ffn(multi_head_att_out)
        ffn_out = self.norms[1](multi_head_att_out + ffn_out)
        out = self.dropout(ffn_out)
        return out


class Encoder(nn.Module):
    def __init__(self, vocab_size, pad_idx, d_model, heads, num_layers, d_ff, max_seq_len=512, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, pad_idx)
        self.positional_encode = PositionEncoding(d_model, max_seq_len)
        self.encoder_layers = nn.ModuleList([EncoderLayer(heads, d_model, d_ff, dropout) for i in range(num_layers)])


    def forward(self, x, src_mask):
        embed_x = self.embedding(x)
        pos_encode_x = self.positional_encode(embed_x)
        for layer in self.encoder_layers:
            pos_encode_x = layer(pos_encode_x, src_mask)
        return pos_encode_x





class DecoderLayer(nn.Module):
    def __init__(self, heads, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.masked_att = MultiHeadAttention(heads, d_model, dropout)
        self.att = MultiHeadAttention(heads, d_model, dropout)
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for i in range(3)])
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encode_kv, dst_mask=None, src_dst_mask=None):
        masked_att_out = self.masked_att(x, x, x, dst_mask)
        masked_att_out = self.norms[0](x + masked_att_out)
        att_out = self.att(masked_att_out, encode_kv, encode_kv, src_dst_mask)
        att_out = self.norms[1](att_out + masked_att_out)
        ffn_out = self.ffn(att_out)
        ffn_out = self.norms[2](ffn_out + att_out)
        out = self.dropout(ffn_out)
        return out





class Decoder(nn.Module):
    def __init__(self,vocab_size, pad_idx, d_model, heads, num_layers, d_ff, max_seq_len=512, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, pad_idx)
        self.positional_encode = PositionEncoding(d_model, max_seq_len)
        self.decoder_layers = nn.ModuleList([DecoderLayer(heads, d_model, d_ff, dropout) for i in range(num_layers)])

    def forward(self, x, encoder_kv, dst_mask=None, src_dst_mask=None):
        embed_x = self.embedding(x)
        pos_encode_x = self.positional_encode(embed_x)
        for layer in self.decoder_layers:
            pos_encode_x = layer(pos_encode_x, encoder_kv, dst_mask, src_dst_mask)
        return pos_encode_x






class Transformer(nn.Module):
    def __init__(self, enc_vocab_size, dec_vocab_size, pad_idx, d_model, heads, num_layers, d_ff, max_seq_len=512, dropout=0.1):
        super().__init__()
        self.encoder = Encoder(enc_vocab_size, pad_idx, d_model, heads, num_layers, d_ff, max_seq_len, dropout)
        self.decoder = Decoder(dec_vocab_size, pad_idx, d_model, heads, num_layers, d_ff, max_seq_len, dropout)
        self.linear = nn.Linear(d_model, dec_vocab_size)
        self.pad_idx = pad_idx

    def generate_mask(self, query, key, is_triu_mask=False):    # 最后一个参数用于判断是否是用于masked多头还是padding mask
        device = query.device
        # batch, seq_len
        batch, seq_q = query.shape
        _, seq_k = key.shape
        # batch, head, seq_q, seq_k
        mask = (key == self.pad_idx).unsqueeze(1).unsqueeze(2)
        mask = mask.expand(batch, 1, seq_q, seq_k).to(device)
        if is_triu_mask:
            dst_triu_mask = torch.triu(torch.ones(seq_q, seq_k, dtype=torch.bool), diagonal=1)
            dst_triu_mask = dst_triu_mask.unsqueeze(0).unsqueeze(1).expand(batch, 1, seq_q, seq_k).to(device)
            return mask|dst_triu_mask
        return mask


    def forward(self, src, dst):
        src_mask = self.generate_mask(src, src)  # 输入部分的padding mask
        encoder_out = self.encoder(src, src_mask)
        dst_mask = self.generate_mask(dst, dst, True)
        src_dst_mask = self.generate_mask(dst, src)
        decoder_out = self.decoder(dst, encoder_out, dst_mask, src_dst_mask)
        out = self.linear(decoder_out)
        return out



if __name__ == '__main__':
    # PositionEncoding(512, 100) 测试位置编码样式
    # att = MultiHeadAttention(8, 512, 0.2) 测试多头注意力的维度变化是否正确
    # x = torch.randn(4, 100, 512)
    # out = att(x, x, x)
    # print(out.shape)
    att = Transformer(100, 200, 0, 512, 8, 6, 1024, 512, 0.1)
    x = torch.randint(0, 100, (4, 64))
    y = torch.randint(0, 200, (4, 64))
    out = att(x, y)
    print(out.shape)