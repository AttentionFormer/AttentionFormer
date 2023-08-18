import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding, DataEmbedding_wo_pos
from layers.AttentionCorrelation import AutoCorrelation, AutoCorrelationLayer,SAttentionLayer
from layers.MultiWaveletCorrelation import MultiWaveletCross, MultiWaveletTransform
from layers.SelfAttention_Family import FullAttention, ProbAttention
from layers.Attentionformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp, series_decomp_multi
import math
import numpy as np


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class FeedForward(nn.Module):
    def __init__(self, in_channels=None, out_channels=None, seq_len=None, example=None):
        super(FeedForward, self).__init__()
        if example:
            self.in_channels=example.in_channels
            self.out_channels=example.out_channels
            self.seq_len=example.seq_len
            self.weights1 = nn.Parameter(torch.Tensor(8, self.in_channels//8 , self.out_channels//8 , self.seq_len))
            return

        self.in_channels=in_channels
        self.out_channels=out_channels
        self.seq_len=seq_len
        self.weights1 = nn.Parameter(torch.Tensor(8, in_channels//8 , out_channels//8 , seq_len))

    def forward(self, q, k, v, mask):
        out_ft = torch.einsum("bxhe,heox->bhox", q, self.weights1)
        x = out_ft
        return (x, None)
    

class AttentionFormer(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(AttentionFormer, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection
    def forward(self, x, cross, x_mask=None, cross_mask=None, trend=None):
        for layer in self.layers:
            x, residual_trend = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
            # trend = trend + residual_trend
            
        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x, trend  


class AttentionFormerLayer(nn.Module):
    """
    Autoformer decoder layer with the progressive decomposition architecture
    """
    def __init__(self, feed_forward,cross_attention,forth_attention, d_model, c_out, d_ff=None,
                 moving_avg=25, dropout=0.1, activation="relu"):
        super(AttentionFormerLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        #feed forward
        self.self_attention = feed_forward
        self.cross_attention = AutoCorrelationLayer(
                        FeedForward(example=feed_forward.inner_correlation),
                        d_model, feed_forward.n_heads)

       
        
        self.third_attention = cross_attention
        self.third_attention2 =  SAttentionLayer(cross_attention.configs,
                        cross_attention.segmented_v, cross_attention.segmented_ratio)
        
        self.forth_attention=forth_attention

        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)

        if isinstance(moving_avg, list):
            self.decomp1 = series_decomp_multi(moving_avg)
            self.decomp2 = series_decomp_multi(moving_avg)
            self.decomp3 = series_decomp_multi(moving_avg)
        else:
            self.decomp1 = series_decomp(moving_avg)
            self.decomp2 = series_decomp(moving_avg)
            self.decomp3 = series_decomp(moving_avg)

        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=3, stride=1, padding=1,
                                    padding_mode='circular', bias=False)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):

        x = x+self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )[0])

       
        x = x + self.dropout(self.cross_attention(
            x, x, x,
            attn_mask=cross_mask
        )[0])

        # cross = cross + self.dropout(self.cross_attention(
        #     cross, cross, cross,
        #     attn_mask=cross_mask
        # )[0])
  

        # x = x+ self.dropout(self.third_attention(
        #     x, cross, cross,
        #     attn_mask=cross_mask
        # )[0])

        x = x+ self.dropout(self.third_attention(
            x, x, x,
            attn_mask=cross_mask
        )[0])

        cross = cross+ self.dropout(self.third_attention2(
            cross, cross, cross,
            attn_mask=cross_mask
        )[0])

        x = x+ self.dropout(self.forth_attention(
            x, cross, cross,
            attn_mask=cross_mask
        )[0])
        
        # x = x+ self.dropout(self.third_attention(
        #     cross, x, x,
        #     attn_mask=cross_mask
        # )[0])
        

        # x, trend2 = self.decomp2(x)
        # y = x
        # y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        # y = self.dropout(self.conv2(y).transpose(-1, 1))
        # x, trend3 = self.decomp3(x + y)

        # residual_trend = trend1 + trend2 + trend3
        # residual_trend = trend2 + trend3
        # residual_trend = self.projection(residual_trend.permute(0, 2, 1)).transpose(1, 2)
        residual_trend=x

        return x, residual_trend
    

class Model(nn.Module):
    """
    FEDformer performs the attention mechanism on frequency domain and achieved O(N) complexity
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.version = configs.version
        self.mode_select = configs.mode_select
        self.modes = configs.modes
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        # Decomp
        kernel_size = configs.moving_avg
        print(kernel_size,'flag')
        if isinstance(kernel_size, list):
            self.decomp = series_decomp_multi(kernel_size)
        else:
            self.decomp = series_decomp(kernel_size)

        # Embedding
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.
        self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                  configs.dropout)
        self.dec_embedding = DataEmbedding_wo_pos(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                  configs.dropout)

        decoder_self_att = FeedForward(in_channels=configs.d_model,
                                            out_channels=configs.d_model,
                                            seq_len=self.seq_len//2+self.pred_len)
        decoder_self_att2 = FeedForward(example=decoder_self_att)
        print(configs.d_model,self.seq_len,configs.modes,configs.mode_select)
        # Encoder
        enc_modes = int(min(configs.modes, configs.seq_len//2))
        dec_modes = int(min(configs.modes, (configs.seq_len//2+configs.pred_len)//2))
        print('enc_modes: {}, dec_modes: {}'.format(enc_modes, dec_modes))

        # Decoder
        self.decoder = AttentionFormer(
            [
                AttentionFormerLayer(
                    AutoCorrelationLayer(decoder_self_att,configs.d_model, configs.n_heads),
                    SAttentionLayer(configs,configs.segmented_v, configs.segmented_ratio),
                    SAttentionLayer(configs,configs.segmented_v, configs.segmented_ratio),
                    configs.d_model,
                    configs.c_out,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
            ],
            norm_layer=my_Layernorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        x_enc= torch.cat([x_enc,x_dec[:,:self.label_len,:]],dim=1)

        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        seasonal_init, trend_init = self.decomp(x_enc)
        # seasonal_init, trend_init = self.decomp(input)
        # decoder input
 
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = F.pad(seasonal_init[:, -self.label_len:, :], (0, 0, 0, self.pred_len))
 
        seasonal_init = self.dec_embedding(seasonal_init, x_mark_dec)
        
        trend_cross = self.enc_embedding(trend_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(seasonal_init, trend_cross, x_mask=dec_self_mask, cross_mask=dec_enc_mask, trend=trend_init)
    
        dec_out = seasonal_part + trend_init

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]


if __name__ == '__main__':
    class Configs(object):
        ab = 0
        modes = 32
        mode_select = 'random'
        # version = 'Fourier'
        version = 'Wavelets'
        moving_avg = [12, 24]
        L = 1
        base = 'legendre'
        cross_activation = 'tanh'
        seq_len = 96
        label_len = 48
        pred_len = 96
        output_attention = True
        enc_in = 7
        dec_in = 7
        d_model = 16
        embed = 'timeF'
        dropout = 0.05
        freq = 'h'
        factor = 1
        n_heads = 8
        d_ff = 16
        e_layers = 2
        d_layers = 1
        c_out = 7
        activation = 'gelu'
        wavelet = 0

    configs = Configs()
    model = Model(configs)

    print('parameter number is {}'.format(sum(p.numel() for p in model.parameters())))
    enc = torch.randn([3, configs.seq_len, 7])
    enc_mark = torch.randn([3, configs.seq_len, 4])

    dec = torch.randn([3, configs.seq_len//2+configs.pred_len, 7])
    dec_mark = torch.randn([3, configs.seq_len//2+configs.pred_len, 4])
    out = model.forward(enc, enc_mark, dec, dec_mark)
    print(out)
