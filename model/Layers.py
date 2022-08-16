# https://velog.io/@minchoul2/nn.Module-init%EC%8B%9C-super.init-%ED%95%B4%EC%95%BC%ED%95%98%EB%8A%94-%EC%9D%B4%EC%9C%A0    nn.Module super().__init__() 이유

import torch.nn as nn
import torch
from model.SubLayers import MultiHeadAttention, PositionwiseFeedForward

class EncoderLayer(nn.Module):
    
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout = 0.1):
        super(EncoderLayer, self).__init__()    # grad, forward 등 계산에 필요한 여러 변수들을 미리 선언해주는 super().__init__() / 직접 모듈을 만드는 경우 이를 반드시 선언해줘야 정상적인 모듈이 된다.
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout)
        
    # self attention -- ffn 
    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn
    
    
class DecoderLayer(nn.Module):
    
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
    
    # self attention - enc attention - ffn
    def forward(self, dec_input, enc_output, slf_attn_mask=None, dec_enc_attn_mask=None):
        
        dec_output, dec_slf_attn = self.slf_attn(dec_input, dec_input, dec_input, mask=slf_attn_mask)   # self attn layer 통과
        dec_output, dec_enc_attn = self.enc_attn(dec_output, enc_output, enc_output, mask=dec_enc_attn_mask)    # dec enc attn layer 통과
        dec_output = self.pos_ffn(dec_output)   # ffn 통과
        return dec_output, dec_slf_attn, dec_enc_attn