# https://paperswithcode.com/method/layer-normalization     Layer Nromalization
# https://jimmy-ai.tistory.com/122      contiguous() 사용하는 이유 --> transpose 후 메모리 주소 align 시키기 위해. == compute bottleneck 제거

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from model.Modules import ScaledDotProductAttention

# embedded input(q,v,k) 받으면 -> q, v, k 계산을 해서 -> Scaled_dotproduct_attn 계산을 거쳐 -> dropout먹이고 -> residual(add, norm) 까지 수행하는 SubLayer
# input을 받는거랑, residual(add and norm) 이 다른거구나.
class MultiHeadAttention(nn.Module):
    """
    
    """
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        
        super().__init__()
        
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        
        self.w_q = nn.Linear(d_model, n_head * d_k, bias = False)
        self.w_k = nn.Linear(d_model, n_head * d_k, bias = False)
        self.w_v = nn.Linear(d_model, n_head * d_v, bias = False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)
        
        self.attention = ScaledDotProductAttention(scale=d_k ** 0.5)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        
        
    def forward(self, q, k, v, mask=None):
        
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        
        residual = q    # 여기서 q는 지금 MHA 의 순수한 input이다. ( B L d_model )
        
        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_q(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_k(k).view(sz_b, len_q, n_head, d_k)
        v = self.w_v(v).view(sz_b, len_q, n_head, d_v)
        
        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1,2), k.transpose(1,2), v.transpose(1,2)
        
        if mask is not None:
            mask = mask.unsqueeze(1)    # For head axis broadcasting
        
        q, attn = self.attention(q, k, v, mask=mask)
        
        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1,2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual   # add
        
        q = self.layer_norm(q)  # norm
        
        return q, attn
        
# 2 affine layer + dropout + residual + layerNorm
class PositionwiseFeedForward(nn.Module):
    
    def __init__(self, d_in, d_hid, dropout=0.1):
        
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)   # position-wise?
        self.w_2 = nn.Linear(d_hid, d_in)   # position-wise?
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        
        
    def forward(self, x):
        
        residual = x
        
        x = self.w_2(F.relu(self.w_1(x)))   # linear -> relu -> linear
        x = self.dropout(x)
        x += residual   # add
        #hey
        x = self.layer_norm(x)  # and norm !
        
        return x
        