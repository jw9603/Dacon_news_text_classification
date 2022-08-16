# https://pytorch.org/docs/stable/generated/torch.unsqueeze.html#torch.unsqueeze    Tensor.unsqueeze(dim) = np.newaxis
# https://powerofsummary.tistory.com/158    register_buffer 이란?
# https://dololak.tistory.com/84    buffer 란? : 임시 저장공간 like 동영상 버퍼링, dynamic programming

import torch
import torch.nn as nn
import numpy as np
from model.Layers import EncoderLayer, DecoderLayer

def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)   # seq( b s ) --> pad_mask( b 1 s )

def get_subsequent_mask(seq):
    """For masking out the subsequent info."""
    sz_b, len_s = seq.size()
    subsequent_mask = (1-torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1
    )).bool()
    return subsequent_mask


class PostiionalEncoding(nn.Module):
    
    def __init__(self, d_hid, n_position=200):
        super(PostiionalEncoding, self).__init__()
        
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))
        # register_buffer: optimizer가 업데이트하지 않음. 하지만 GPU연산 가능한 layer.
        # 네트워크를 end2end 로 학습하려는데, 중간에 업데이트하지 않는 레이어 넣고싶을 때 사용하면 된다.
        # positional encoding 은 업데이트되는 값이 아니므로 이를 사용함.
        """ https://teamdable.github.io/techblog/PyTorch-Module
        
        >>> torch.nn.Module.register_buffer('running_mean', torch.zeros(num_features))  # example
        
        parameter가 말 그대로 buffer을 수행하기 위한 목적으로 활용한다.
        buffer도 state_dict에 저장되지만, backprop을 진행하지 않고 최적화에 사용되지 않는다는 의미이다.
        단순한 buffer로써의 역할을 맡는 본 모듈이다.
        """

    # return sinusoid FloatTensor table (1 S d)
    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        """sinusoid position encoding table"""
        
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2*(hid_j // 2) / d_hid) for hid_j in range(d_hid)]
        
        ## in original paper code,
        # angles = 1 / tf.pow(10000, (2* (i//2)) / tf.cast(d_model, tf.float32) )   # 임베딩 길이만큼 유니크한 값을 만든다.
        
        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])   # sin, cos 에 넣을 서로다른 radian 값 만들기 완료.
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
        # -1 ~ 1 사이의 비규칙적 분포값들로 mapping 된 sinusoid_table !
        
        return torch.FloatTensor(sinusoid_table).unsqueeze(0)   # batch용 새로운 차원만들기?
    
    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()   # clone() detach() ?
    

class Encoder(nn.Module):
    
    def __init__(
        self, n_src_vocab, d_word_vec, n_layers, n_head, d_k, d_v, d_model, d_inner, pad_idx, dropout=0.1, n_position=200, scale_emb=False
    ):
        super().__init__()
        
        self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=pad_idx)
        self.position_enc = PostiionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList(
            [EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout) for _ in range(n_layers)]     # Encoders
        )
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model
    
    def forward(self, src_seq, src_mask, return_attns=False):
        
        enc_slf_attn_list = []
        
        # input embedding, positional encoding
        enc_output = self.src_word_emb(src_seq)
        if self.scale_emb:
            enc_output *= self.d_model ** 0.5
        enc_output = self.dropout(self.position_enc(enc_output))
        enc_output = self.layer_norm(enc_output)
        
        # N encoders
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []
        
        # Output
        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,



class Decoder(nn.Module):

    def __init__(
        self, n_trg_vocab, d_word_vec, n_layers, n_head, d_k, d_v, d_model, d_inner, pad_idx, n_position=200, dropout=0.1, scale_emb=False
    ):
        super().__init__()
        
        self.trg_word_emb = nn.Enbedding(n_trg_vocab, d_word_vec, padding_idx=pad_idx)
        self.position_enc = PostiionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList(
            [DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout) for _ in range(n_layers)]
        )
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model


    def forward(self, trg_seq, trg_mask, enc_output, src_mask, return_attns=False):
        
        dec_slf_attn_list, dec_enc_attn_list = [], []
        
        # input embedding, positional encoding
        dec_output = self.trg_word_emb(trg_seq)
        if self.scale_emb:
            dec_output *= self.d_model ** 0.5
        dec_output = self.dropout(self.position_enc(dec_output))
        dec_output = self.layer_norm(dec_output)
        
        # N decoders
        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output, slf_attn_mask=trg_mask, dec_enc_attn_mask = src_mask
            )
            dec_slf_attn_list += [dec_slf_attn] if return_attns else []
            dec_enc_attn_list += [dec_enc_attn] if return_attns else []
        
        # output
        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output,


class _None_Used(nn.Module):
    
    def __init__(
        self, n_src_vocab, src_pad_idx, d_word_vec=512, d_model=512, d_inner=2048, n_layers=12, n_head=8,
        d_k=64, d_v=64, dropout=0.1, n_position=200, 
        scale_emb_or_prj='prj'
    ):
        ## What's different with Transformer's args
        # n_trg_vocab, trg_pad_idx, trg_emb_prj_weight_sharing, emb_src_trg_weight_sharing
        super().__init__()
        
        self.src_pad_idx = src_pad_idx
        """
        # Sharing the same matrix between the two embedding layers and the pre-softmax linear transformation.
        # In the embedding layers, thoese weights were multiplied by \sqrt{d_model}.
        #
        # Options (made by @github. Huang)
        #   'emb': multiply \sqrt{d_model} to embedding output
        #   'prj': multiply (\sqrt{d_model} ^ -1) to linear projection output
        #   'none': no multiplication
        
        assert scale_emb_or_prj in ['emb', 'prj', 'none'], 'Model arg \'scale_emb_or_prj\' : emb, prj, none'
        scale_emb = (scale_emb_or_prj == 'emb') if trg_emb_prj_weight_sharing else False
        self.scale_prj = (scale_emb_or_prj == 'prj') if trg_emb_prj_weight_sharing else False
        
        이 부분 없어도 되는게 맞나?
        """
        self.d_model = d_model
        
        self.encoder = Encoder(
            n_src_vocab=n_src_vocab, d_word_vec=d_word_vec,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v, d_model=d_model,
            d_inner=d_inner, pad_idx=src_pad_idx, dropout=dropout, n_position=200, scale_emb=None
        )
        
        self.src_word_prj = nn.Linear(d_model, n_src_vocab, bias=False)
        
         # 2차원 이상 param에 대해서 xavier 초기화 적용.
        for p in self.parameters():
            if p.dim() >1:
                nn.init.xavier_uniform(p)
        
        assert d_model == d_word_vec, 'In order to make use of residual connections, the dimensions of all module ouputs shall be the same.'
        
    
    def forward(self, src_seq):
        
        src_mask = get_pad_mask(src_seq, self.src_pad_idx)
        
        enc_output, *_ = self.encoder(src_seq, src_mask)
        seq_logit = self.src_word_prj(enc_output)
        # if self.scale_prj:
        #     seq_logit *= self.d_model ** -0.5
        
        return seq_logit.view(-1, seq_logit.size(2))

class Transformer(nn.Module):
    
    def __init__( 
        self, n_src_vocab, n_trg_vocab, src_pad_idx, trg_pad_idx, d_word_vec=512, d_model=512, d_inner=2048,
        n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1, n_position=200,
        trg_emb_prj_weight_sharing=True, emb_src_trg_weight_sharing=True,
        scale_emb_or_prj='prj'
    ):
        super().__init__()

        self.src_pad_idx, self.trg_pad_idx = src_pad_idx, trg_pad_idx
        
        # Sharing the same matrix between the two embedding layers and the pre-softmax linear transformation.
        # In the embedding layers, thoese weights were multiplied by \sqrt{d_model}.
        #
        # Options (made by @github. Huang)
        #   'emb': multiply \sqrt{d_model} to embedding output
        #   'prj': multiply (\sqrt{d_model} ^ -1) to linear projection output
        #   'none': no multiplication

        assert scale_emb_or_prj in ['emb', 'prj', 'none']
        scale_emb = (scale_emb_or_prj == 'emb') if trg_emb_prj_weight_sharing else False
        self.scale_prj = (scale_emb_or_prj == 'prj') if trg_emb_prj_weight_sharing else False
        self.d_model = d_model
        
        self.encoder = Encoder(
            n_src_vocab=n_src_vocab, d_word_vec=d_word_vec, 
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v, d_model=d_model, 
            d_inner=d_inner, pad_idx=src_pad_idx, dropout=dropout, n_position=200, scale_emb=scale_emb
        )
        
        self.decoder = Decoder(
            n_trg_vocab=n_trg_vocab, n_position=n_position,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=trg_pad_idx, dropout=dropout, scale_emb=scale_emb
        )
        
        self.trg_word_prj = nn.Linear(d_model, n_trg_vocab, bias=False)
        
        # 2차원 이상 param에 대해서 xavier 초기화 적용.
        for p in self.parameters():
            if p.dim() >1:
                nn.init.xavier_uniform(p)
        
        assert d_model == d_word_vec, 'To make use of the residual connections, the dimensions of all module outputs shall be the same.'
        
        # softmax 이후 word emb 공유
        if trg_emb_prj_weight_sharing:
            # Share the weight between target word embedding & last dense layer
            self.trg_word_prj.weight = self.decoder.trg_word_emb.weight
           
        # enc dec weight 공유
        if emb_src_trg_weight_sharing:
            self.encoder.src_word_emb.weight = self.decoder.trg_word_emb.weight


    def forward(self, src_seq, trg_seq):
        
        src_mask = get_pad_mask(src_seq, self.src_pad_idx)  # ( b 1 seq_max_len ) : mask를 한번 더 감싸줬다. 왜?
        trg_mask = get_pad_mask(trg_seq, self.trg_pad_idx) & get_subsequent_mask(trg_seq)
        
        enc_output, *_ = self.encoder(src_seq, src_mask)
        dec_output, *_ = self.decoder(trg_seq, trg_mask, enc_output, src_mask)
        seq_logit = self.trg_word_prj(dec_output)   # Linear(d_model -> vocab_size)
        if self.scale_prj:
            seq_logit *= self.d_model ** -0.5   # seq_logit 값을 scaling 해줬다.
            
        return seq_logit.view(-1, seq_logit.size(2))    # seq_logit (B S V)  -> ((B S) V)