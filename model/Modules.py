# https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Modules.py  의 코드를 전적으로 참고하여 모델을 구성하였음.
# https://anweh.tistory.com/21  를 참고하여 torch api를 익힘.
# https://pytorch.org/docs/stable/generated/torch.matmul.html     torch.matmul()
# https://pytorch.org/docs/stable/generated/torch.Tensor.masked_fill.html     torch.Tensor.masked_fill()
# https://pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html#torch.nn.functional.softmax  torch.nn.functional.softmax()
# https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html   torch.nn.Dropout()

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention"""
    
    def __init__(self, scale, attn_dropout=0.1):
        # super 을 통해 nn.Module의 init 메소드에 저장되어있는 정보를 상속 받아야 forward를 통한 다양한 클래스의 기능을 수행가능.
        super().__init__()
        
        # 그 외의 내가 만든 모델의 forward 연산에 필요한 것들을 객체의 변수로 추가해준다.
        self.scale = scale          # scaling factor? root(d_k)
        self.dropout = nn.Dropout(attn_dropout) # 출력 node 를 꺼버린다.
     
    def forward(self, q, k, v, mask=None):
        """
        q, k, v     (B N S d_model/N) 
        mask        ()
        
        q k 로 scaled dot product attention 계산을 해 v에 반영한 결과를 return 한다.
        
        """
        attn = torch.matmul(q / self.scale, k.transpose(-1,-2))     # k( B, N, S, d ) -> k(B, N, d, S), q( B N d ) -> k(b n d s)*q(b n d s(repeat)) => attn(b n )
        
        if mask is not None:
            attn = attn.masked_fill(mask==0, -1e9)  # mask에서 0인 값들을 -1e-9 값으로 대체한다.
        
        attn = self.dropout(F.softmax(attn, dim = -1))  # 마지막 차원을 softmax 한 결과를 dropout   ( B N d S )인 attn 의 S 차원
        output = torch.matmul(attn, v)
        
        return output, attn     # value 에 먹인 값, attention score 두개를 반환한다.