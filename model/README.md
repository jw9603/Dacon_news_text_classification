# model

> https://github.com/jadore801120/attention-is-all-you-need-pytorch   해당 transformer 모델의 인코더를 참고하였음.<br/>
> Moduels 부터 가장 작은 단위로 구현하여 Models 까지 구현하였음.

- Models.py :       Encoder, (Decoder) 구현<br/>
- Layers.py :       EncoderLayer, (DecoderLayer) 구현<br/>
- SubLayers.py :    Multi-head Attention, Position-wise Feed Forward 구현<br/>
- Modules.py :      ScaledDotProductAttention 구현<br/><br/>

- Optim.py :        lr_schedueling 을 위한 wrapper class<br/>
- \_\_init\_\_.py :     모든 파일들 호출