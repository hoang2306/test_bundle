from modules.transformer import TransformerEncoder
from models.cross_attention import Cross_Attn
import torch

if __name__ == '__main__':
    # encoder = TransformerEncoder(300, 4, 2)
    # # x = torch.tensor(torch.rand(20, 2, 300))
    # x = torch.randn(20, 2, 300)
    # print(f'x shape: {x.shape}')
    # print(f'x encoded shape: {encoder(x).shape}')

    cross_encoder = Cross_Attn()

    print(f'trainable params: {sum(p.numel() for p in cross_encoder.parameters() if p.requires_grad)}')

    # print(cross_encoder)
    x_ = torch.rand(10, 64) # bs=10, dim=64
    out, out_f = cross_encoder(x_, x_)
    print(f'out shape: {out.shape}')
    print(f'out f shape: {out_f.shape}')