import torch
import torch.nn as nn

class CrossAttentionFusion(nn.Module):
    def __init__(self, embedding_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=embedding_dim, 
            num_heads=num_heads, 
            dropout=dropout, 
            batch_first=True
        )
        self.norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: [batch_size, modality=2, embedding_dim]
        """
        # Query = modality 1, Key/Value = modality 2
        q = x[:, 0:1, :]  # [B, 1, D]
        k = x[:, 1:2, :]  # [B, 1, D]
        v = x[:, 1:2, :]  # [B, 1, D]

        out12, _ = self.attn(q, k, v)  # [B, 1, D]

        # Query = modality 2, Key/Value = modality 1
        q = x[:, 1:2, :]
        k = x[:, 0:1, :]
        v = x[:, 0:1, :]
        out21, _ = self.attn(q, k, v)  # [B, 1, D]

        # concat to get [B, 2, D]
        fused = torch.cat([out12, out21], dim=1)

        # residual + norm
        fused = self.norm(x + self.dropout(fused))

        return fused  # [B, 2, D]

# if __name__ == "__main__":
#     batch_size = 5
#     embedding_dim = 16
#     x = torch.randn(batch_size, 2, embedding_dim)
#     print(f'x shape: {x.shape}')

#     fusion = CrossAttentionFusion(embedding_dim)
#     out = fusion(x)
#     print(f'out shape: {out.shape}')  # expected: [5, 2, 16]
