import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseTopKMoE(nn.Module):
    def __init__(self, d, num_experts=16, k=2, capacity_factor=1.25):
        super().__init__()
        assert k >= 1 and k <= num_experts
        self.d = d
        self.E = num_experts
        self.k = k
        self.capacity_factor = capacity_factor  # not enforced in this simple impl
        expert_hidden = 4*d
        gate_hidden = 2*d

        self.gate = nn.Sequential(
            nn.Linear(3*d, gate_hidden),
            nn.ReLU(),
            nn.Linear(gate_hidden, num_experts)
        )
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(3*d, expert_hidden),
                nn.ReLU(),
                nn.Linear(expert_hidden, d)
            ) for _ in range(num_experts)
        ])
        self.output_proj = nn.Linear(3*d, d)

    def forward(self, x1, x2, x3):
        x = torch.cat([x1, x2, x3], dim=-1)  # (B,3d)
        logits = self.gate(x)                # (B,E)
        topk_vals, topk_idx = torch.topk(logits, self.k, dim=-1)  # (B,k)
        # compute softmax over selected topk vals
        topk_weights = F.softmax(topk_vals, dim=-1)  # (B,k)

        B = x.shape[0]
        # compute outputs for selected experts only
        # naive implementation: compute all experts then select (works but computes E experts)
        expert_outs = torch.stack([ex(x) for ex in self.experts], dim=1)  # (B,E,d)
        print(f'expert outputs: {expert_outs.shape}')

        # gather the top-k expert outputs
        idx = topk_idx.unsqueeze(-1).expand(-1, -1, self.d)  # (B,k,d)
        selected = torch.gather(expert_outs, dim=1, index=idx)  # (B,k,d)

        weights = topk_weights.unsqueeze(-1)  # (B,k,1)
        y = (weights * selected).sum(dim=1)   # (B,d)

        # optional residual
        y = 0.5 * y + 0.5 * self.output_proj(x)
        return y, logits  # return logits for potential load-balancing loss

# test
if __name__ == "__main__":
    model = SparseTopKMoE(d=128, num_experts=4, k=2)
    x1 = torch.randn(32, 128)
    x2 = torch.randn(32, 128)
    x3 = torch.randn(32, 128)
    y, logits = model(x1, x2, x3)
    print(y.shape, logits.shape)

    hidden_size = 64
    expert_num = 4
    experts = [nn.Parameter(torch.Tensor(1, hidden_size * 3), requires_grad=True) for _ in range(expert_num)]
    print(experts[0].shape)

    x_cat = torch.randn(3, hidden_size * 3) # [3, 3*d]
    # experts[0 # [1, 3*d]
    x_ = (x_cat * experts[0]).unsqueeze(2) 
    print(f'x_ shape: {x_.shape}')