import torch
import torch.nn as nn 
import torch.nn.functional as F

class Expert(nn.Module):
    # expert = mlp layer
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class GatingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts):
        super().__init__()
        self.gate = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        weights = F.softmax(self.gate(x), dim=-1)  # [batch_size, num_experts]
        return weights

class NoisyTopKGating(nn.Module):
    def __init__(self, input_dim, num_experts, k=2, noise_std=1.0):
        super().__init__()
        self.k = k
        self.noise_std = noise_std
        self.w_gate = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        clean_logits = self.w_gate(x)  # [batch, num_experts]
        noise = torch.randn_like(clean_logits) * self.noise_std
        noisy_logits = clean_logits + noise

        topk_vals, topk_idx = torch.topk(noisy_logits, self.k, dim=-1)  # [batch, k]

        # Convert to one-hot for top-k positions
        mask = torch.zeros_like(clean_logits).scatter(1, topk_idx, 1.0)  # [batch, num_experts]

        # Softmax only over top-k
        topk_softmax = F.softmax(topk_vals, dim=-1)                      # [batch, k]
        gate_weights = torch.zeros_like(clean_logits)
        gate_weights.scatter_(1, topk_idx, topk_softmax)

        return gate_weights  # [batch, num_experts]

class MixtureOfExperts(nn.Module):
    def __init__(
        self, text_dim, image_dim, 
        hidden_dim, output_dim, 
        num_experts=4,
        k=2
    ):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.image_proj = nn.Linear(image_dim, hidden_dim)

        self.experts = nn.ModuleList([
            Expert(hidden_dim * 2, hidden_dim, output_dim) for _ in range(num_experts)
        ])

        # self.gating = GatingNetwork(hidden_dim * 2, num_experts)
        self.gating = NoisyTopKGating(hidden_dim * 2, num_experts, k=k)

    def forward(self, text_emb, image_emb):
        text_proj = self.text_proj(text_emb)
        image_proj = self.image_proj(image_emb)
        combined = torch.cat([text_proj, image_proj], dim=-1)  # [batch, hidden*2]

        gate_weights = self.gating(combined)  # [batch, num_experts]

        expert_outputs = []
        for i, expert in enumerate(self.experts):
            out = expert(combined)  # [batch, output_dim]
            weight = gate_weights[:, i].unsqueeze(-1)  # [batch, 1]
            expert_outputs.append(out * weight)

        output = sum(expert_outputs)  # [batch, output_dim]
        return output