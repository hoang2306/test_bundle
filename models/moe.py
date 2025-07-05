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

class MixtureOfExperts(nn.Module):
    def __init__(self, text_dim, image_dim, hidden_dim, output_dim, num_experts):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.image_proj = nn.Linear(image_dim, hidden_dim)

        self.experts = nn.ModuleList([
            Expert(hidden_dim * 2, hidden_dim, output_dim) for _ in range(num_experts)
        ])

        self.gating = GatingNetwork(hidden_dim * 2, num_experts)

def forward(self, text_emb, image_emb):
        text_proj = self.text_proj(text_emb)         # [batch, hidden_dim]
        image_proj = self.image_proj(image_emb)      # [batch, hidden_dim]

        combined = torch.cat([text_proj, image_proj], dim=-1)  # [batch, hidden_dim*2]

        gate_weights = self.gating(combined)         # [batch, num_experts]

        expert_outputs = torch.stack([expert(combined) for expert in self.experts], dim=1)  # [batch, num_experts, output_dim]

        output = torch.sum(gate_weights.unsqueeze(-1) * expert_outputs, dim=1)  # [batch, output_dim]
        return output