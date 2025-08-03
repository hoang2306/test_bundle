import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class WeCopy(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = F.leaky_relu(self.fc1(x))
        a = self.fc2(h)
        return a


class PWC(nn.Module):
    def __init__(self, n_users, input_dim, hidden_dim, device, base, w1):
        super().__init__()
        self.weco_a = WeCopy(input_dim, hidden_dim)
        self.weco_b = WeCopy(input_dim, hidden_dim)
        self.weco_c = WeCopy(input_dim, hidden_dim)
        self.theta = nn.Parameter(nn.init.xavier_normal_(
            torch.tensor(np.random.randn(n_users, 3), dtype=torch.float32, requires_grad=True).to(device)))
        self.w1, self.w2 = w1, 1 - w1
        self.base = base
        self.last_att = self.theta.data

    def update_w(self, epoch):
        self.w1, self.w2 = update_weights(self.w1, self.w2, epoch, base=self.base)
        self.theta.data = self.last_att

    def forward(self, a, b, c):
        att_a = self.weco_a(a)
        att_b = self.weco_b(b)
        att_c = self.weco_c(c)
        f_att = torch.cat((att_a, att_b, att_c), dim=1)
        # fusion
        _att2 = self.w1 * f_att + self.w2 * self.theta
        _att2 = F.softmax(_att2, dim=1)
        self.last_att = _att2

        _att2 = torch.unsqueeze(_att2, dim=1)
        fused_representation = torch.cat((_att2[:, :, 0] * a, _att2[:, :, 1] * b, _att2[:, :, 2] * c), dim=1)
        return fused_representation


import math

def update_weights(w1, w2, epoch, base=0.9):
    """
    Update the weights exponentially over training epochs.

    Parameters:
    w1 (float): Initial weight for the first network.
    w2 (float): Initial weight for the second network.
    epoch (int): Current epoch number.
    total_epochs (int): Total number of training epochs.
    base (float): Base of the exponential, determines the rate of weight update.
    Smaller base will result in a faster transfer of weight from the first network to the second.

    Returns:
    tuple: Updated weights (w1, w2).
    """
    # Calculate the decay factor for the current epoch
    decay_factor = math.pow(base, epoch)

    # Update w1 and w2 based on the decay factor
    w1_updated = w1 * decay_factor
    w2_updated = w2

    # Normalize the weights to ensure they sum up to 1
    weight_sum = w1_updated + w2_updated
    w1_normalized = w1_updated / weight_sum
    w2_normalized = w2_updated / weight_sum

    return w1_normalized, w2_normalized


print('import pwc done')