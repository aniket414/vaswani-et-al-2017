import torch
import torch.nn as nn
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()
        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]
        sinusoid_table = torch.FloatTensor([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = torch.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = torch.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        return sinusoid_table.unsqueeze(0)  # return is (1, n_position, d_hid)

    def forward(self, x):
        return self.pos_table[:, :x.size(1), :].clone().detach() # return is (1, x.size(1), d_hid)

class FeedForward(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hid),
            nn.ReLU(),
            nn.Dropout(dropout), 
            nn.Linear(d_hid, d_in),
            nn.Dropout(dropout),   
        )
        
    def forward(self, x):
        """
        Args:
            x is (B, T, C) = (batch_size, input/output_size, d_model)
        """
        out = self.net(x)
        return out # (batch_size, input/output_size, d_model)
