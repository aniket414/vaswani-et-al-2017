import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.2) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        """
        Args:
            q is (batch_size, n_head, input/output_size, d_k)
            k is (batch_size, n_head, input/output_size, d_k)
            v is (batch_size, n_head, input/output_size, d_k)
            mask is (batch_size, 1, input_size) or (output_size, output_size) or (batch_size, output_size, output_size).
        """
        assert q.shape[3 ] == k.shape[3] and q.shape[3 ] == v.shape[3] and k.shape[2 ] == v.shape[2], "the dimensions of q, k, v in ScaledDotProductAttention are not matched!"

        wei = q @ k.transpose(-2, -1) * q.shape[-1] **-0.5 # wei is (batch, n_head, input/output_size, input/output_size)
        if mask is not None:
            wei = wei.masked_fill(mask==0, float('-inf')) # wei is (batch, n_head, input/output_size, input/output_size)
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = wei @ v  # out is (batch, n_head, input/output_size, d_k)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, d_embd, d_model, dropout=0.2) -> None:
        super().__init__()

        self.d_model = d_model
        self.d_k = d_embd
        self.n_heads = num_heads
        self.liner_key = nn.Linear(self.d_model, num_heads * self.d_k, bias=True)
        self.liner_query = nn.Linear(self.d_model, num_heads * self.d_k, bias=True)
        self.liner_value = nn.Linear(self.d_model, num_heads * self.d_k, bias=True)

        self.attn_head = ScaledDotProductAttention(dropout=dropout)
        self.linear = nn.Linear(num_heads * self.d_k, d_model, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        """
        Args:
            q is (B, T, C) = (batch_size, input/output_size, d_model)
            k is (B, T, C) = (batch_size, input/output_size, d_model)
            v is (B, T, C) = (batch_size, input/output_size, d_model)
            mask is (B, T, C) = (batch_size, 1, input_size) or (output_size, output_size) or (batch_size, output_size, output_size).
        """
        batch_size = q.size(0)
        len_q, len_k, len_v = q.size(1), k.size(1), v.size(1)

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.liner_query(q).view(batch_size, len_q, self.n_heads, self.d_k)   # (batch, input/output_size, n_heads, d_k)
        k = self.liner_key(k).view(batch_size, len_k, self.n_heads, self.d_k)   # (batch, input/output_size, n_heads, d_k)
        v = self.liner_value(v).view(batch_size, len_v, self.n_heads, self.d_k)   # (batch, input/output_size, n_heads, d_k)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1 ,2), k.transpose(1 ,2), v.transpose(1 ,2)

        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(0)   # For head axis broadcasting.
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)   # For head axis broadcasting.
            else:
                raise RuntimeError("The shape of mask is not correct!")

        out = self.attn_head(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        out = out.transpose(1, 2).contiguous().view(batch_size, len_q, -1)  # (batch, input/output_size, d_model)
        out = self.linear(out) # (batch, input/output_size, d_model)
        out = self.dropout(out)
        return out  # (batch, input/output_size, d_model)