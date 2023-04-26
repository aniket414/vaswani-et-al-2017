import torch
import torch.nn as nn
from transformer.positional import PositionalEncoding, FeedForward
from transformer.attention import MultiHeadAttention

class Encoder(nn.Module):
    """encoder"""
    def __init__(self, input_vocab_size, d_model, word_max_len, num_heads, d_k_embd,
                 layers, d_ff_hid, dropout=0.2, is_pos_embd=False, device="cuda") -> None:
        super().__init__()

        self.is_pos_embd = is_pos_embd  # true is positional embedding, false is positional encoding
        self.d_model = d_model
        self.d_k_embd = d_k_embd
        self.device = device

        # input&output embedding
        self.input_embedding = nn.Embedding(input_vocab_size, d_model)  # is a table of (input_vocab_size, d_model)

        # input&output positional encoding
        if self.is_pos_embd:
            self.input_pos_encoding = nn.Embedding(word_max_len, d_model) # is a table of (word_max_len, d_model)
        else:
            self.input_pos_encoding = PositionalEncoding(d_hid=d_model, n_position=word_max_len) # is a table of (1, word_max_len, d_model)
        self.dropout_input = nn.Dropout(dropout)

        # encoder layer
        self.encoder_layers = nn.ModuleList([EncoderLayer(num_heads=num_heads, d_k_embd=d_k_embd, d_model=d_model, \
                                                          d_ff_hid=d_ff_hid, dropout=dropout) for i in range(layers)])

    def forward(self, src_seq=None, src_mask=None):
        """
        Args:
            src_seq is (B, T) = (batch_size, input_size) or None,
            src_mask is (B, T) = (batch_size, 1, input_size) or None,
        """
        if src_seq is not None:
            inputs_embd = self.input_embedding \
                (src_seq) * self.d_model ** 0.5 # inputs_embd is (batch, input_size, d_model)
            if self.is_pos_embd:
                inputs_pos_en = self.input_pos_encoding \
                    (torch.arange(src_seq.shape[1], device=self.device))  # inputs_pos_en is (input_size, d_model)
            else:
                inputs_pos_en = self.input_pos_encoding(src_seq) # inputs_pos_en is (1, input_size, d_model)

            inputs = self.dropout_input(inputs_embd + inputs_pos_en)  # inputs is (batch, input_size, d_model)
            encoder_outs = inputs
            for enc_layer in self.encoder_layers:
                encoder_outs = enc_layer(enc_input=encoder_outs, slf_attn_mask=src_mask)  # encoder_outs is (batch, input_size, d_model)

        else:
            encoder_outs = None
        return encoder_outs


class EncoderLayer(nn.Module):
    def __init__(self, num_heads, d_k_embd, d_model, d_ff_hid, dropout=0.2) -> None:
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(
            d_model)  ## difference: move layer norm from behind multi-head attetion to before
        self.multi_head_attention = MultiHeadAttention(num_heads, d_k_embd, d_model, dropout=dropout)
        self.layer_norm_2 = nn.LayerNorm(d_model)  ## difference: move layer norm from behind FeedForward to before
        self.feed_forward = FeedForward(d_in=d_model, d_hid=d_ff_hid, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        """
        Args:
            enc_input is (B, T, C) = (batch_size, input_size, d_model)
            slf_attn_mask is (batch_size, 1, input_size)
        """
        enc_output = self.layer_norm_1(enc_input)  # enc_input is (batch_size, input_size, d_model)
        enc_output = enc_input + self.multi_head_attention(enc_output, enc_output, enc_output,
                                                           mask=slf_attn_mask)  # enc_output is (batch_size, input_size, d_model)
        out = enc_output + self.feed_forward(self.layer_norm_2(enc_output))  # (batch_size, input_size, d_model)
        return out