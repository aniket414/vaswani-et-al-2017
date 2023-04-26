import torch
import torch.nn as nn
from transformer.positional import PositionalEncoding, FeedForward
from transformer.attention import MultiHeadAttention

class Decoder(nn.Module):
    def __init__(self, output_vocab_size, d_model, word_max_len, num_heads, d_k_embd,
                 layers, d_ff_hid, dropout=0.2, is_pos_embed=False, device="cuda") -> None:
        super().__init__()
        self.d_model = d_model
        self.layers = layers
        self.is_pos_embd = is_pos_embed  # true is positional embedding, false is positional encoding
        self.device = device

        # output embedding
        self.output_embedding = nn.Embedding(output_vocab_size, d_model)  # is a table of (output_vocab_size, d_model)

        # output positional encoding
        if is_pos_embed:
            self.output_pos_encoding = nn.Embedding(word_max_len, d_model)  # is a table of (word_max_len, d_model)
        else:
            self.output_pos_encoding = PositionalEncoding(d_hid=d_model,
                                                          n_position=word_max_len)  # is a table of (1, word_max_len, d_model)
        self.dropout_output = nn.Dropout(dropout)

        # decoder
        self.decoder_layers = nn.ModuleList([DecoderLayer(num_heads=num_heads, d_k_embd=d_k_embd, d_model=d_model, \
                                                          d_ff_hid=d_ff_hid, dropout=dropout) for i in range(layers)])

    def forward(self, trg_seq, trg_mask, src_mask=None, enc_output=None):
        """
        Args:
            trg_seq is (batch_size, output_size)
            trg_mask is (output_size, output_size) or (batch_size, output_size, output_size)
            src_mask is (batch_size, 1, input_size) or None
            enc_output is (batch_size, input_size, d_model),
        """
        outputs_embd = self.output_embedding(
            trg_seq) * self.d_model ** 0.5  # outputs_embd is (batch, output_size, d_model)
        if self.is_pos_embd:
            outputs_pos_en = self.output_pos_encoding(
                torch.arange(trg_seq.shape[1], device=self.device))  # outputs_pos_en is (output_size, d_model)
        else:
            outputs_pos_en = self.output_pos_encoding(trg_seq)  # outputs_pos_en is (1, output_size, d_model)

        outputs = self.dropout_output(outputs_embd + outputs_pos_en)  # outputs is (batch, output_size, d_model)
        decoder_ints = outputs
        for dec_layer in self.decoder_layers:
            decoder_ints = dec_layer(dec_input=decoder_ints, enc_output=enc_output,
                                     slf_attn_mask=trg_mask, dec_enc_attn_mask=src_mask)

        return decoder_ints  # decoder_ints is (batch, output_size, d_model)


class DecoderLayer(nn.Module):
    def __init__(self, num_heads, d_k_embd, d_model, d_ff_hid, dropout=0.2) -> None:
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.masked_multi_head_attention = MultiHeadAttention(num_heads, d_k_embd, d_model, dropout=dropout)
        self.layer_norm_2 = nn.LayerNorm(d_model)
        self.dec_enc_multi_head_attention = MultiHeadAttention(num_heads, d_k_embd, d_model, dropout=dropout)
        self.layer_norm_3 = nn.LayerNorm(d_model)
        self.feed_forward = FeedForward(d_in=d_model, d_hid=d_ff_hid, dropout=dropout)

    def forward(self, dec_input, enc_output, slf_attn_mask=None, dec_enc_attn_mask=None):
        """
        Args:
            dec_input is (B, T, C) = (batch_size, output_size, d_model).
            enc_output is (B, T, C) = (batch_size, input_size, d_model).
            slf_attn_mask is (output_size, output_size) or (batch_size, output_size, output_size).
            dec_enc_attn_mask is (batch_size, 1, input_size).
        """
        dec_output = self.layer_norm_1(dec_input)  # (batch_size, output_size, d_model)
        dec_output = dec_input + self.masked_multi_head_attention(dec_output, dec_output, dec_output,
                                                                  mask=slf_attn_mask)  # (batch_size, output_size, d_model)
        dec_output = dec_output + self.dec_enc_multi_head_attention(self.layer_norm_2(dec_output), enc_output,
                                                                    enc_output,
                                                                    mask=dec_enc_attn_mask)  # (batch_size, output_size, d_model)
        out = dec_output + self.feed_forward(self.layer_norm_3(dec_output))  # (batch_size, output_size, d_model)
        return out