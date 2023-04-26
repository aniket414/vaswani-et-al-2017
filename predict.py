import argparse
import torch
import torch.nn as nn
from transformer.transformer import Transformer
from dataloader.constants import BOS_WORD, EOS_WORD
from dataloader.wmt16_dataset import prepare_dataloaders

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def translate(model_path, args):
    checkpoint = torch.load(model_path, map_location=DEVICE)
    _, _, _, _, _, src_pad_idx, trg_pad_idx, trg_bos_idx, trg_eos_idx, src_vocab_size, trg_vocab_size, unk_idx, src_stoi, trg_itos, _, _ \
        = prepare_dataloaders(100, 3, 128, DEVICE)

    model = Transformer(
            input_vocab_size=src_vocab_size,
            output_vocab_size=trg_vocab_size,
            d_model=512,
            word_max_len=300,
            num_heads=8,
            d_k_embd=512//8,
            layers=6,
            d_ff_hid=512*4,
            src_pad_idx=src_pad_idx,
            trg_pad_idx=trg_pad_idx,
            trg_eos_idx=trg_eos_idx,
            trg_bos_idx=trg_bos_idx,
            dropout=0.1,
            is_pos_embed=False,
            beam_size=5,
            alpha=0.7,
            device=DEVICE)
    model.load_state_dict(checkpoint['model'])
    print('[Info] Trained model state loaded.')
    print("German input: ", "[", len(args.inp), "]", args.inp)
    src_seq = [src_stoi.get(word, unk_idx) for word in args.inp]
    pred_seq = model.translate_sentence(torch.LongTensor([src_seq]).to(DEVICE))
    pred_line = ' '.join(trg_itos[idx] for idx in pred_seq)
    pred_line = pred_line.replace(BOS_WORD, '').replace(EOS_WORD, '')
    # print("English output: ", "[", len(pred_seq), "]", pred_seq)
    print("English output: ", "[", len(pred_seq), "]", pred_line)

def main():
    parser = argparse.ArgumentParser(description='Machine Translation')
    parser.add_argument('--inp', type=str, default='Ich bin ein Student der Deep-Learning-Klasse', help="YOUR_INPUT")
    args = parser.parse_args()
    translate(model_path="./output/model.chkpt", args=args)

if __name__ == "__main__":
    main()