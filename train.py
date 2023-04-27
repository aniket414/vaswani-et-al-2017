import os
import argparse
import torch
import torch.optim as optim
import time
import math

from torchtext.data import bleu_score
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torch.nn as nn
from transformer.transformer import Transformer
from transformer.optim import ScheduledOptim
from dataloader.wmt16_dataset import prepare_dataloaders, tokenize_en


class TrainerTransformer:
    def __init__(self, args):
        self.batch_size = args.batch_size  # 128
        self.word_max_len = 300  ## word_max_len must be larger than output_size/input_size
        self.d_model = args.emb_size  # 512
        self.d_ff_hid = 512 * 4
        self.num_heads = args.num_heads  # 8
        self.d_k_embd = self.d_model // self.num_heads
        self.layers = 6
        self.dropout = args.dropout  # 0.1
        self.epoch = args.epoch
        self.lr_mul = 0.5  # 2.0
        self.n_warmup_steps = 4000  # 128000
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.output_dir = "./output"
        self.tb_writer = SummaryWriter(log_dir=os.path.join(self.output_dir, 'tensorboard'))
        self.label_smoothing = True
        self.max_len = 100
        self.min_freq = args.n_layers  # 3
        self.is_pos_embd = False

        # for translate
        self.alpha = 0.7
        self.beam_size = 5
        self.max_seq_len = 100

        # create WMT16 dataset
        self.training_data, self.validation_data, self.test_dataset, self.train_datasets_len, self.val_datasets_len, self.src_pad_idx, self.trg_pad_idx, \
            self.trg_bos_idx, self.trg_eos_idx, self.src_vocab_size, self.trg_vocab_size, self.unk_idx, \
            self.src_stoi, self.trg_itos, self.BOS_WORD, self.EOS_WORD \
            = prepare_dataloaders(self.max_len, self.min_freq, self.batch_size, self.device)
        self.train_datasets_iter = iter(self.training_data)
        self.val_datasets_iter = iter(self.validation_data)

        # create model
        self.model = Transformer(
            input_vocab_size=self.src_vocab_size,
            output_vocab_size=self.trg_vocab_size,
            d_model=self.d_model,
            word_max_len=self.word_max_len,
            num_heads=self.num_heads,
            d_k_embd=self.d_k_embd,
            layers=self.layers,
            d_ff_hid=self.d_ff_hid,
            src_pad_idx=self.src_pad_idx,
            trg_pad_idx=self.trg_pad_idx,
            trg_eos_idx=self.trg_eos_idx,
            trg_bos_idx=self.trg_bos_idx,
            dropout=self.dropout,
            is_pos_embed=self.is_pos_embd,
            beam_size=self.beam_size,
            alpha=self.alpha,
            device=self.device)
        self.model = nn.DataParallel(self.model) # for using multiple gpus
        self.model.to(self.device)

        # create optimizer
        self.optimizer = ScheduledOptim(
            optim.Adam(self.model.parameters(), betas=(0.9, 0.98), eps=1e-09),
            self.lr_mul, self.d_model, self.n_warmup_steps)

    def train(self):
        valid_losses = []
        epoch_load = tqdm(range(self.epoch), desc="Epochs", leave=False)
        count = 1
        for epoch_i in epoch_load:

            # train epoch
            print('[ Epoch', epoch_i, ']')
            start = time.time()
            train_loss, train_accu = self.train_epoch()
            train_ppl = math.exp(min(train_loss, 100))
            lr = self.optimizer._optimizer.param_groups[0]['lr']
            self.print_performances('Training', train_ppl, train_accu, start, lr)

            # eval epoch
            start = time.time()
            valid_loss, valid_accu = self.eval_epoch()
            valid_ppl = math.exp(min(valid_loss, 100))
            self.print_performances('Validation', valid_ppl, valid_accu, start, lr)

            valid_losses += [valid_loss]
            checkpoint = {'epoch': epoch_i, 'model': self.model.state_dict()}
            if valid_loss <= min(valid_losses):
                torch.save(checkpoint, os.path.join(self.output_dir, 'model.chkpt'))
                print('- [Info] The checkpoint file has been updated.')

            # write to tensorboard
            self.tb_writer.add_scalars('ppl', {'train': train_ppl, 'val': valid_ppl}, epoch_i)
            self.tb_writer.add_scalars('accuracy', {'train': train_accu * 100, 'val': valid_accu * 100}, epoch_i)
            self.tb_writer.add_scalar('learning_rate', lr, epoch_i)

            epoch_load.set_postfix(count=count)
            count += 1

    def train_epoch(self):
        self.model.train()
        total_loss, n_word_total, n_word_correct = 0, 0, 0
        print("train epoch is processing: ")

        for i in range(self.train_datasets_len // self.batch_size):
            # prepare data
            batch = next(self.train_datasets_iter)
            src_seq = self._patch_src(batch['src'], self.src_pad_idx).to(
                self.device)  # src_seq is (batch_size, output_size-1)
            trg_seq, gold = map(lambda x: x.to(self.device), self._patch_trg(batch['trg'], self.trg_pad_idx))
            # trg_seq is (batch_size, output_size-1), gold is (batch_size * (output_size-1))

            # forward
            self.optimizer.zero_grad()
            pred = self.model(src_seq=src_seq, trg_seq=trg_seq)
            pred = pred.view(-1, pred.size(2))

            # backward and update parameters
            loss, n_correct, n_word = self.cal_performance(
                pred, gold, self.trg_pad_idx, smoothing=self.label_smoothing)
            loss.backward()
            self.optimizer.step_and_update_lr()

            # note keeping
            n_word_total += n_word
            n_word_correct += n_correct
            total_loss += loss.item()

        loss_per_word = total_loss / n_word_total
        accuracy = n_word_correct / n_word_total
        return loss_per_word, accuracy

    def eval_epoch(self):
        self.model.eval()
        total_loss, n_word_total, n_word_correct = 0, 0, 0
        print("val epoch is processing: ")

        with torch.no_grad():
            for i in range(self.val_datasets_len // self.batch_size):
                # prepare data
                batch = next(self.val_datasets_iter)
                src_seq = self._patch_src(batch['src'], self.src_pad_idx).to(self.device)
                trg_seq, gold = map(lambda x: x.to(self.device), self._patch_trg(batch['trg'], self.trg_pad_idx))

                # forward
                pred = self.model(src_seq=src_seq, trg_seq=trg_seq)
                pred = pred.view(-1, pred.size(2))
                loss, n_correct, n_word = self.cal_performance(
                    pred, gold, self.trg_pad_idx, smoothing=False)

                # note keeping
                n_word_total += n_word
                n_word_correct += n_correct
                total_loss += loss.item()

        loss_per_word = total_loss / n_word_total
        accuracy = n_word_correct / n_word_total
        return loss_per_word, accuracy

    def _patch_src(self, src, pad_idx):
        src = src.transpose(0, 1)
        return src

    def _patch_trg(self, trg, pad_idx):
        trg = trg.transpose(0, 1)
        trg, gold = trg[:, :-1], trg[:, 1:].contiguous().view(-1)
        return trg, gold

    def print_performances(self, header, ppl, accu, start_time, lr):
        print('  - {header:12} ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, lr: {lr:8.5f}, ' \
              'elapse: {elapse:3.3f} min'.format(
            header=f"({header})", ppl=ppl,
            accu=100 * accu, elapse=(time.time() - start_time) / 60, lr=lr))

    def cal_loss(self, pred, gold, trg_pad_idx, smoothing=False):
        gold = gold.contiguous().view(-1)

        if smoothing:
            eps = 0.1
            n_class = pred.size(1)
            one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
            log_prb = F.log_softmax(pred, dim=1)
            non_pad_mask = gold.ne(trg_pad_idx)
            loss = -(one_hot * log_prb).sum(dim=1)
            loss = loss.masked_select(non_pad_mask).sum()  # average later
        else:
            loss = F.cross_entropy(pred, gold, ignore_index=trg_pad_idx, reduction='sum')
        return loss

    def cal_performance(self, pred, gold, trg_pad_idx, smoothing=False):
        loss = self.cal_loss(pred, gold, trg_pad_idx, smoothing=smoothing)

        pred = pred.max(1)[1]
        # print(pred.shape, " pred after: ", pred)

        gold = gold.contiguous().view(-1)
        # print(gold.shape, " gold: ", gold)

        non_pad_mask = gold.ne(trg_pad_idx)
        n_correct = pred.eq(gold).masked_select(non_pad_mask).sum().item()
        n_word = non_pad_mask.sum().item()
        return loss, n_correct, n_word

    def translate(self, model_path):
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'], strict=False)
        print('[Info] Trained model state loaded.')
        test_load = tqdm(self.test_dataset, mininterval=2, desc='  - (Test)', leave=False)
        count = 0
        for src, trg in test_load:
            print("")
            print("English input: ", "[", len(src), "]", ' '.join(src))
            print("German output: ", "[", len(trg), "]", ' '.join(trg))
            src_seq = [self.src_stoi.get(word, self.unk_idx) for word in src]
            pred_seq = self.model.translate_sentence(torch.LongTensor([src_seq]).to(self.device))
            pred_line = ' '.join(self.trg_itos[idx] for idx in pred_seq)
            pred_line = pred_line.replace(self.BOS_WORD, '').replace(self.EOS_WORD, '')
            print("German output: ", "[", len(pred_seq), "]", pred_seq)
            print("German output: ", "[", len(pred_seq), "]", pred_line)

            test_load.set_postfix(count=count)
            count += 1

        print('[Info] Finished.')

    def get_bleu(self, model_path):
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'], strict=False)
        print('[Info] Trained model state loaded.')
        targets = []
        outputs = []
        test_load = tqdm(self.test_dataset, mininterval=2, desc='  - (Test)', leave=False)
        count = 0
        for src, trg in test_load:
            src_seq = [self.src_stoi.get(word, self.unk_idx) for word in src]
            pred_seq = self.model.translate_sentence(torch.LongTensor([src_seq]).to(self.device))
            pred_line = ' '.join(self.trg_itos[idx] for idx in pred_seq)
            pred_line = pred_line.replace(self.BOS_WORD, '').replace(self.EOS_WORD, '')

            targets.append([trg])
            outputs.append(pred_line.split())

            test_load.set_postfix(count=count)
            count += 1

        print('BLEU: ', bleu_score(outputs, targets))

    def translate_single(self, model_path, input="a girl on a seashore with a mountain in the background."):
        inp = tokenize_en(input)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'], strict=False)
        print('[Info] Trained model state loaded.')
        print("English input: ", "[", len(inp), "]", ' '.join(inp))
        src_seq = [self.src_stoi.get(word, self.unk_idx) for word in inp]
        pred_seq = self.model.translate_sentence(torch.LongTensor([src_seq]).to(self.device))
        pred_line = ' '.join(self.trg_itos[idx] for idx in pred_seq)
        pred_line = pred_line.replace(self.BOS_WORD, '').replace(self.EOS_WORD, '')
        print("German output: ", "[", len(pred_seq), "]", pred_line)


def main():
    parser = argparse.ArgumentParser(description='Train Transformer')
    parser.add_argument('--batch_size', type=int, default=128, help='BATCH_SIZE')
    parser.add_argument('--emb_size', type=int, default=512, help='EMBEDDING_SIZE')
    parser.add_argument('--num_heads', type=int, default=8, help='ATTENTION_HEADS')
    parser.add_argument('--dropout', type=float, default=0.1, help='DROPOUT')
    parser.add_argument('--epoch', type=int, default=20, help='EPOCH')
    parser.add_argument('--n_layers', type=int, default=3, help='NEURAL_NET_LAYERS')
    args = parser.parse_args()

    torch.manual_seed(1337)

    trainer = TrainerTransformer(args)
    trainer.train()

    # trainer.translate_single(model_path="./output/model.chkpt")
    # trainer.translate(model_path="./output/model.chkpt")

    trainer.get_bleu(model_path="./output/model.chkpt")


if __name__ == "__main__":
    main()
