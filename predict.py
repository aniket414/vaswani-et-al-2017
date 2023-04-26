import argparse
import torch
import torch.nn as nn
from transformer.transformer import Transformer

def translate(model_path, args):
    return NotImplemented

def main():
    parser = argparse.ArgumentParser(description='Machine Translation')
    parser.add_argument('--inp', type=str, default='Ich bin ein Student der Deep-Learning-Klasse', help="YOUR_INPUT")
    args = parser.parse_args()
    translate(model_path="./output/model.chkpt", args=args)

if __name__ == "__main__":
    main()