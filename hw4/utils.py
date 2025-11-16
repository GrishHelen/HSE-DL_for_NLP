from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import os

def get_tokenizer():
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-1.4b")
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    return tokenizer

def get_dataloaders(tokenizer, dataset_name='Anthropic/hh-rlhf', train_len=5000, val_len=500):
    batch_size = 32
    train_dataset = load_dataset(dataset_name, data_dir="harmless-base", split=f'train[:{train_len}]')
    val_dataset = load_dataset(dataset_name, data_dir="harmless-base", split=f'train[{train_len}:{train_len + val_len}]')

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=8)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, num_workers=8)

    return train_loader, val_loader

def count_trinable_params(model):
    return sum([p.numel() for p in model.parameters() if p.requires_grad])
