import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
import numpy as np
import time
import os


class SFTDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]

def prepare_sft_data(dataset):
    processed_data = []
    
    for example in dataset:
        text = example['chosen']
        last_assistant_idx = text.rfind("Assistant:")

        if last_assistant_idx == -1:
            continue
        
        prompt = text[:last_assistant_idx].strip()
        response = text[last_assistant_idx:].strip()
        
        if len(prompt) and len(response):
            processed_data.append({
                'prompt': prompt,
                'response': response
            })
    
    return processed_data

def get_sft_dataloaders(train_loader, val_loader, batch_size=64):
    pocessed_train = prepare_sft_data(train_loader.dataset)
    processed_tval = prepare_sft_data(val_loader.dataset)

    pocessed_train_loader = DataLoader(SFTDataset(pocessed_train), 
                                       shuffle=True, batch_size=batch_size, num_workers=8)
    pocessed_val_loader = DataLoader(SFTDataset(processed_tval), 
                                     shuffle=False, batch_size=batch_size, num_workers=8)
    
    return pocessed_train_loader, pocessed_val_loader


@torch.no_grad()
def eval_model_sft(model, tokenizer, val_loader, max_length=512):
    device = model.device
    model.eval()
    losses = []

    for batch in val_loader:
        texts = batch['prompt']
        true_response = batch['response']

        inputs = tokenizer(texts, max_length=max_length, truncation=True, 
                           padding='max_length', return_tensors='pt').to(device)
        
        outputs = model(**inputs)
        next_word_logits = outputs.logits[:, : -1, :]
        true_next_tokens = inputs['input_ids'][:, 1:]
        loss = F.cross_entropy(next_word_logits.flatten(0, 1), true_next_tokens.flatten(0, 1))
        losses.append(loss.item())

    print("Val loss:", np.mean(losses))


def train_model_sft(model, tokenizer, train_loader, val_loader, epochs=5, max_length=512):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    model.to(device)

    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start_time = time.perf_counter()

    for epoch in range(n_epochs):
        model.train()
        losses = []
        for batch in train_loader:
            texts = batch['chosen']
            inputs = tokenizer(texts, max_length=max_length, truncation=True,
                               padding='max_length', return_tensors='pt').to(device)
            outputs = model(**inputs)
            next_word_logits = outputs.logits[:, : -1, :]
            true_next_tokens = inputs['input_ids'][:, 1:]
            loss = F.cross_entropy(next_word_logits.flatten(0, 1), true_next_tokens.flatten(0, 1))
            losses.append(loss.item())

            loss.backward()
            opt.step()
            opt.zero_grad()

        print('Train loss:', np.mean(losses))
        
        model.eval()
        eval_model_exp(model, tokenizer, val_loader, max_length=max_length)
        
    torch.cuda.synchronize()
    time_elapsed = time.perf_counter() - start_time
    memory = torch.cuda.max_memory_allocated()

    return model, time_elapsed, memory

