import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
import numpy as np
from numpy.random import randint
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
        prompt = batch['prompt']
        response = batch['response']
         
        resp_inputs = tokenizer(response,max_length=max_length, truncation=True,
                           padding='max_length', return_tensors='pt')['input_ids'].to(device)
        
        len_test_ids = len(tokenizer(prompt, max_length=max_length, truncation=True,
                           padding='max_length', return_tensors='pt')['input_ids'].to(device))
        num_prompts = len(resp_inputs)
        
        space_for_prompts = torch.full([len_test_ids, num_prompts], fill_value=tokenizer.pad_token_id,
                                       dtype=torch.int64, device=device)
        batch['input_ids'] = torch.cat([space_for_prompts, resp_inputs['input_ids']], dim=1)
        batch['attention_mask'] = torch.cat([torch.ones_like(space_for_prompts), resp_inputs['attention_mask']], dim=1)

        outputs = model(**batch)
        next_word_logits = outputs.logits[:, num_prompts : -1, :]
        true_next_tokens = resp_inputs['input_ids'][:, num_prompts + 1:]
        loss = F.cross_entropy(next_word_logits.flatten(0, 1), true_next_tokens.flatten(0, 1))
        losses.append(loss.item())

    print("Val loss:", np.mean(losses))


def train_model_sft(model, tokenizer, train_loader, val_loader, n_epochs=5, max_length=512):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opt = torch.optim.Adam(model.parameters(), lr=2e-4)
    model.to(device)

    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start_time = time.perf_counter()

    for epoch in range(n_epochs):
        model.train()
        losses = []
        for batch in train_loader:
            prompt = batch['prompt']
            response = batch['response']
            
            resp_inputs = tokenizer(response, max_length=max_length, truncation=True,
                           padding='max_length', return_tensors='pt')['input_ids'].to(device)
            
            len_test_ids = len(tokenizer(prompt, max_length=max_length, truncation=True,
                           padding='max_length', return_tensors='pt')['input_ids'].to(device))
            num_prompts = len(resp_inputs)
            
            space_for_prompts = torch.full([len_test_ids, num_prompts], fill_value=tokenizer.pad_token_id,
                                        dtype=torch.int64, device=device)
            batch['input_ids'] = torch.cat([space_for_prompts, resp_inputs['input_ids']], dim=1)
            batch['attention_mask'] = torch.cat([torch.ones_like(space_for_prompts), resp_inputs['attention_mask']], dim=1)

            outputs = model(**batch)
            next_word_logits = outputs.logits[:, num_prompts : -1, :]
            true_next_tokens = resp_inputs['input_ids'][:, num_prompts + 1:]
            loss = F.cross_entropy(next_word_logits.flatten(0, 1), true_next_tokens.flatten(0, 1))
            losses.append(loss.item())

            loss.backward()
            opt.step()
            opt.zero_grad()

        print('Train loss:', np.mean(losses))
        
        model.eval()
        eval_model_sft(model, tokenizer, val_loader, max_length=max_length)
        
    torch.cuda.synchronize()
    time_elapsed = time.perf_counter() - start_time
    memory = torch.cuda.max_memory_allocated()

    return model, time_elapsed, memory


def model_apply(model, tokenizer, prompt, max_length=32):
    model.eval()
    device = model.device
    n = 16
    
    test_input_ids = tokenizer(prompt, max_length=max_length, truncation=True,
                           padding='max_length', return_tensors='pt')['input_ids'].to(device)
    space_for_prompts = torch.full([len(test_input_ids), n], fill_value=tokenizer.pad_token_id,
                                dtype=torch.int64, device=device)

    batch = tokenizer(prompt, return_tensors='pt', return_token_type_ids=False).to(device)
    batch['input_ids'] = torch.cat([space_for_prompts, batch['input_ids']], dim=1)
    batch['attention_mask'] = torch.cat([torch.ones_like(space_for_prompts), batch['attention_mask']], dim=1)

    length = randint(min(5, max_length), max_length + 1)
    for i in range(length):
        next_token = model(**batch).logits[0, -1].argmax(-1).reshape(1, 1)
        batch['input_ids'] = torch.cat([batch['input_ids'], next_token], dim=-1)
        batch['attention_mask'] = torch.cat([batch['attention_mask'], torch.ones_like(next_token)], dim=-1)

    print("\nOutput:", tokenizer.decode(batch['input_ids'][0, n:].cpu().numpy().tolist()))

