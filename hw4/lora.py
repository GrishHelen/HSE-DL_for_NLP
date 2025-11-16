import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
import numpy as np
import time


class LoRALayer(nn.Module):
    def __init__(self, base_layer: nn.Module, lora_rank: int = 4) -> None:
        super(LoRALayer, self).__init__()
        self.lora_rank = lora_rank

        self.base_layer = base_layer
        self.base_layer.eval()
        for param in base_layer.parameters():
            param.requires_grad = False

        self.lora_A = nn.Linear(base_layer.in_features, lora_rank, bias=False)
        self.lora_B = nn.Linear(lora_rank, base_layer.out_features, bias=False)

        nn.init.normal_(self.lora_A.weight)
        nn.init.zeros_(self.lora_B.weight)


class LoRALinear(LoRALayer):
    def __init__(self, base_layer: nn.Linear, lora_rank: int = 4) -> None:
        super().__init__(base_layer, lora_rank)

    def forward(self, x: Tensor) -> Tensor:
        base_out = self.base_layer(x)

        if self.lora_rank > 0:
            return base_out + self.lora_B(self.lora_A(x))

        return base_out


def get_base_model():
    model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-1.4b")
    return model

def substitute_layers(root, lora_rank):
    for name, module in root.named_children():
        if isinstance(module, nn.Linear):
            setattr(root, name, LoRALinear(module, lora_rank))
        elif len(list(module.children())) > 0:
            substitute_layers(module, lora_rank)

def apply_lora_to_gpt2(model: nn.Module, lora_rank: int):
    for name, param in model.named_parameters():
        param.requires_grad = False
    substitute_layers(model, lora_rank)
    return model

def get_lora_model(lora_rank: int = 4):
    base_model = get_base_model()
    with_lora_model = apply_lora_to_gpt2(base_model, lora_rank)

    return with_lora_model

def train_model_exp(model, tokenizer, train_loader, val_loader, n_epochs=1, max_length=512):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    model.to(device)

    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start_time = time.perf_counter()

    for epoch in range(n_epochs):
        print(f'epoch {epoch + 1}/{n_epochs}')
        model.train()
        losses = []
        n_steps = len(train_loader)
        for i, batch in enumerate(train_loader):
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
            print(f'iter {i+1}/{n_steps};  train loss: {loss.item()}')

        print('Epoch train loss:', np.mean(losses))
        print()

        model.eval()
        eval_model_exp(model, tokenizer, val_loader, max_length=max_length)

    torch.cuda.synchronize()
    time_elapsed = time.perf_counter() - start_time
    memory = torch.cuda.max_memory_allocated()

    print(f'elapsed time: {round(time_elapsed, 2)} s')
    print(f'memory: {memory}')

    return model

@torch.no_grad()
def eval_model_exp(model, tokenizer, val_loader, max_length=512):
    device = model.device
    model.eval()
    losses = []

    for batch in val_loader:
        texts = batch['chosen']
        inputs = tokenizer(texts, max_length=max_length, truncation=True,
                           padding='max_length', return_tensors='pt').to(device)
        outputs = model(**inputs)
        next_word_logits = outputs.logits[:, : -1, :]
        true_next_tokens = inputs['input_ids'][:, 1:]
        loss = F.cross_entropy(next_word_logits.flatten(0, 1), true_next_tokens.flatten(0, 1))
        losses.append(loss.item())

    print("Val loss:", np.mean(losses))

