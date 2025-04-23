#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import os
import torch.nn.functional as F
import torch
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.amp import GradScaler, autocast
import matplotlib.pyplot as plt
import re
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from torch.utils.data import Dataset
import re
import torch.nn as nn
import math
import numpy as np
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device name:", torch.cuda.get_device_name(0))


# In[2]:


import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_sparse_causal_mask(seq_len, window_size, device):
    """
    Generates a sparse causal mask for attention.
    Each token at position i attends only to tokens in the range:
       [max(0, i - window_size + 1), i]
    All other positions are masked out.
    
    Returns a Boolean tensor of shape [seq_len, seq_len] where True indicates a masked position.
    """
    # Causal (lower triangular) mask.
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
    # Sliding window mask: zero out positions too far in the past.
    window_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=-(window_size - 1))
    allowed = causal_mask & window_mask
    return ~allowed  # True means masked out.


def get_rotary_embeddings(seq_len, head_dim, device):
    """
    Computes cosine and sine embeddings for ROPE.
    head_dim is assumed to be even.
    Returns tensors of shape [1, 1, seq_len, head_dim//2] for cos and sin.
    """
    # Create inverse frequency vector
    inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    # Compute the outer product: [seq_len, head_dim//2]
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    cos = torch.cos(freqs).unsqueeze(0).unsqueeze(0)  # shape: [1, 1, seq_len, head_dim//2]
    sin = torch.sin(freqs).unsqueeze(0).unsqueeze(0)  # shape: [1, 1, seq_len, head_dim//2]
    return cos, sin

def apply_rotary_pos_emb(x, cos, sin):
    """
    Applies rotary positional embedding.
    x: Tensor of shape [batch_size, num_heads, seq_len, head_dim] where head_dim is even.
    Splits the head dimension into even and odd parts, rotates, and then interleaves them back.
    """
    # Split into even and odd parts: each of shape [..., head_dim//2]
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    # Apply rotation: for each position, compute:
    # new_even = x_even * cos - x_odd * sin
    # new_odd  = x_even * sin + x_odd * cos
    x_rotated_even = x_even * cos - x_odd * sin
    x_rotated_odd  = x_even * sin + x_odd * cos
    # Interleave the even and odd parts back together.
    # One way is to stack along a new last dimension then flatten it.
    x_out = torch.stack((x_rotated_even, x_rotated_odd), dim=-1).flatten(-2)
    return x_out

class HybridMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, sparse_window_size=63, causal_heads=0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by the number of heads."
        # For ROPE, we assume head_dim is even.
        assert self.head_dim % 2 == 0, "head_dim must be even for ROPE encoding."
        self.causal_heads = causal_heads
        self.sparse_window_size = sparse_window_size

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, key_padding_mask=None):
        batch_size, seq_len, _ = query.size()

        # Linear projections.
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # Reshape to [batch_size, num_heads, seq_len, head_dim].
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        device = query.device
        # --- ROPE: apply rotary positional encoding to q and k ---
        cos, sin = get_rotary_embeddings(seq_len, self.head_dim, device)
        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)
        # ----------------------------------------------------------

        scale = math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale  # [batch_size, num_heads, seq_len, seq_len]

        # Apply key padding mask if provided.
        if key_padding_mask is not None:
            kp_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_len]
            scores = scores.masked_fill(kp_mask, float('-inf'))

        # Apply causal mask to the first 'causal_heads'.
        if self.causal_heads > 0:
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
            scores[:, :self.causal_heads] = scores[:, :self.causal_heads].masked_fill(causal_mask, float('-inf'))

        # Apply sparse sliding-window mask to remaining heads.
        num_sparse_heads = self.num_heads - self.causal_heads
        if num_sparse_heads > 0:
            sparse_mask = get_sparse_causal_mask(seq_len, self.sparse_window_size, device)
            sparse_mask = sparse_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
            scores[:, self.causal_heads:] = scores[:, self.causal_heads:].masked_fill(sparse_mask, float('-inf'))

        # Check for rows that are entirely masked; replace them with zeros.
        all_masked = (scores == float('-inf')).all(dim=-1, keepdim=True)
        if all_masked.any():
            scores = scores.masked_fill(all_masked, 0)

        # Compute attention weights.
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        # Compute attention output.
        attn_output = torch.matmul(attn_weights, v)  # [batch_size, num_heads, seq_len, head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        output = self.out_proj(attn_output)

        return output.to(query.dtype)

# In your TransformerLM class, you no longer need to add a precomputed sinusoidal positional encoding.
# Instead, you simply use token embeddings, and the rotary encoding will be applied in the attention layer.
class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size,
        pad_token_id,
        embed_dim=512,
        num_heads=8,
        num_layers=6,
        ff_dim=2048,
        dropout_rate=0.1,
        max_seq_len=250,
        drop_path_rate=0.0,
        sparse_window_size=None,  # Use an integer (e.g., 16 or 32) to use sparse attention.
        causal_heads=None
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.pad_token_id = pad_token_id  # For building the padding_mask

        # Token Embeddings
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # (Remove sinusoidal positional encoding here; ROPE is applied inside attention)
        # self.positional_encoding = nn.Parameter(get_sinusoid_encoding(max_seq_len, embed_dim), requires_grad=False)

        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(embed_dim)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout_rate, drop_path_rate,
                             sparse_window_size=sparse_window_size, causal_heads=causal_heads)
            for _ in range(num_layers)
        ])

        self.fc_out = nn.Linear(embed_dim, vocab_size, bias=False)
        self.fc_out.weight = self.embedding.weight  # Weight Tying

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        """
        :param x: [batch_size, seq_len] of token IDs.
        """
        padding_mask = (x == self.pad_token_id)  # [batch_size, seq_len]
        seq_len = x.size(1)

        # Token embedding; note that we no longer add a fixed positional encoding.
        x = self.embedding(x) * math.sqrt(self.embed_dim)
        x = self.dropout(x)
        x = self.layer_norm(x)

        for block in self.transformer_blocks:
            x = block(x, padding_mask)

        logits = self.fc_out(x)  # [batch_size, seq_len, vocab_size]
        return logits

# Note: The rest of your code (e.g., TransformerBlock, LabelSmoothingCrossEntropy, etc.)
# remains the same.



class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate, drop_path_rate=0.0, sparse_window_size=0, causal_heads=None):
        """
        :param embed_dim: Embedding dimension.
        :param num_heads: Number of attention heads.
        :param ff_dim: Hidden dimension in the feed-forward network.
        :param dropout_rate: Dropout rate.
        :param drop_path_rate: DropPath rate.
        :param sparse_window_size: If set, uses sparse sliding-window attention for non-causal heads.
        :param causal_heads: Number of heads to use full causal attention. If not provided, you can default to half the heads.
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        # Set default: half the heads use full causal attention.
        if causal_heads is None:
            causal_heads = num_heads // 2
        self.attn = HybridMultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout_rate,
            causal_heads=causal_heads,
            sparse_window_size=sparse_window_size
        )
        self.drop_path = nn.Dropout(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout_rate)
        )
        self.sparse_window_size = sparse_window_size

    def forward(self, x, padding_mask=None):
        # Pre-LayerNorm.
        x_norm = self.norm1(x)  # [batch_size, seq_len, embed_dim]
        # The custom attention module handles masking per head.
        attn_out = self.attn(x_norm, x_norm, x_norm, key_padding_mask=padding_mask)
        
        # Residual connection with DropPath.
        x = x + self.drop_path(attn_out)
        
        # Feed-Forward network.
        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm)
        x = x + self.drop_path(ffn_out)
        return x




class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, ignore_index=-250, smoothing=0.0, reduction="mean"):
        """
        Constructor for the LabelSmoothingCrossEntropy module.
        
        :param smoothing: Label smoothing factor.
        :param ignore_index: Specifies a target value that is ignored and does not contribute to the input gradient.
        :param reduction: Specifies the reduction to apply to the output: 'mean' or 'sum'.
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.ignore_index = ignore_index
        self.smoothing = smoothing
        self.reduction = reduction
        self.confidence = 1.0 - smoothing

    def forward(self, pred, target):
        """
        Forward pass for label smoothing cross entropy.
        
        :param pred: Predictions (logits) [batch_size, num_classes].
        :param target: Ground truth labels [batch_size].
        :return: Smoothed cross entropy loss.
        """
        log_pred = F.log_softmax(pred, dim=-1)

        # If no smoothing is required, use the standard loss for efficiency.
        if self.smoothing == 0:
            return F.nll_loss(log_pred, target, ignore_index=self.ignore_index, reduction=self.reduction)
        
        with torch.no_grad():
            # Initialize the target distribution with smoothing value.
            true_dist = torch.full_like(log_pred, self.smoothing / (pred.size(1) - 1))
            # Set the confidence for the correct labels.
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
            # Zero out the probabilities for the ignore_index targets.
            mask = target.unsqueeze(1) == self.ignore_index
            true_dist.masked_fill_(mask, 0)

        # Compute the loss.
        loss = -torch.sum(true_dist * log_pred, dim=-1)

        # Apply the chosen reduction on non-ignored targets.
        valid_loss = loss[target != self.ignore_index]
        if self.reduction == "mean":
            loss = valid_loss.mean()
        elif self.reduction == "sum":
            loss = valid_loss.sum()
        else:
            loss = valid_loss  # no reduction
        return loss



# In[3]:


from datasets import Dataset
from transformers import GPT2Tokenizer
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from torch.amp import GradScaler, autocast
import os
from datasets import load_dataset
from transformers import GPT2Tokenizer

#wikitext-103-raw-v1
#wikitext-2-raw-v1
def prepare_datasets(
    dataset_variant="wikitext-103-raw-v1", 
    batch_size=1000, 
    max_length=250, 
    num_proc=16, 
    use_cache=True
):
    from datasets import load_dataset
    from transformers import GPT2Tokenizer
    import os

    # 1. Load the dataset variant (WikiText-2 or WikiText-3)
    dataset = load_dataset("wikitext", dataset_variant)

    # 2. Setup tokenizer and add a pad token
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")  
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    pad_id = tokenizer.pad_token_id

    # 3. Define a tokenization function (with a simple text cleanup)
    def tokenize(batch):
        cleaned_text = [text.replace("@-@", "-") for text in batch["text"]]
        return tokenizer(
            cleaned_text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            add_special_tokens=True
        )
    
    # 4. Process datasets in parallel; use caching if desired
    train_data = dataset["train"].map(
        tokenize,
        batched=True,
        batch_size=batch_size,
        num_proc=num_proc,
        remove_columns=["text"],
        load_from_cache_file=use_cache
    )
    
    val_data = dataset["validation"].map(
        tokenize,
        batched=True,
        batch_size=batch_size,
        num_proc=num_proc,
        remove_columns=["text"],
        load_from_cache_file=use_cache
    )

    # 5. Filter out sequences that are entirely padding
    train_data = train_data.filter(lambda x: any(token != pad_id for token in x["input_ids"]))
    val_data = val_data.filter(lambda x: any(token != pad_id for token in x["input_ids"]))

    print(f"Train dataset length: {len(train_data)}")
    print(f"Val dataset length: {len(val_data)}")

    # 6. Count tokens efficiently
    total_train_tokens = sum(map(len, train_data["input_ids"]))
    total_val_tokens = sum(map(len, val_data["input_ids"]))
    print(f"Total tokens in train dataset: {total_train_tokens}")
    print(f"Total tokens in validation dataset: {total_val_tokens}")
    print(f"Total tokens in dataset: {total_train_tokens + total_val_tokens}")

    return train_data, val_data, tokenizer



def collate_fn(batch):
    # Each item in the batch is {"input_ids": [...list of ints...]}
    input_ids_list = [item["input_ids"] for item in batch]
    # Convert each list of ints to a 1D tensor
    input_ids_tensors = [torch.tensor(ids, dtype=torch.long) for ids in input_ids_list]
    # Pad/stack them if needed
    input_ids = torch.stack(input_ids_tensors, dim=0)  # shape: [batch_size, seq_len]

    # Return shifted inputs/targets
    return input_ids[:, :-1], input_ids[:, 1:]


# In[4]:


from torch.optim import AdamW

def initialize_training(model, tokenizer):
    device = torch.device("cuda")
    model = model.to(device)
    


    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=.01, foreach=True)
    
    # Mixed precision scaler
    scaler = GradScaler()
    
    # Enable cudnn benchmarking
    torch.backends.cudnn.benchmark = True
    
    return model, optimizer, scaler, device

# ====================
# 4. TRAINING LOOP
# ====================
# Cell 4: Corrected Training Loop
def train_batch(model, inputs, targets, optimizer, scaler, device, pad_id):
    """Process a single batch"""
    # Use non_blocking transfers if possible
    inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
    
    with torch.amp.autocast('cuda'):
        outputs = model(inputs)
        loss = F.cross_entropy(
            outputs.view(-1, outputs.size(-1)),
            targets.view(-1),
            ignore_index=pad_id
        )
    
    # Backward pass with scaled loss
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    # Simple NaN check on scalar loss
    if torch.isnan(loss):
        print("NaN loss detected!")
        
    return loss.item()



# In[5]:


import torch
import torch.nn.functional as F
from torch.amp import autocast  # Updated import

def top_filtering(logits, top_k=0, top_p=1.0, filter_value=-float('Inf')):
    """Apply top-k and nucleus (top-p) filtering to logits.
    
    Args:
        logits (torch.Tensor): Logits distribution of shape (vocab_size,).
        top_k (int): Keep only top k tokens with highest probability.
        top_p (float): Keep the smallest set of tokens with cumulative probability >= top_p.
        filter_value (float): Logits to assign to filtered tokens.
    Returns:
        torch.Tensor: Filtered logits.
    """
    # Ensure logits is a 1D tensor
    assert logits.dim() == 1

    # Top-k filtering
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        values, _ = torch.topk(logits, top_k)
        min_value = values[-1]
        logits[logits < min_value] = filter_value

    # Top-p (nucleus) filtering
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        
        # Shift the indices to ensure at least one token is kept
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
        sorted_indices_to_remove[0] = 0
        
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
        
    return logits

def generate_text_strict(
    model,
    tokenizer,
    device,
    prompt="Attempts have been made...",
    max_length=50,
    temperature=0.7,
    top_k=40,
    top_p=0.9,
    repetition_penalty=1.2,
    no_repeat_ngram_size=3
):
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    for _ in range(max_length):
        with torch.no_grad():
            outputs = model(input_ids)
            # Apply temperature
            logits = outputs[:, -1, :] / temperature
            logits = logits.squeeze(0)
            
            # Repetition penalty
            if repetition_penalty != 1.0:
                logits = apply_repetition_penalty(logits, input_ids[0], penalty=repetition_penalty)

            # No-repeat n-gram
            if no_repeat_ngram_size > 0:
                logits = no_repeat_ngram(logits, input_ids, n=no_repeat_ngram_size)

            # Top-k + top-p
            filtered_logits = top_filtering(logits, top_k=top_k, top_p=top_p)
            
            # Sample
            probs = F.softmax(filtered_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).unsqueeze(0)
        
        input_ids = torch.cat([input_ids, next_token], dim=-1)
        
        # Stop if EOS
        if next_token.item() == tokenizer.eos_token_id:
            break
        if input_ids.shape[-1] >= max_length:
            break
    
    return tokenizer.decode(input_ids[0], skip_special_tokens=True)


def apply_repetition_penalty(logits, input_ids, penalty=2.5):
    """
    Applies a repetition penalty by down-weighting the logits
    of any previously generated token IDs. The idea is to reduce
    the likelihood of repeating the exact same token if it appears
    in input_ids.

    Args:
        logits (torch.Tensor): The predicted logits of shape (vocab_size,).
        input_ids (torch.Tensor): The sequence of generated tokens so far
                                  (shape (sequence_length,)).
        penalty (float): The factor by which to down-weight repeated tokens.

    Returns:
        torch.Tensor: Modified logits with repetition penalty applied.
    """
    # For each unique token ID in input_ids, divide the logit by penalty
    for token_id in set(input_ids.tolist()):
        logits[token_id] /= penalty

    return logits

def no_repeat_ngram(logits, input_ids, n=2):
    if input_ids.shape[-1] >= n:
        # get last (n-1) tokens
        recent_ngram = tuple(input_ids[0, -n+1:].tolist())
        # find all possible (n-1) + next_token combos in input_ids
        for i in range(input_ids.shape[-1] - n + 1):
            # if you find the same n-1 tokens
            if tuple(input_ids[0, i:i+n-1].tolist()) == recent_ngram:
                # forbid the next token from that occurrence
                forbidden_token = input_ids[0, i+n-1].item()
                logits[forbidden_token] = -float('Inf')
    return logits

def validate(model, val_loader, device, pad_id):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            # Use non_blocking transfers if data is in pinned memory
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = F.cross_entropy(
                    outputs.view(-1, outputs.size(-1)),
                    targets.view(-1),
                    ignore_index=pad_id
                )
            
            batch_size = inputs.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
    return total_loss / total_samples if total_samples > 0 else float('inf')



# In[6]:


import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader



# Configuration (easily adjustable)
CONFIG = {
    "batch_size": 64,      # RTX 4090 can handle larger batches
    "num_epochs": 10,
    "eval_interval": 2,    # Generate text every N epochs
    "max_seq_len": 250,
    "temp": 0.9,           # Generation temperature
    "top_k": 20
}


# In[7]:


# Cell 2: Prepare Data
print("Loading and tokenizing data...")
train_data, val_data, tokenizer = prepare_datasets()
print(any(all(token_id == tokenizer.pad_token_id for token_id in seq) for seq in train_data["input_ids"]))


# In[8]:


# Filter out all-pad sequences
pad_id = tokenizer.pad_token_id
def remove_all_pad(example):
    ids = example["input_ids"]
    return any(token_id != pad_id for token_id in ids)

train_data = train_data.filter(remove_all_pad)
val_data   = val_data.filter(remove_all_pad)


# In[9]:


train_loader = DataLoader(
    train_data, 
    batch_size=64, 
    shuffle=True, 
    collate_fn=collate_fn
)
val_loader = DataLoader(
    val_data, 
    batch_size=64, 
    shuffle=False, 
    collate_fn=collate_fn
)



# In[10]:


# Cell 3: Initialize Model
print("\nInitializing model...")
model = TransformerLM(
    vocab_size=len(tokenizer),
    pad_token_id=tokenizer.pad_token_id,
    embed_dim=512,      
    num_heads=8,        
    num_layers=6,       
    ff_dim=2056,        
    dropout_rate=0.1,   
    max_seq_len=250,
    drop_path_rate=0.0,
    sparse_window_size=25,
    causal_heads = 4
)

model, optimizer, scaler, device = initialize_training(model, tokenizer)
print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
print(f"Training on: {device}")



# In[11]:


# Cell 4: Training Loop
train_losses = []
val_losses = []

num_epochs = CONFIG["num_epochs"]

label_smoothing = 0.000 # Adjust as needed
ignore_index = tokenizer.pad_token_id  # Ensure this matches your padding token ID

loss_fn = LabelSmoothingCrossEntropy(
    ignore_index=ignore_index,
    smoothing=label_smoothing
)

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0

    for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        # Use non_blocking transfers if using pinned memory
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        
        # Zero gradients once per iteration, more memory efficient
        optimizer.zero_grad(set_to_none=True)
        
        with torch.amp.autocast('cuda'):
            outputs = model(inputs)  # [batch_size, seq_len, vocab_size]
            # Flatten outputs and targets for loss computation
            outputs = outputs.view(-1, outputs.size(-1))  # [batch_size * seq_len, vocab_size]
            targets = targets.view(-1)  # [batch_size * seq_len]
            loss = loss_fn(outputs, targets)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.8)
        scaler.step(optimizer)
        scaler.update()
        
        epoch_loss += loss.item()
    
    avg_train_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    print(f"Epoch {epoch+1}: Training Loss = {avg_train_loss:.4f}")
    
    # Validation loop
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                outputs = outputs.view(-1, outputs.size(-1))
                targets = targets.view(-1)
                loss = loss_fn(outputs, targets)
            val_loss += loss.item()
    
    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    perplexity = math.exp(avg_val_loss)
    print(f"Epoch {epoch+1}: Validation Loss = {avg_val_loss:.4f}, Perplexity = {perplexity:.2f}")

# Cell 5: Save Model (optional)
print("Training complete!")
# model.save_pretrained("transformer_lm")  # Uncomment to save


# In[197]:


prompt_text = "The United States is best known for"
generated = generate_text_strict(
    model=model,
    tokenizer=tokenizer,
    device=device,
    prompt=prompt_text,
    temperature=.75,
    top_k=100,
    top_p=.90,
    repetition_penalty=1.2,
    no_repeat_ngram_size=1
)
print("Generated:", generated)
   


# In[ ]:





# In[ ]:




