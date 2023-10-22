import torch
import torch.nn.functional as F
from torch import nn

import numpy as np
import re


# %% [markdown]
# # GPT

# %%
with open('./shakespeare.txt', encoding='utf-8') as f:
    text = f.read()

text = text.lower()
# text

# %%
batch_size = 8  # How many independent sequences will we process in parallel?
block_size = 256  # What is the maximum context length for prediction?
att_size = 16
max_iters = 14000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 256
n_head = 6
n_layer = 6
dropout = 0.2

# %%
def token(string):
    return re.findall('\w+', string)

# %%
# from collections import Counter

# %%
# token(text)

words = list(token(text))


# %%
vocab_size = len(set(words))

# %%
# Counter(words).most_common()

# %%
stoi = { ch:i for i,ch in enumerate(set(words))}
itos = { i:ch for ch,i in stoi.items()}

encode = lambda s : [stoi[c] for c in s]  # encoder: take a string, output a list of integers
decode = lambda l : ' '.join([itos[i] for i in l])  # decoder: take a list of integers, output a string

# %%
data = torch.tensor(encode(words), dtype=torch.long)
n = int(0.9*len(data))  # first 90% will be train, rest val
train_data = data[:n]
val_data = data[:n]

# %%
torch.cuda.empty_cache()  # 释放未使用的 GPU 内存
torch.backends.cudnn.benchmark = True  # 提高性能，但可能会增加 GPU 内存使用
torch.backends.cudnn.deterministic = True  # 使运行更可重复，但可能会减少性能

# %%
class SelfAttention(nn.Module):
    def __init__(self, att_size):
        super().__init__()
        self.query = nn.Linear(n_embd, att_size, bias=False)
        self.key = nn.Linear(n_embd, att_size, bias=False)
        self.value = nn.Linear(n_embd, att_size, bias=False)
        self.dropout = nn.Dropout(dropout)  # 增加难度，随机将一些位置置为0

    def forward(self, x):
        B, L, D = x.shape

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        weights = q @ k.transpose(1, 2)
        self.tril = torch.tril(torch.ones(L, L)).to(device)
        
        weights = F.softmax(weights.masked_fill(self.tril[:L, :L] == 0, float('-inf')), dim=-1)
        weights = self.dropout(weights)

        out = weights @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, att_size):
        super().__init__()
        self.heads = nn.ModuleList(SelfAttention(att_size) for _ in range(num_heads))
        self.proj = nn.Linear(num_heads * att_size, n_embd)  # 映射：一个线性变换将输出降为n_embsize维
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

# %%
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        hidden_size = 4 * n_embd  # 这是一个任意的中间状态
        self.net =  nn.Sequential(
            nn.Linear(n_embd, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_embd),  # 变回n_embd维
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

# %%
class TransformerBlock(nn.Module):
    def __init__(self, n_embd, att_size):
        super().__init__()
        head_size = n_embd // att_size  # 这里head_size即多头个数，理论上可以任意设置，不过这里采用谷歌推荐的写法
        self.sa = MultiHeadAttention(head_size, att_size)
        self.ffw = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)  # Layer Normalization有助于稳定训练，通过标准化输入的均值和方差，以减少梯度消失问题，也是为了确保残差连接的稳定性
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))  # self.sa 是多头自注意力（Multi-Head Self-Attention）操作的组件。多头自注意力接受规范化后的输入 self.ln1(x)，然后执行注意力操作，将输入序列中不同位置的信息进行交互。这可以理解为模型在不同位置对输入进行关注，以便更好地理解输入的上下文。
        x = x + self.ffw(self.ln2(x))  # 前馈
        return x

# %%
class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.stacked_transformers = nn.Sequential(*[TransformerBlock(n_embd, att_size) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # better init
        self.apply(self._init_weight)

    def _init_weight(self, module):
         if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, target=None):
        B, L = idx.shape
        tok_emb = self.token_embedding_table(idx)  # (B, L, D)
        pos_emb = self.position_embedding_table(torch.arange(L, device=device))
        
        x = tok_emb + pos_emb  # (B, L, D)
        x = self.stacked_transformers(x)
        x = self.ln_f(x)
        # logits = F.softmax(self.lm_head(x))
        logits = self.lm_head(x)

        if target is None:
            loss = None
        else:
            B, L, D = logits.shape
            logits = logits.view(B * L, D)
            target = target.view(B * L)
            loss = F.cross_entropy(logits, target)
            
        return logits, loss
    
    def generate(self, idx, max_new_tokens):

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# %%
model = GPTLanguageModel()
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# %%
# data loading
def get_batch(split):
    data = train_data if split == 'trian' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# %%
losses = []
loss_history = []

for iter in range(max_iters):
    # every once in a while evaluate the loss on trian and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    loss_history.append(loss)
    print(f'iter: {iter}, loss: {loss}')
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# %%
torch.save(model.state_dict(), 'model2.pth')


# %%
prompt = 'to be'

# %%
idx_input = [stoi[w] for w in prompt.split()]

# %%
context = torch.tensor([idx_input], device=device)
print(decode(m.generate(context, max_new_tokens=100)[0].tolist()))

# %%


# %%
# 保存模型
# torch.save(model.state_dict(), 'model.pth')

# %%
# 加载模型
# model = GPTLanguageModel()  # 创建一个新的模型实例
# model.load_state_dict(torch.load('model.pth'))  # 加载已保存的权重和参数
# model.eval()  # 设置模型为评估模式

# m = model.to(device)

# prompt = 'to be'
# idx_input = [stoi[w] for w in prompt.split()]
# context = torch.tensor([idx_input], device=device)
# print(decode(m.generate(context, max_new_tokens=10)[0].tolist()))


