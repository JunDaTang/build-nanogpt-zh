"""
GPT-2模型训练脚本
简单运行方式:
$ python train_gpt2.py

使用DDP(分布式数据并行)运行，例如8个GPU:
$ torchrun --standalone --nproc_per_node=8 train_gpt2.py
"""

import os
import math
import time
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from hellaswag import render_example, iterate_examples
# -----------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    """
    因果自注意力层
    实现了GPT-2中的自注意力机制，确保每个位置只能看到之前的token
    """
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # 为所有注意力头创建key、query、value投影，批量处理
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # 输出投影
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # 正则化参数
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size() # batch size, 序列长度, 嵌入维度(n_embd)
        # 计算所有头的query、key、value，并将head维度移到batch维度
        # nh是"头数"，hs是"头大小"，C(通道数) = nh * hs
        # 例如在GPT-2 (124M)中，n_head=12，hs=64，所以nh*hs=C=768个通道
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # 使用flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, C) # 重新组装所有头的输出
        # 输出投影
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    """
    多层感知机层
    实现了GPT-2中的前馈网络
    """
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU(approximate='tanh')
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    """
    Transformer块
    包含一个自注意力层和一个MLP层，每个层都有层归一化
    """
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    """
    GPT模型配置类
    定义了模型的基本参数
    """
    block_size: int = 1024 # 最大序列长度
    vocab_size: int = 50257 # token数量: 50,000个BPE合并 + 256个字节token + 1个<|endoftext|>token
    n_layer: int = 12 # 层数
    n_head: int = 12 # 注意力头数
    n_embd: int = 768 # 嵌入维度

class GPT(nn.Module):
    """
    GPT模型实现
    包含完整的Transformer架构
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        # 创建Transformer组件
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd), # token嵌入
            wpe = nn.Embedding(config.block_size, config.n_embd), # 位置嵌入
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), # Transformer块
            ln_f = nn.LayerNorm(config.n_embd), # 最终层归一化
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False) # 语言模型头

        # 权重共享方案
        self.transformer.wte.weight = self.lm_head.weight

        # 初始化参数
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        初始化模型权重
        使用正态分布初始化线性层和嵌入层
        """
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """
        前向传播
        idx: 输入token索引，形状为(B, T)
        targets: 目标token索引，用于计算损失
        """
        B, T = idx.size()
        assert T <= self.config.block_size, f"序列长度{T}超过最大长度{self.config.block_size}"
        # 前向传播token和位置嵌入
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # 形状(T)
        pos_emb = self.transformer.wpe(pos) # 位置嵌入，形状(T, n_embd)
        tok_emb = self.transformer.wte(idx) # token嵌入，形状(B, T, n_embd)
        x = tok_emb + pos_emb
        # 前向传播Transformer块
        for block in self.transformer.h:
            x = block(x)
        # 前向传播最终层归一化和分类器
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """
        从预训练的GPT-2模型加载权重
        """
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("正在从预训练GPT加载权重: %s" % model_type)

        # 根据模型类型确定n_layer、n_head和n_embd
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M参数
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M参数
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M参数
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M参数
        }[model_type]
        config_args['vocab_size'] = 50257 # GPT模型检查点总是50257
        config_args['block_size'] = 1024 # GPT模型检查点总是1024
        # 创建一个从头初始化的minGPT模型
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # 丢弃这个mask/buffer，不是参数

        # 初始化一个huggingface/transformers模型
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # 复制参数，确保所有参数对齐且名称和形状匹配
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # 忽略这些，只是buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # 同上，只是mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # OpenAI检查点使用"Conv1D"模块，但我们只想使用普通的Linear
        # 这意味着在导入这些权重时需要转置
        assert len(sd_keys_hf) == len(sd_keys), f"键不匹配: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # 对需要转置的Conv1D权重进行特殊处理
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # 普通复制其他参数
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        """
        配置优化器
        使用AdamW优化器，对权重进行衰减，对偏置和层归一化不进行衰减
        """
        # 收集所有需要梯度的候选参数
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # 创建优化组。任何2D参数都将进行权重衰减，否则不进行。
        # 即所有矩阵乘法和嵌入中的权重张量进行衰减，所有偏置和层归一化不进行。
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        if master_process:
            print(f"衰减参数张量数量: {len(decay_params)}, 参数数量: {num_decay_params:,}")
            print(f"非衰减参数张量数量: {len(nodecay_params)}, 参数数量: {num_nodecay_params:,}")
        # 创建AdamW优化器，如果可用则使用fused版本
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        if master_process:
            print(f"使用fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

# -----------------------------------------------------------------------------
import tiktoken
import numpy as np

def load_tokens(filename):
    """
    从文件加载token
    """
    npt = np.load(filename)
    npt = npt.astype(np.int32)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoaderLite:
    """
    轻量级数据加载器
    用于加载和处理训练数据
    """
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        # 获取分片文件名
        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"未找到{split}的分片"
        if master_process:
            print(f"找到{len(shards)}个{split}分片")
        self.reset()

    def reset(self):
        """
        重置数据加载器状态
        """
        # 状态，从分片0开始
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        """
        获取下一个批次的数据
        """
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # 输入
        y = (buf[1:]).view(B, T) # 目标
        # 在张量中前进位置
        self.current_position += B * T * self.num_processes
        # 如果加载下一个批次会超出边界，前进到下一个分片
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y

# -----------------------------------------------------------------------------
# HellaSwag评估的辅助函数
# 接收tokens、mask和logits，返回损失最低的补全索引

def get_most_likely_row(tokens, mask, logits):
    """
    计算每个补全的损失，返回损失最低的索引
    """
    # 评估所有位置的自回归损失
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # 获取补全区域(where mask == 1)的平均损失
    shift_mask = (mask[..., 1:]).contiguous() # 必须移动mask，从最后一个提示token开始
    masked_shift_losses = shift_losses * shift_mask
    # 求和并除以mask中1的数量
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # 现在我们有4个补全的损失值
    # 损失最低的应该是最可能的
    pred_norm = avg_loss.argmin().item()
    return pred_norm

# -----------------------------------------------------------------------------
# 设置DDP(分布式数据并行)
# torchrun命令设置环境变量RANK、LOCAL_RANK和WORLD_SIZE
ddp = int(os.environ.get('RANK', -1)) != -1 # 这是DDP运行吗？
if ddp:
    # DDP目前需要CUDA，我们根据rank适当设置设备
    assert torch.cuda.is_available(), "目前我们认为DDP需要CUDA"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # 此进程将进行日志记录、检查点等
else:
    # 普通的非DDP运行
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # 尝试自动检测设备
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"使用设备: {device}")

# 注意：PyTorch对device和device_type的区分很严格
device_type = "cuda" if device.startswith("cuda") else "cpu"

# 设置随机种子以确保可重复性
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

# 初始化tokenizer
enc = tiktoken.get_encoding("gpt2")

# 设置训练参数
total_batch_size = 524288 # 2**19, ~0.5M tokens
B = 64 # 微批次大小
T = 1024 # 序列长度
assert total_batch_size % (B * T * ddp_world_size) == 0, "确保total_batch_size能被B * T * ddp_world_size整除"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f"期望的总批次大小: {total_batch_size}")
    print(f"=> 计算的梯度累积步数: {grad_accum_steps}")

# 创建数据加载器
train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train")
val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val")

# 设置PyTorch的矩阵乘法精度
torch.set_float32_matmul_precision('high')

# 创建模型
model = GPT(GPTConfig(vocab_size=50304))
# model = GPT.from_pretrained("gpt2") # 或从OpenAI GPT-2初始化
model.to(device)
use_compile = False # torch.compile会干扰HellaSwag评估和生成。TODO修复
if use_compile:
    model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model # 始终包含"原始"未包装的模型

# 设置学习率调度
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715
max_steps = 19073 # 19,073步约等于1个epoch，如果数据是10B tokens且批次大小0.5M tokens
def get_lr(it):
    """
    计算当前步的学习率
    1) 预热阶段线性增加
    2) 如果超过最大步数，返回最小学习率
    3) 中间使用余弦衰减到最小学习率
    """
    # 1) 预热阶段线性增加
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) 如果超过最大步数，返回最小学习率
    if it > max_steps:
        return min_lr
    # 3) 中间使用余弦衰减到最小学习率
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff从1开始到0
    return min_lr + coeff * (max_lr - min_lr)

# 创建优化器
optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type=device_type)

# 创建日志目录
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")
with open(log_file, "w") as f: # 打开文件清空内容
    pass

# 训练循环
for step in range(max_steps):
    t0 = time.time()
    last_step = (step == max_steps - 1)

    # 定期评估验证损失
    if step % 250 == 0 or last_step:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
            print(f"验证损失: {val_loss_accum.item():.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} val {val_loss_accum.item():.4f}\n")
            if step > 0 and (step % 5000 == 0 or last_step):
                # 可选：保存模型检查点
                checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'config': raw_model.config,
                    'step': step,
                    'val_loss': val_loss_accum.item()
                }
                # 你可能还想添加optimizer.state_dict()和
                # 随机种子等，如果你想更精确地恢复训练
                torch.save(checkpoint, checkpoint_path)

    # 定期评估HellaSwag
    if (step % 250 == 0 or last_step) and (not use_compile):
        num_correct_norm = 0
        num_total = 0
        for i, example in enumerate(iterate_examples("val")):
            # 只处理i % ddp_world_size == ddp_rank的示例
            if i % ddp_world_size != ddp_rank:
                continue
            # 将示例转换为tokens和标签
            _, tokens, mask, label = render_example(example)
            tokens = tokens.to(device)
            mask = mask.to(device)
            # 获取logits
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(tokens)
                pred_norm = get_most_likely_row(tokens, mask, logits)
            num_total += 1
            num_correct_norm += int(pred_norm == label)
        # 在所有进程间减少统计信息
        if ddp:
            num_total = torch.tensor(num_total, dtype=torch.long, device=device)
            num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
            dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
            num_total = num_total.item()
            num_correct_norm = num_correct_norm.item()
        acc_norm = num_correct_norm / num_total
        if master_process:
            print(f"HellaSwag准确率: {num_correct_norm}/{num_total}={acc_norm:.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} hella {acc_norm:.4f}\n")

    # 定期从模型生成文本(除了第0步，那是噪声)
    if ((step > 0 and step % 250 == 0) or last_step) and (not use_compile):
        model.eval()
        num_return_sequences = 4
        max_length = 32
        tokens = enc.encode("Hello, I'm a language model,")
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42 + ddp_rank)
        while xgen.size(1) < max_length:
            # 前向传播模型获取logits
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(xgen) # (B, T, vocab_size)
                # 获取最后一个位置的logits
                logits = logits[:, -1, :] # (B, vocab_size)
                # 获取概率
                probs = F.softmax(logits, dim=-1)
                # 进行top-k采样，k=50(huggingface pipeline默认值)
                # topk_probs变为(5, 50)，topk_indices为(5, 50)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                # 从top-k概率中选择一个token
                # 注意：multinomial不要求输入和为1
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
                # 收集对应的索引
                xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
                # 追加到序列
                xgen = torch.cat((xgen, xcol), dim=1)
        # 打印生成的文本
        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            decoded = enc.decode(tokens)
            print(f"rank {ddp_rank} 样本 {i}: {decoded}")

    # 执行一步优化
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        # 这个字段也被前向传播使用
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        # 我们必须缩放损失以考虑梯度累积，
        # 因为梯度在每个连续的backward()上都会累加。
        # 梯度的加法对应于目标中的SUM，但我们想要MEAN。
        # 在这里缩放损失，使其正确
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # 确定并设置当前迭代的学习率
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    if device_type == "cuda":
        torch.cuda.synchronize() # 等待GPU完成工作
    t1 = time.time()
    dt = t1 - t0 # 时间差(秒)
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed / dt
    if master_process:
        print(f"步数 {step:5d} | 损失: {loss_accum.item():.6f} | 学习率 {lr:.4e} | 范数: {norm:.4f} | 时间: {dt*1000:.2f}ms | tokens/秒: {tokens_per_sec:.2f}")
        with open(log_file, "a") as f:
            f.write(f"{step} train {loss_accum.item():.6f}\n")

if ddp:
    destroy_process_group()
