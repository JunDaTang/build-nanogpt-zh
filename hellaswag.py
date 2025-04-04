"""
在Python中下载和评估HellaSwag数据集。
https://github.com/rowanz/hellaswag

HellaSwag json数据项示例:
{
    "ind": 24,                    # 数据集ID
    "activity_label": "Roof shingle removal",  # ActivityNet或WikiHow标签
    "ctx_a": "A man is sitting on a roof.",    # 上下文第一部分
    "ctx_b": "he",                              # 上下文第二部分(不完整的名词短语)
    "ctx": "A man is sitting on a roof. he",    # 完整上下文
    "split": "val",                             # 数据集划分(train/val/test)
    "split_type": "indomain",                   # 划分类型(indomain/zeroshot)
    "label": 3,                                 # 正确答案的索引(0-3)
    "endings": [                                # 4个可能的结尾选项
        "is using wrap to wrap a pair of skis.",
        "is ripping level tiles off.",
        "is holding a rubik's cube.",
        "starts pulling up roofing on a roof."
    ],
    "source_id": "activitynet~v_-JhWjGDPHMY"   # 数据来源ID
}

模型性能参考:
gpt2 (124M参数)
- eleuther harness报告准确率: 28.92%, 标准化准确率: 31.14% (多选形式)
- 本脚本: 10042个样本 准确率: 0.2859 标准化准确率: 0.2955 (补全形式)

gpt2-xl (1558M参数)
- eleuther harness报告准确率: 40.04%, 标准化准确率: 50.89% (多选形式)
- 本脚本: 10042个样本 准确率: 0.3842 标准化准确率: 0.4893 (补全形式)

HellaSwag验证集总共有10,042个样本。
"""

import os
import json
import requests
import tiktoken
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import GPT2LMHeadModel

# -----------------------------------------------------------------------------
# 设置数据缓存目录
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), "hellaswag")

def download_file(url: str, fname: str, chunk_size=1024):
    """
    从给定URL下载文件的辅助函数
    """
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    with open(fname, "wb") as file, tqdm(
        desc=fname,
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)

# HellaSwag数据集URL
hellaswags = {
    "train": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_train.jsonl",
    "val": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl",
    "test": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_test.jsonl",
}

# 初始化tokenizer
enc = tiktoken.get_encoding("gpt2")

def download(split):
    """
    下载HellaSwag数据集到DATA_CACHE_DIR目录
    """
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    data_url = hellaswags[split]
    data_filename = os.path.join(DATA_CACHE_DIR, f"hellaswag_{split}.jsonl")
    if not os.path.exists(data_filename):
        print(f"正在从{data_url}下载到{data_filename}...")
        download_file(data_url, data_filename)

def render_example(example):
    """
    将示例转换为三个torch张量:
    - tokens: 上下文+补全的token，大小为4xN(因为有4个候选答案)
    - mask: 在评估似然度的候选补全区域为1
    - label: 正确答案的索引
    """
    ctx = example["ctx"]
    label = example["label"]
    endings = example["endings"]

    # 用于在C size上重现此评估的数据
    data = {
        "label": label,
        "ctx_tokens": None,
        "ending_tokens": [],
    }

    # 收集所有token
    ctx_tokens = enc.encode(ctx)
    data["ctx_tokens"] = ctx_tokens
    tok_rows = []
    mask_rows = []
    for end in endings:
        end_tokens = enc.encode(" " + end) # 注意：在GPT-2 tokenizer前添加空格
        tok_rows.append(ctx_tokens + end_tokens)
        mask_rows.append([0]*len(ctx_tokens) + [1]*len(end_tokens))
        data["ending_tokens"].append(end_tokens)

    # 在合并时需要小心，因为每行的token数量可能不同
    max_len = max(len(row) for row in tok_rows)
    tokens = torch.zeros((4, max_len), dtype=torch.long)
    mask = torch.zeros((4, max_len), dtype=torch.long)
    for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
        tokens[i, :len(tok_row)] = torch.tensor(tok_row)
        mask[i, :len(mask_row)] = torch.tensor(mask_row)

    return data, tokens, mask, label

def iterate_examples(split):
    """
    迭代指定split的所有示例
    """
    download(split)
    with open(os.path.join(DATA_CACHE_DIR, f"hellaswag_{split}.jsonl"), "r") as f:
        for line in f:
            example = json.loads(line)
            yield example

@torch.no_grad()
def evaluate(model_type, device):
    """
    评估模型在HellaSwag数据集上的性能
    """
    torch.set_float32_matmul_precision('high') # 使用tf32
    model = GPT2LMHeadModel.from_pretrained(model_type)
    model.to(device)
    # model = torch.compile(model) # 可选：使用torch编译模型

    num_correct_norm = 0
    num_correct = 0
    num_total = 0
    for example in iterate_examples("val"):
        data, tokens, mask, label = render_example(example)
        tokens = tokens.to(device)
        mask = mask.to(device)

        # 获取logits
        logits = model(tokens).logits
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
        pred = sum_loss.argmin().item()
        pred_norm = avg_loss.argmin().item()

        # 累积统计信息
        num_total += 1
        num_correct += int(pred == label)
        num_correct_norm += int(pred_norm == label)
        print(f"{num_total} 标准化准确率: {num_correct_norm}/{num_total}={num_correct_norm/num_total:.4f}")

        # 调试：打印前几个示例和每种情况的损失
        if num_total < 10:
            print("---")
            print(f"上下文:\n {example['ctx']}")
            print(f"结尾选项:")
            for i, end in enumerate(example["endings"]):
                print(f"{i} (损失: {avg_loss[i].item():.4f}) {end}")
            print(f"预测: {pred_norm}, 实际: {label}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_type", type=str, default="gpt2", help="要使用的模型类型")
    parser.add_argument("-d", "--device", type=str, default="cuda", help="要使用的设备")
    args = parser.parse_args()
    evaluate(args.model_type, args.device)
