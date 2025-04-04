"""
FineWeb-Edu数据集(用于srs预训练)
https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
下载并标记化数据，将数据分片保存到磁盘。
运行方式:
$ python fineweb.py
数据将保存在本地目录"edu_fineweb10B"中。
"""

import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset # pip install datasets
from tqdm import tqdm # pip install tqdm

# ------------------------------------------
# 配置参数
local_dir = "edu_fineweb10B"  # 本地保存目录
remote_name = "sample-10BT"   # 远程数据集名称
shard_size = int(1e8)         # 每个分片100M个token，总共100个分片

# 如果本地缓存目录不存在则创建
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# 下载数据集
fw = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train")

# 初始化tokenizer
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>'] # 文本结束标记

def tokenize(doc):
    """
    对单个文档进行标记化，返回uint16类型的numpy数组
    """
    tokens = [eot] # 使用特殊的<|endoftext|>标记分隔所有文档
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens)
    # 确保token值在uint16范围内
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token字典太大，超出uint16范围"
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16

def write_datafile(filename, tokens_np):
    """
    将token数组保存到文件
    """
    np.save(filename, tokens_np)

# 使用多进程处理所有文档，将输出分片保存，每个分片包含shard_size个token(最后一个分片可能包含剩余token)
nprocs = max(1, os.cpu_count()//2)  # 使用一半的CPU核心数
with mp.Pool(nprocs) as pool:
    shard_index = 0
    # 预分配缓冲区来存储当前分片
    all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
    token_count = 0
    progress_bar = None
    for tokens in pool.imap(tokenize, fw, chunksize=16):
        # 检查当前分片是否有足够空间存储新的token
        if token_count + len(tokens) < shard_size:
            # 直接将token追加到当前分片
            all_tokens_np[token_count:token_count+len(tokens)] = tokens
            token_count += len(tokens)
            # 更新进度条
            if progress_bar is None:
                progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"分片 {shard_index}")
            progress_bar.update(len(tokens))
        else:
            # 写入当前分片并开始新的分片
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
            # 将文档分割成适合当前分片的部分，剩余部分进入下一个分片
            remainder = shard_size - token_count
            progress_bar.update(remainder)
            all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
            write_datafile(filename, all_tokens_np)
            shard_index += 1
            progress_bar = None
            # 将当前文档的剩余部分填充到下一个分片
            all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
            token_count = len(tokens)-remainder

    # 将剩余的token写入最后一个分片
    if token_count != 0:
        split = "val" if shard_index == 0 else "train"
        filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
        write_datafile(filename, all_tokens_np[:token_count])
