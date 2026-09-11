"""
Attention Is All You Need
http://arxiv.org/abs/1706.03762
"""

from typing import cast, Literal, Tuple
from torch import nn
import torch
from time import time
from .utils import KV_Cache

class STDConfig:
    vocab_num  = 256
    layer_num  = 6
    embed_dims = 512
    heads = 6
    if_bias=False
    p = 0.0

    def __init__(
        self,
        vocab_num=256,
        layer_num=6,
        embed_dims=512,
        heads=6,
        if_bias=False,
        p=0.0
    ) -> None:
        self.vocab_num = vocab_num
        self.layer_num = layer_num
        self.embed_dims = embed_dims
        self.heads = heads
        self.if_bias = if_bias
        self.p = p

        # validate
        heads_dims = embed_dims // heads
        assert embed_dims == heads_dims * heads


class MultiHeadAttention(nn.Module):
    def __init__(self, config: STDConfig, layer_id: int):
        super().__init__()
        embed_dims = config.embed_dims
        if_bias = config.if_bias
        p = config.p

        self.layer_id = layer_id
        self.heads = config.heads
        self.heads_dims = embed_dims // self.heads

        self.W_Q = nn.Linear(embed_dims, embed_dims, bias=if_bias)
        self.W_K = nn.Linear(embed_dims, embed_dims, bias=if_bias)
        self.W_V = nn.Linear(embed_dims, embed_dims, bias=if_bias)

        self.score_dropout = nn.Dropout(p=p)
        self.residual_dropout = nn.Dropout(p=p)

        # output_tran
        self.output_proj = nn.Linear(embed_dims, embed_dims)


    def forward(self, input_embs: torch.Tensor, if_cache: bool = False, kv_cache: KV_Cache | None = None) -> Tuple[torch.Tensor, KV_Cache|None]:
        B, L, d = input_embs.shape

        query = cast(torch.Tensor, self.W_Q(input_embs))
        key = cast(torch.Tensor, self.W_K(input_embs))
        value = cast(torch.Tensor, self.W_V(input_embs))

        if if_cache and kv_cache is not None:
            kv_cache.update(self.layer_id, (key, value))
            key, value = kv_cache[self.layer_id]


        query = query.reshape(B, -1, self.heads, self.heads_dims).transpose(1, 2)
        key = key.reshape(B, -1, self.heads, self.heads_dims).transpose(1, 2)
        value = value.reshape(B, -1, self.heads, self.heads_dims).transpose(1, 2)

        # A: score matrix
        QK = torch.matmul(query, key.transpose(-1, -2)) / self.heads_dims ** 0.5
        mask =  torch.log(torch.tril(torch.ones((query.shape[-2], key.shape[-2]), device=query.device), diagonal=key.shape[-2]-query.shape[-2]))
        A = torch.softmax(QK + mask, dim=-1)
        output = torch.matmul(self.score_dropout(A), value).transpose(1, 2).contiguous().reshape(B, -1, d)

        # output
        output = self.residual_dropout(self.output_proj(output))

        return output, kv_cache


class TransformerBlock(nn.Module):
    def __init__(self, config: STDConfig, layer_id: int):
        super().__init__()
        if_bias = config.if_bias
        embed_dims = config.embed_dims

        self.attention = MultiHeadAttention(config=config, layer_id=layer_id)
        self.LN1 = nn.LayerNorm(embed_dims)
        self.FFN = nn.Sequential(
            nn.Linear(embed_dims, embed_dims*4, bias=if_bias),
            nn.ReLU(),
            nn.Linear(embed_dims*4, embed_dims, bias=if_bias)
        )
        self.LN2 = nn.LayerNorm(embed_dims)


    def forward(self, input_embs: torch.Tensor, if_cache: bool = False, kv_cache: KV_Cache | None = None) -> Tuple[torch.Tensor, KV_Cache|None]:
        # Attention
        embs, kv_cache = self.attention(input_embs, if_cache, kv_cache)
        embs = self.LN1(embs)
        output_embs = input_embs + embs

        # FFN
        embs = self.FFN(output_embs)
        embs = self.LN2(embs)
        output_embs = output_embs + embs

        return output_embs, kv_cache


class STDModel(nn.Module):
    def __init__(self, config: STDConfig) -> None:
        super().__init__()
        layer_num = config.layer_num
        self.layer_num = layer_num

        self.layer_block = nn.ModuleList()
        for layer_id in range(layer_num):
            transformerBlock = TransformerBlock(config, layer_id)
            self.layer_block.append(transformerBlock)

    def forward(self, input_embs: torch.Tensor, if_cache: bool = False, kv_cache: KV_Cache | None = None):

        for layer in self.layer_block:
            x, kv_cache = layer(input_embs, if_cache, kv_cache)

        return x, kv_cache


class STDTransformer(nn.Module):
    def __init__(self, config: STDConfig):
        super().__init__()
        vocab_num = config.vocab_num
        embed_dims = config.embed_dims
        self.layer_num = config.layer_num
        self.embed_dims = config.embed_dims

        self.embeddings = nn.Embedding(vocab_num, embed_dims)
        self.model = STDModel(config)
        self.output_proj = nn.Linear(embed_dims, vocab_num)


    def position_vector(self, seq_len: int, base: float=10000) -> torch.Tensor:
        """
        input_embs: (X, L, d) -> position_embs (X, L(1:L), d((sin, cos), ...))

        PE_{i, 2j}   = sin(i / 10000^{2j/d})
        PE_{i, 2j+1} = cos(i / 10000^{2j/d})

        """
        position_index = torch.arange(seq_len).unsqueeze(1)
        dim_index = torch.arange(self.embed_dims)
        dim_index[::2] = dim_index[::2] / 2
        dim_index[1::2] = (dim_index[1::2] - 1) / 2
        coef = base ** (dim_index.unsqueeze(0)/self.embed_dims)
        position_matrix = position_index/coef

        position_matrix[:, ::2] = torch.sin(position_matrix[:, ::2])
        position_matrix[:, 1::2] = torch.cos(position_matrix[:, 1::2])

        return position_matrix


    def forward(self, input_ids: torch.Tensor, if_cache: bool = False, kv_cache: KV_Cache | None = None) -> Tuple[torch.Tensor, KV_Cache|None]:
        # 编码与位置编码
        embeddings = self.embeddings(input_ids)
        if if_cache and kv_cache is not None:
            L = kv_cache.get_kv_len() + input_ids.shape[1]
            position_embeddings = self.position_vector(L).to(embeddings.device)
            embeddings = embeddings + position_embeddings[-input_ids.shape[1]:]
        else:
            L = input_ids.shape[1]
            position_embeddings = self.position_vector(L).to(embeddings.device)
            embeddings = embeddings + position_embeddings

        # model
        embeddings, kv_cache = self.model(embeddings, if_cache, kv_cache)

        # output
        output = self.output_proj(embeddings)

        return output, kv_cache


    @staticmethod
    def top_k(logits: torch.Tensor, thres = 0.9) -> torch.Tensor:
        k = int((1 - thres) * logits.shape[-1])
        val, ind = torch.topk(logits, k)
        probs = torch.full_like(logits, float('-inf'))
        probs.scatter_(1, ind, val)
        return probs


    def get_grad_norm(self, norm_type=2.0) -> float:
        """
        计算模型所有参数梯度的总范数。
        norm_type=2.0 表示 L2 范数，即常见的 grad_norm。
        """
        total_norm = 0.0
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(norm_type)
                total_norm += param_norm.item() ** norm_type
        total_norm = total_norm ** (1.0 / norm_type)
        return total_norm


    @torch.inference_mode()
    def generate(self, src_data: torch.Tensor, gen_num: int = 0, back_num: int = 0, if_cache: bool = False, mode: Literal["greedy", "sample"] = "sample"):
        self.model.eval()

        batch, src_len = src_data.shape
        assert gen_num >= 0 and back_num < src_len and back_num >= 0

        start_idx = src_len - back_num
        end_idx = start_idx + gen_num

        response = torch.zeros((batch, end_idx), dtype=torch.int32).to(src_data.device)
        response[:, :start_idx] = src_data[:, :start_idx]
        start = time()

        kv_cache = KV_Cache(self.layer_num) if if_cache else None

        # process
        for i in range(start_idx, end_idx):
            if if_cache:
                inp =  response[:, :i] if i == start_idx else response[:, i-1:i]
                pred, kv_cache = self.forward(inp, if_cache, kv_cache)
            else:
                pred, kv_cache = self.forward(response[:, :i], if_cache, kv_cache)
            output_pred = pred[:, -1, :]

            if mode == "greedy":
                response[:, i] = torch.argmax(output_pred, dim=-1)
            elif mode == "sample":
                top_p = 0.9
                temperature = 1.0
                filtered_logits = self.top_k(output_pred, thres = top_p)
                probs = nn.functional.softmax(filtered_logits / temperature, dim=-1)
                sample = torch.multinomial(probs, 1)
                response[:, i] = sample.squeeze(1)


        end = time()
        rate = gen_num / (end-start)
        print(f"speed:{rate:.2f} tokens/s")
        return response