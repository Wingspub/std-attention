# (Llama) The Llama 3 Herd of Models
# http://arxiv.org/abs/2407.21783

# (RoPE) RoFormer: Enhanced Transformer with Rotary Position Embedding
# http://arxiv.org/abs/2104.09864

# (GLU) GLU Variants Improve Transformer
# https://arxiv.org/abs/2002.05202

# (RMSNorm) Root Mean Square Layer Normalization
# https://arxiv.org/abs/1910.07467

# (SiLU) Swish: a Self-Gated Activation Function
# https://arxiv.org/abs/1710.05941


from typing import cast, Tuple, Literal

from torch import nn
from torch.nn.functional import scaled_dot_product_attention
import torch
from time import time
from .utils import KV_Cache


def precompute_rope_freqs(dim: int, max_seq_length: int = 32*1024, rope_base: float = 1e6) -> Tuple[torch.Tensor, torch.Tensor]:
    freqs, atten_factor = 1.0/ (rope_base ** (torch.arange(0, dim, 2)[: dim // 2].float() / dim)), 1.0
    t = torch.arange(max_seq_length, device=freqs.device)

    # Length * dim
    freqs = torch.outer(t, freqs).float()

    freqs_cos = torch.cos(freqs.repeat(1, 2)) * atten_factor
    freqs_sin = torch.sin(freqs.repeat(1, 2)) * atten_factor

    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos_weight: torch.Tensor, sin_weight: torch.Tensor, unsqueeze_dim=1) -> Tuple[torch.Tensor, torch.Tensor]:
    def rotate_half(x): return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)
    q_embed = ((q * cos_weight.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin_weight.unsqueeze(unsqueeze_dim))).to(q.dtype)
    k_embed = ((k * cos_weight.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin_weight.unsqueeze(unsqueeze_dim))).to(k.dtype)
    return q_embed, k_embed


class ModernConfig:
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


class RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dims))


    def norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(torch.mean(x.pow(2), dim=-1, keepdim=True) + self.eps)


    def forward(self, input_seq_emb: torch.Tensor) -> torch.Tensor:
        return (self.weight * self.norm(input_seq_emb.float())).type_as(input_seq_emb)


class FFN(nn.Module):
    def __init__(self, dims: int, hidden_dims: int) -> None:
        super().__init__()

        self.gated = nn.Linear(dims, hidden_dims, bias=False)
        self.up = nn.Linear(dims, hidden_dims,bias=False)
        self.down = nn.Linear(hidden_dims, dims, bias=False)
        self.activateion = nn.SiLU()


    def forward(self, input_seq_embs: torch.Tensor) -> torch.Tensor:
        return self.down(self.activateion(self.gated(input_seq_embs) * self.up(input_seq_embs)))


class MultiHeadAttention(nn.Module):
    def __init__(self, config: ModernConfig, layer_id: int) -> None:
        super().__init__()
        if_bias = config.if_bias
        embed_dims = config.embed_dims
        p = config.p

        self.heads = config.heads
        self.heads_dims = embed_dims // self.heads

        self.W_Q = nn.Linear(embed_dims, embed_dims, bias=if_bias)
        self.W_K = nn.Linear(embed_dims, embed_dims, bias=if_bias)
        self.W_V = nn.Linear(embed_dims, embed_dims, bias=if_bias)
        self.q_norm = RMSNorm(self.heads_dims)
        self.k_norm = RMSNorm(self.heads_dims)

        self.score_dropout = nn.Dropout(p=p)
        self.residual_dropout = nn.Dropout(p=p)

        # output_proj
        self.output_proj = nn.Linear(embed_dims, embed_dims, bias=if_bias)


    def forward(self, input_embs: torch.Tensor, position_embeddings: Tuple[torch.Tensor, torch.Tensor], is_causal: bool=False) -> torch.Tensor:
        B, L, d = input_embs.shape

        query = cast(torch.Tensor, self.W_Q(input_embs)).reshape(B, L, self.heads, self.heads_dims)
        key = cast(torch.Tensor, self.W_K(input_embs)).reshape(B, L, self.heads, self.heads_dims)
        value = cast(torch.Tensor, self.W_V(input_embs)).reshape(B, L, self.heads, self.heads_dims)
        query, key = self.q_norm(query), self.k_norm(key)

        cos, sin = position_embeddings
        query, key = apply_rotary_pos_emb(query, key, cos, sin, unsqueeze_dim=1)
        query, key, value = query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)

        # A: score matrix
        QK = torch.matmul(query, key.transpose(-1, -2)) / (self.heads_dims)**0.5
        mask =  torch.log(torch.tril(torch.ones((query.shape[-2], key.shape[-2]), device=query.device), diagonal=key.shape[-2]-query.shape[-2]))
        A = torch.softmax(QK + mask, dim=-1)
        output = torch.matmul(self.score_dropout(A), value).transpose(1, 2).contiguous().reshape(B, -1, d)

        ## fast impl
        # scaled_dot_product_attention

        # output
        output = self.residual_dropout(self.output_proj(output))

        return output


class AttentionBlock(nn.Module):
    def __init__(self, config: ModernConfig, layer_id: int):
        super().__init__()

        embed_dims = config.embed_dims
        self.attention = MultiHeadAttention(config=config, layer_id=layer_id)
        self.LN1 = RMSNorm(embed_dims)
        self.FFN = FFN(dims=embed_dims, hidden_dims=2*embed_dims)
        self.LN2 = RMSNorm(embed_dims)


    def forward(self, input_embs: torch.Tensor, position_embeddings: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        # Attention
        normed = self.LN1(input_embs)
        embs = self.attention(normed, position_embeddings=position_embeddings, is_causal=True)
        x = input_embs + embs

        # FFN
        normed = self.LN2(x)
        embs = self.FFN(normed)
        output_embs = x + embs

        return output_embs


class ModernModel(nn.Module):
    def __init__(self, config: ModernConfig) -> None:
        super().__init__()
        layer_num = config.layer_num

        self.layer_list = nn.ModuleList()
        for layer_id in range(layer_num):
            transformerBlock = AttentionBlock(config=config, layer_id=layer_id)
            self.layer_list.append(transformerBlock)


    def forward(self, input_embs: torch.Tensor, position_embs: torch.Tensor):
        for layer in self.layer_list:
            embeddings = layer(input_embs, position_embs)

        return embeddings


class ModernTransformer(nn.Module):
    def __init__(self, config: ModernConfig):
        super().__init__()
        vocab_num = config.vocab_num
        embed_dims = config.embed_dims
        head_dims = embed_dims // config.heads

        self.layer_num = config.layer_num

        self.embeddings = nn.Embedding(vocab_num, embed_dims)
        freqs_cos, freqs_sin = precompute_rope_freqs(dim=head_dims)
        self.freqs_cos = nn.Buffer(freqs_cos, persistent=False)
        self.freqs_sin = nn.Buffer(freqs_sin, persistent=False)

        self.model = ModernModel(config=config)

        self.output_proj = nn.Linear(embed_dims, vocab_num, bias=config.if_bias)


    def forward(self, input_idx: torch.Tensor) -> torch.Tensor:
        # 编码与位置编码
        L = input_idx.shape[1]
        embeddings = self.embeddings(input_idx)
        position_embeddings = (self.freqs_cos[:L], self.freqs_sin[:L])

        # layer
        embeddings = self.model(embeddings, position_embeddings)

        # output
        output = self.output_proj(embeddings)

        return output


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