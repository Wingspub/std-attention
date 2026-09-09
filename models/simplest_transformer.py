from typing import cast, Literal, Tuple
import torch
from torch import nn
from time import time

class KV_Cache():
    def __init__(self, layer_num: int) -> None:
        # K:(B, L, d), V:(B, L, d)
        # DynamicCache from transformers
        self.kv_cache = [[[], []] for _ in range(layer_num)]


    def __getitem__(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        assert layer_id < len(self.kv_cache)
        return torch.concat(self.kv_cache[layer_id][0]), torch.concat(self.kv_cache[layer_id][1])


    def update(self, layer_id: int,  kv_vector: Tuple[torch.Tensor, torch.Tensor]) -> None:
        # k
        self.kv_cache[layer_id][0].append(kv_vector[0])

        # v
        self.kv_cache[layer_id][1].append(kv_vector[1])


class SimpleSequentialModelV0(nn.Module):
    def __init__(self, dims: int, layer_id: int):
        super().__init__()
        self.layer_id = layer_id
        self.V_trans = nn.Linear(dims, dims, bias=False)

        self.output_proj = nn.Linear(dims, dims, bias=False)


    def forward(self, input_seq_embeddings: torch.Tensor) -> torch.Tensor:
        # input shape -> (B, L, d)
        # output shape -> (B, L, d)

        L = input_seq_embeddings.shape[1]
        V = cast(torch.Tensor, self.V_trans(input_seq_embeddings))

        ## simplest casual matrix
        # A = torch.tril(torch.ones((L, L), device=V.device))

        ## Length mean
        A = torch.tril(torch.ones((L, L), device=V.device)) / torch.arange(1, L+1, device=V.device).unsqueeze(1)

        ## rand matrix
        # A = torch.tril(torch.rand((L, L), device=V.device) + 1e-6)
        # A = A / A.sum(dim=-1, keepdim=True)

        ## Length decay
        # A = torch.tril(torch.arange(1, 1+L, device=V.device).repeat((L, 1)) + 1e-6)
        # A = A / A.sum(dim=-1, keepdim=True)

        output = torch.matmul(A, V)
        output = self.output_proj(output)

        return output


class AdvancedSequentialModel(nn.Module):
    def __init__(self, dims: int, layer_id: int) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.W_Q = nn.Linear(dims, dims, bias=False)
        self.W_K = nn.Linear(dims, dims, bias=False)
        self.W_V = nn.Linear(dims, dims, bias=False)
        nn.init.normal_(self.W_Q.weight, mean=0, std=1e-8)
        nn.init.normal_(self.W_K.weight, mean=0, std=1e-8)
        nn.init.normal_(self.W_V.weight, mean=0, std=1e-8)
        self.output_proj = nn.Linear(dims, dims, bias=False)


    def forward(self, input_embs: torch.Tensor, if_cache: bool = False, kv_cache: KV_Cache | None = None):
        # input_embs (B, L, d)
        B, L, d = input_embs.shape

        if if_cache:
            ...
        else:
            query = cast(torch.Tensor, self.W_Q(input_embs))
            key = cast(torch.Tensor, self.W_K(input_embs))
            value = cast(torch.Tensor, self.W_V(input_embs))

        # weight
        ## V0 std softmax
        QK = query @ key.transpose(1, 2)
        causal_mask = torch.log(torch.tril(torch.ones((L, L), device=query.device)))
        A = torch.softmax(QK + causal_mask, dim=-1)

        # V0.1 not softmax
        # A = torch.tril(query @ key.transpose(1, 2) + 1e-6)
        # A = A / A.sum(dim=-1, keepdim=True)

        ## V1
        # QK = query @ key.transpose(1, 2)
        # A = torch.tril(QK / torch.arange(1, L+1, device=value.device).unsqueeze(1))

        output = A @ value
        output = self.output_proj(output)

        return output


class SimplestBlock(nn.Module):
    def __init__(self, dims: int, layer_id: int) -> None:
        super().__init__()
        # self.attention = SimpleSequentialModelV0(dims=dims, layer_id=layer_id)
        self.attention = AdvancedSequentialModel(dims=dims, layer_id=layer_id)

        self.FFN = nn.Sequential(
            nn.Linear(dims, dims),
            nn.ReLU(),
            nn.Linear(dims, dims)
        )

    def forward(self, input_embs: torch.Tensor, if_cache: bool = False, kv_cache: KV_Cache | None = None):

        # Attention
        temp_x = self.attention(input_embs, if_cache, kv_cache)
        x = input_embs + temp_x

        # FFN
        output = x + self.FFN(x)

        return output


class SimplestTransformer(nn.Module):
    def __init__(self, vocab_num: int, layers_num: int, dims: int):
        super().__init__()
        self.layers_num = layers_num

        self.embeddings = nn.Embedding(vocab_num, dims)
        self.output_trans = nn.Linear(dims, vocab_num)

        self.layers_block = nn.ModuleList()

        for id in range(layers_num):
            block = SimplestBlock(dims, id)
            self.layers_block.append(block)


    def forward(self, input_seq: torch.Tensor, if_cache: bool = False, kv_cache: KV_Cache | None = None) -> Tuple[torch.Tensor, KV_Cache|None]:
        x = self.embeddings(input_seq)

        for i in range(self.layers_num):
            x = self.layers_block[i](x, if_cache, kv_cache)

        output = self.output_trans(x)

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
        self.layers_block.eval()

        batch, src_len = src_data.shape
        assert gen_num >= 0 and back_num < src_len and back_num >= 0

        start_idx = src_len - back_num
        end_idx = start_idx + gen_num

        response = torch.zeros((batch, end_idx), dtype=torch.int32).to(src_data.device)
        response[:, :start_idx] = src_data[:, :start_idx]
        start = time()

        kv_cache = KV_Cache(self.layers_num) if if_cache else None

        # process
        for i in range(start_idx, end_idx):
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

            del output_pred

        end = time()
        rate = gen_num / (end-start)
        print(f"speed:{rate:.2f} tokens/s")
        return response