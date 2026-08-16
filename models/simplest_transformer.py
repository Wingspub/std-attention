from typing import cast
import torch
from torch import nn


class SimpleSequentialModelV0(nn.Module):
    def __init__(self, dims: int):
        super().__init__()
        self.V_trans = nn.Linear(dims, dims, bias=False)

        self.output_proj = nn.Linear(dims, dims, bias=False)


    def forward(self, input_seq_embeddings: torch.Tensor) -> torch.Tensor:
        # input shape -> (B, L, d)
        # output shape -> (B, L, d)

        L = input_seq_embeddings.shape[1]
        V = cast(torch.Tensor, self.V_trans(input_seq_embeddings))

        # simplest casual matrix
        # A = torch.tril(torch.ones((L, L), device=V.device)) / torch.arange(1, L+1, device=V.device).unsqueeze(1)

        ## rand matrix
        # A = torch.tril(torch.rand((L, L), device=V.device) + 1e-6)
        # A = A / A.sum(dim=-1, keepdim=True)

        ## Length decay
        A = torch.tril(torch.arange(1, 1+L, device=V.device).repeat((L, 1)) + 1e-6)
        A = A / A.sum(dim=-1, keepdim=True)


        output = torch.matmul(A, V)
        output = self.output_proj(output)

        return output


class AdvancedSequentialModel(nn.Module):
    def __init__(self, dims: int) -> None:
        super().__init__()
        self.W_Q = nn.Linear(dims, dims, bias=False)
        self.W_K = nn.Linear(dims, dims, bias=False)
        self.W_V = nn.Linear(dims, dims, bias=False)
        nn.init.zeros_(self.W_Q.weight)
        nn.init.zeros_(self.W_K.weight)
        nn.init.zeros_(self.W_V.weight)
        self.output_proj = nn.Linear(dims, dims, bias=False)


    def forward(self, input_embs: torch.Tensor):
        # input_embs (B, L, d)
        B, L, _ = input_embs.shape
        query = cast(torch.Tensor, self.W_Q(input_embs))
        key = cast(torch.Tensor, self.W_K(input_embs))
        value = cast(torch.Tensor, self.W_V(input_embs))

        # weight
        ## V0 std softmax
        # causal_mask = torch.log(torch.tril(torch.ones((L, L), device=query.device)))
        # A = torch.softmax(query @ key.transpose(1, 2) + causal_mask, dim=-1)

        ## V0.1 not softmax
        # A = torch.tril(query @ key.transpose(1, 2) + 1e-6)
        # A = A / A.sum(dim=-1, keepdim=True)

        ## V1
        A = query @ key.transpose(1, 2) / torch.arange(1, L+1, device=value.device).unsqueeze(1)

        output = A @ value
        output = self.output_proj(output)

        return output


class SimplestBlock(nn.Module):
    def __init__(self, dims: int) -> None:
        super().__init__()
        self.attention = SimpleSequentialModelV0(dims=dims)
        # self.attention = AdvancedSequentialModel(dims=dims)

        self.FFN = nn.Sequential(
            nn.Linear(dims, dims),
            nn.ReLU(),
            nn.Linear(dims, dims)
        )

    def forward(self, input_embs: torch.Tensor):

        # Attention
        # print(self.attention(input_embs), self.attention(input_embs).shape)
        x = input_embs + self.attention(input_embs)


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

        for _ in range(layers_num):
            block = SimplestBlock(dims)
            self.layers_block.append(block)


    def forward(self, input_seq: torch.Tensor) -> torch.Tensor:
        x = self.embeddings(input_seq)

        for i in range(self.layers_num):
            x = self.layers_block[i](x)


        output = self.output_trans(x)

        return output