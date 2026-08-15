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
        A = torch.tril(torch.ones((L, L), device=V.device)) / torch.arange(1, L+1, device=V.device).unsqueeze(1)
        output = torch.matmul(A, V)
        output = self.output_proj(output)

        return output


class AdvancedSequentialModel(nn.Module):
    def __init__(self, dims: int) -> None:
        super().__init__()







    def forward(self, input_embs: torch.Tensor):
        



        ...


class SimplestBlock(nn.Module):
    def __init__(self, dims: int) -> None:
        super().__init__()
        self.attention = SimpleSequentialModelV0(dims=dims)

        self.FFN = nn.Sequential(
            nn.Linear(dims, dims),
            nn.ReLU(),
            nn.Linear(dims, dims)
        )

    def forward(self, input_embs: torch.Tensor):

        # Attention
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