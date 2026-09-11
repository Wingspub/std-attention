from typing import Tuple
import torch

class KV_Cache:
    def __init__(self, layer_num: int) -> None:
        # K:(B, L, d), V:(B, L, d)
        # DynamicCache from transformers
        self.kv_cache = [[[], []] for _ in range(layer_num)]


    def __getitem__(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        assert layer_id < len(self.kv_cache)
        if len(self.kv_cache[layer_id][0]) == 0 and len(self.kv_cache[layer_id][1]) == 0:
            return torch.Tensor([]), torch.Tensor([])
        else:
            return torch.concat(self.kv_cache[layer_id][0], dim=1), torch.concat(self.kv_cache[layer_id][1], dim=1)


    def update(self, layer_id: int,  kv_vector: Tuple[torch.Tensor, torch.Tensor]) -> None:
        # k
        self.kv_cache[layer_id][0].append(kv_vector[0])

        # v
        self.kv_cache[layer_id][1].append(kv_vector[1])


    def get_kv_len(self) -> int:
        if len(self.kv_cache[0][0]) == 0 and len(self.kv_cache[0][1]) == 0:
            return 0
        else:
            return torch.concat(self.kv_cache[0][0], dim=1).shape[1]
