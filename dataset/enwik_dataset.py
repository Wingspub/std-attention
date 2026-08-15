import torch
from torch.utils.data import IterableDataset

class EnwikDataset(IterableDataset):
    def __init__(self, data: torch.Tensor, seq_len: int):
        super().__init__()
        self.data = data.to(torch.int64)
        self.data_len = len(data)
        self.seq_len = seq_len


    def __iter__(self):
        while True:
            start_index = torch.randint(0, self.data_len-self.seq_len-1, (1,))
            seq = self.data[start_index: start_index + self.seq_len]
            yield seq