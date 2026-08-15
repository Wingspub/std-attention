"""字节级tokenizer"""
from typing import List
import pickle as pkl


class ByteTokenizer:
    def __init__(self, path: str | None = None) -> None:
        self.b2id = dict()
        self.id2b = dict()
        self.vocab_num = 0

        if isinstance(path, str):
            self.load(path)


    def train(self, byte_data: bytes) -> None:
        b2id, id2b = dict(), dict()
        data = list(set(byte_data))
        print("total vocab num: ", len(data))

        for idx, b in enumerate(data):
            b2id[b] = idx
            id2b[idx] = b

        self.b2id = b2id
        self.id2b = id2b
        self.vocab_num = len(data)


    def encode(self, text: bytes) -> List[int]:
        return [self.b2id[b] for b in text]


    def decode(self, ids: List[int]) -> bytes:
        return bytes([self.id2b[idx] for idx in ids])


    def save(self, path: str) -> None:
        metadata = {
            "b2id": self.b2id,
            "id2b": self.id2b,
            "vocab_num": self.vocab_num
        }

        f = open(path, 'wb')
        pkl.dump(metadata, f, protocol=pkl.HIGHEST_PROTOCOL)
        f.close()


    def load(self, path: str) -> None:
        try:
            f = open(path, "rb")
            metadata = pkl.load(f)
            self.b2id = metadata["b2id"]
            self.id2b = metadata["id2b"]
            self.vocab_num = metadata["vocab_num"]
            f.close()

        except Exception as e:
            print(e)

