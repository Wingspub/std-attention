import os
from dataset.byte_tokenizer import ByteTokenizer
import numpy as np

# settings
train_vaild_split_rate = 0.99

# data loading
ROOT = os.path.dirname(__file__)
## enwik8(0.1B bytes):http://mattmahoney.net/dc/enwik8.zip
# import gzip
# input_file_path = os.path.join(ROOT, 'enwik8.gz')
# with gzip.open(input_file_path, 'r') as f:
#     text = f.read()
#     data = np.frombuffer(text, dtype=np.uint8)

## enwik9(1B bytes):http://mattmahoney.net/dc/enwik9.zip
input_file_path = os.path.join(ROOT, 'enwik9')
with open(input_file_path, 'br') as f:
    text = f.read()

n = len(text)
print(n)

# 字符统计得到 -> tokenizer
tokenzier = ByteTokenizer()
tokenzier.train(byte_data=text)
tokenzier_path = os.path.join(ROOT, "tokenizer.pkl")
tokenzier.save(tokenzier_path)

# 拆分得到trian 和 valid数据集
train_data_path = os.path.join(ROOT, "train.bin")
valid_data_path = os.path.join(ROOT, "valid.bin")
train_idx = int(n * train_vaild_split_rate)
train_data = np.array(tokenzier.encode(text[:train_idx]), dtype=np.uint8).tofile(train_data_path)
valid_data = np.array(tokenzier.encode(text[train_idx:]), dtype=np.uint8).tofile(valid_data_path)
