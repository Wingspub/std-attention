import numpy as np
import os

# settings
train_vaild_split_rate = 0.9

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
    data = np.frombuffer(text, dtype=np.uint8)
    print(len(data))

# 字符统计得到 -> tokenizer




# 映射得到 -> ID
# 拆分得到trian 和 valid数据集







