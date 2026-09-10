import numpy as np
import os
from dataset.byte_tokenizer import ByteTokenizer
from dataset.enwik_dataset import EnwikDataset
from models.simplest_transformer import SimplestTransformer
from models.standard_transformer import STDConfig, STDTransformer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import optim
from torch.nn.functional import cross_entropy
from typing import Tuple, cast
import torch

print("this is a next token prediction task")

SEQ_LEN = 256
GEN_LEN = 128
batch_size = 32
iter_num = 200000
loss_print_num = 100
eval_num = 1000

device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')

## model
dims = 512
layer_num = 6
lr = 1e-4

# data init
dataset_name = "enwik_dataset"
root_path = os.path.dirname(os.path.dirname(__file__))
dataset_path = os.path.join(root_path, "data", dataset_name)
train_data = torch.from_numpy(np.fromfile(os.path.join(dataset_path, "train.bin"), dtype=np.uint8))
valid_data = torch.from_numpy(np.fromfile(os.path.join(dataset_path, "valid.bin"), dtype=np.uint8))
print(len(valid_data))
tokenizer = ByteTokenizer(os.path.join(dataset_path, "tokenizer.pkl"))

train_dataset = EnwikDataset(train_data, seq_len=SEQ_LEN)
valid_dataset = EnwikDataset(valid_data, seq_len=SEQ_LEN)
train_dataloader = DataLoader(train_dataset, batch_size=2*batch_size)
valid_dataset = DataLoader(valid_dataset, batch_size=batch_size)

# model
vocab_num = tokenizer.vocab_num
## Simplest
# model = SimplestTransformer(vocab_num=vocab_num, layers_num=layer_num, dims=dims).to(device)
## Std Transformers
config = STDConfig(vocab_num=vocab_num, layer_num=layer_num, embed_dims=dims, heads=4)
model = STDTransformer(config=config).to(device)

torch.set_float32_matmul_precision('high')
# model = torch.compile(model)
optimizer = optim.Adam(model.parameters(), lr=lr)


def train(model, seq_data: torch.Tensor, device: torch.device) -> Tuple[float, float, float]:
    '''模型训练'''
    model.train()
    # source data and target data
    seq_data = seq_data.to(device)
    X = seq_data[:, :-1]
    Y = seq_data[:, 1:]

    y_pred = cast(torch.Tensor, model(X)[0])   # [B, L, token_num]

    # loss = cast(torch.Tensor, loss_func(y_pred.reshape(-1, token_num), Y.reshape(-1)))
    loss = cross_entropy(y_pred.transpose(1, 2), Y)
    loss.backward()
    grad_norm_before = model.get_grad_norm()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    grad_norm_afert = model.get_grad_norm()
    optimizer.step()
    optimizer.zero_grad()

    return loss.cpu().item(), grad_norm_before, grad_norm_afert


@torch.inference_mode()
def eval(model, seq_data: torch.Tensor, gen_flag: bool, device: torch.device) -> Tuple[float, str, str]:
    model.eval()
    # CE loss
    seq_data = seq_data.to(device)
    X = seq_data[:, :-1]
    Y = seq_data[:, 1:]

    y_pred = cast(torch.Tensor, model(X)[0])
    loss = cross_entropy(y_pred.reshape(-1, vocab_num), Y.reshape(-1))

    # generate
    src_text = b""
    gen_text = b""
    if gen_flag:
        gen_bytes = model.generate(src_data=seq_data, gen_num=2*GEN_LEN, back_num=GEN_LEN, if_cache=False)
        src_bytes_text, gen_bytes_text = [c.item() for c in seq_data[0]], [c.item() for c in gen_bytes[0]]
        src_text, gen_text = tokenizer.decode(src_bytes_text), tokenizer.decode(gen_bytes_text)

    return loss.item(), src_text.decode("utf-8", errors="replace"), gen_text.decode("utf-8", errors="replace")


# record
writer = SummaryWriter("logs")

temp_step = 0
for data in train_dataloader:
    if (temp_step+1) % eval_num == 0:
        loss = []
        flag = True
        valid_num = 0
        for valid_data in valid_dataset:
            valid_num += 1
            valid_loss, src_text, gen_text = eval(model=model, seq_data=valid_data, gen_flag=flag, device=device)
            loss.append(valid_loss)
            if flag:
                print(f"[src_text]:\n{src_text}")
                print(f"[gen_text]:\n{gen_text}")
                flag = False
            if valid_num > loss_print_num:
                break
        valid_mean = np.mean(valid_loss)
        print(f"valid_loss:{valid_mean:.6f}")
        writer.add_scalar("Valid/loss", valid_mean, temp_step+1)

    loss, grad_norm_b, grad_norm_a = train(model=model, seq_data=data, device=device)
    writer.add_scalar("Train/loss", loss, temp_step+1)
    writer.add_scalar("Train/grad_norm_before", grad_norm_b, temp_step+1)
    writer.add_scalar("Train/grad_norm_after", grad_norm_a, temp_step+1)

    if temp_step % loss_print_num == 0: print(f"step:{temp_step}, loss:{loss:.6f}, grad_norm_before:{grad_norm_b:.6f}, grad_norm_after:{grad_norm_a:.6f}")

    temp_step += 1
    if temp_step >= iter_num:
        break

writer.close()