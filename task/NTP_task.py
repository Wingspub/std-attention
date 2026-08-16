import numpy as np
import os
from dataset.byte_tokenizer import ByteTokenizer
from dataset.enwik_dataset import EnwikDataset
from models.simplest_transformer import SimplestTransformer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import optim, nn
from torch.nn import CrossEntropyLoss
from typing import Tuple, cast
import torch
from time import time

print("this is a next token prediction task")

SEQ_LEN = 256
GEN_LEN = 128
batch_size = 64
iter_num = 200000
loss_print_num = 100
eval_num = 1000

device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')

## model
dims = 512
layer_num = 6
lr = 5e-4

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
train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
valid_dataset = DataLoader(valid_dataset, batch_size=batch_size)

# model
vocab_num = tokenizer.vocab_num
model = SimplestTransformer(vocab_num=vocab_num, layers_num=layer_num, dims=dims).to(device)

torch.set_float32_matmul_precision('high')
# model = torch.compile(model)
optimizer = optim.Adam(model.parameters(), lr=lr)
loss_func = CrossEntropyLoss()


def get_grad_norm(model, norm_type=2.0) -> float:
    """
    计算模型所有参数梯度的总范数。
    norm_type=2.0 表示 L2 范数，即常见的 grad_norm。
    """
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1.0 / norm_type)
    return total_norm


def train(model, seq_data: torch.Tensor, device: torch.device) -> Tuple[float, float, float]:
    '''模型训练'''
    model.train()
    # source data and target data
    seq_data = seq_data.to(device)
    X = seq_data[:, :-1]
    Y = seq_data[:, 1:]

    y_pred = cast(torch.Tensor, model(X))   # [B, L, token_num]

    # loss = cast(torch.Tensor, loss_func(y_pred.reshape(-1, token_num), Y.reshape(-1)))
    loss = cast(torch.Tensor, loss_func(y_pred.transpose(1, 2), Y))
    loss.backward()
    grad_norm_before = get_grad_norm(model)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    grad_norm_afert = get_grad_norm(model)
    optimizer.step()
    optimizer.zero_grad()

    return loss.cpu().item(), grad_norm_before, grad_norm_afert


def top_k(logits, thres = 0.9):
    k = int((1 - thres) * logits.shape[-1])
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs


@torch.inference_mode()
def generate(
    model,
    src_seq: torch.Tensor,
    seq_len: int,
    device: torch.device,
    temperature: float = 1.,
    filter_logits_fn = top_k,
    filter_thres: float = 0.9,
) -> Tuple[torch.Tensor, torch.Tensor]:
    '''生成字符'''
    model.eval()
    batch, src_len = src_seq.shape
    assert src_len <= seq_len

    gen_len = seq_len - src_len
    sub_len = src_len - gen_len

    response = torch.zeros((batch, seq_len), dtype=torch.int32).to(device)
    response[:, :sub_len] = src_seq[:, :sub_len]
    start = time()
    for i in range(sub_len, seq_len):
        output_pred = model(response[:, :i])[:, -1, :]
        # response[:, i] = torch.argmax(output_pred, dim=-1)
        filtered_logits = filter_logits_fn(output_pred, thres = filter_thres)
        probs = nn.functional.softmax(filtered_logits / temperature, dim=-1)
        sample = torch.multinomial(probs, 1)
        response[:, i] = sample.squeeze(1)
        del output_pred

    end = time()
    rate = (seq_len-sub_len) / (end-start)
    print(f"speed:{rate:.2f} tokens/s")
    return src_seq, response


@torch.inference_mode()
def eval(model, seq_data: torch.Tensor, gen_flag: bool, device: torch.device) -> Tuple[float, str, str]:
    model.eval()
    # CE loss
    seq_data = seq_data.to(device)
    X = seq_data[:, :-1]
    Y = seq_data[:, 1:]

    y_pred = cast(torch.Tensor, model(X))
    loss = cast(torch.Tensor, loss_func(y_pred.reshape(-1, vocab_num), Y.reshape(-1)))

    # generate
    src_text = b""
    gen_text = b""
    if gen_flag:
        src_bytes, gen_bytes = generate(model=model, src_seq=seq_data, seq_len=SEQ_LEN+GEN_LEN, device=device)
        src_bytes_text, gen_bytes_text = [c.item() for c in src_bytes[0]], [c.item() for c in gen_bytes[0]]
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