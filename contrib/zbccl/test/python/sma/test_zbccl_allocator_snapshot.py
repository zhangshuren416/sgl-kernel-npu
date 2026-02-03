import torch
import torch_npu
from torch import nn

import zbccl
from zbccl import record_memory_history, dump_snapshot
import sys
import os


def init():
    # This will allocate memory in the device using the new allocator
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    zbccl.zbccl_set_logger_level(2)
    mem = 1024 * 1024 * 1024
    if not zbccl.zbccl_init(world_size, local_rank, mem):
        print(f"zbccl_init failed on rank {local_rank}.")
        exit(-1)
    else:
        print(f"zbccl_init success on rank {local_rank}")


def train(num_iter=500, device="npu"):
    """a tiny transformer training process"""
    model = nn.Transformer(
        d_model=512, nhead=2, num_encoder_layers=2, num_decoder_layers=2
    ).to(device=device)
    x = torch.randn(size=(1, 1024, 512), device=device)
    tgt = torch.rand(size=(1, 1024, 512), device=device)
    model.train()
    labels = torch.rand_like(model(x, tgt))
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    for _i in range(num_iter):
        y = model(x, tgt)
        loss = criterion(y, labels)
        loss.backward()
        if _i % 20 == 0:
            print(f"[step{_i}] loss: {loss.item()}")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)


if __name__ == "__main__":
    init()

    npu_tensor = torch.zeros(10, device="npu")
    # print('mem info:', zbccl.mem_get_info())
    print(npu_tensor)

    record_memory_history("all", sys.maxsize)
    # full demo test
    train()
    # print('mem info:', zbccl.mem_get_info())

    new_snapshot = dump_snapshot()
    record_memory_history(None, 1)
    import pickle
    with open("snapshot.json", 'wb') as f:
        pickle.dump(new_snapshot, f)
