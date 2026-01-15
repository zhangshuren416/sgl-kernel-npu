import os
import sys
import torch
import torch.distributed as dist
import torch_npu
from zbccl import process_group

torch_npu.npu.config.allow_internal_format = True

def test_init_zbccl_pg():
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"] or 2)
    torch.npu.set_device(local_rank)

    dist.init_process_group("npu:zbccl", rank=local_rank, world_size=world_size)
    print(f"init zbccl success on rank {local_rank=} {world_size=}")

    x = torch.ones(6).npu()
    dist.all_reduce(x)
    print(f"{x=}")
    # assert xxx


if __name__ == "__main__":
    test_init_zbccl_pg()