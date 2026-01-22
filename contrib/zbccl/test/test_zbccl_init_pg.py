import os
import sys
import torch
import torch.distributed as dist
import torch_npu
from zbccl import zbccl_init, zbccl_uninit, zbccl_set_logger_level

torch_npu.npu.config.allow_internal_format = True

def test_init_zbccl_pg():
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"] or 2)
    torch.npu.set_device(local_rank)

    zbccl_set_logger_level(0)
    mem_128M = 128 * 1024 * 1024
    zbccl_init(world_size, local_rank, mem_128M)

    dist.init_process_group("zbccl", rank=local_rank, world_size=world_size)
    print(f"init zbccl success on rank {local_rank=} {world_size=}")

    in_tensor = torch.ones(6).npu()
    out_tensor = torch.zeros(6 * world_size).npu()

    dist.all_gather_into_tensor(out_tensor, in_tensor,)
    print(f"{out_tensor=}")

    assert torch.sum(out_tensor) == 6 * world_size
    zbccl_uninit()


if __name__ == "__main__":
    test_init_zbccl_pg()
