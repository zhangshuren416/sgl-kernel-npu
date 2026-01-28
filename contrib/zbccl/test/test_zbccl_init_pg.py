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
    os.environ["ASCEND_LAUNCH_BLOCKING"] = "1"

    zbccl_set_logger_level(0)
    mem_128M = 128 * 1024 * 1024
    if not zbccl_init(world_size, local_rank, mem_128M):
        print(f"zbccl_init failed on rank {local_rank}.")
        return
    else:
        print(f"zbccl_init success on rank {local_rank}")

    group = dist.init_process_group("zbccl", rank=local_rank, world_size=world_size)
    print(f"init zbccl group success on rank {local_rank=} {world_size=}")
    try:
        in_tensor = torch.ones(6, dtype=torch.int32).npu()
        out_tensor = torch.zeros(6 * world_size, dtype=torch.int32).npu()

        dist.all_gather_into_tensor(out_tensor, in_tensor)
        # out_sum = torch.sum(out_tensor)
        # print(f"{out_sum=}")
        print(f"{out_tensor.shape=}")
        print(f"{out_tensor=}")

        # if out_sum != (world_size * torch.sum(in_tensor)):
        #     print(f"[ERROR] all gather result invalid.")
        # else:
        #     print(f"[SUCCESS] all gather reuslt correct.")
    finally:
        dist.destroy_process_group(group)

    if not zbccl_uninit():
        print("zbccl uninit failed.")


if __name__ == "__main__":
    test_init_zbccl_pg()
