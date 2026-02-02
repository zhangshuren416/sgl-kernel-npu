import os
import sys
import time
import torch
import torch.distributed as dist
import torch_npu
from zbccl import zbccl_init, zbccl_uninit, zbccl_set_logger_level

torch_npu.npu.config.allow_internal_format = True

def test_init_zbccl_pg():
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"] or 4)

    zbccl_set_logger_level(0)
    mem_128M = 256 * 1024 * 1024
    if not zbccl_init(world_size, local_rank, mem_128M):
        print(f"zbccl_init failed on rank {local_rank}.")
        return
    else:
        print(f"zbccl_init success on rank {local_rank}\n")

    dist.init_process_group("zbccl", rank=local_rank, world_size=world_size)
    global_group = dist.group.WORLD
    backend = global_group._get_backend(torch.device("npu", local_rank))
    global_group_name = backend.get_zbccl_comm_name()

    global_group2 = dist.new_group(list(range(world_size)), backend="zbccl")
    backend2 = global_group2._get_backend(torch.device("npu", local_rank))
    global_group_name2 = backend2.get_zbccl_comm_name()

    sub_group_rank = [0, 2]
    sub_group_name = ""
    if local_rank in sub_group_rank:
        sub_group = dist.new_group(sub_group_rank, backend="zbccl")
        backend = sub_group._get_backend(torch.device("npu", local_rank))
        sub_group_name = backend.get_zbccl_comm_name()
        dist.destroy_process_group(sub_group)
        del sub_group
    print(f"init zbccl group success on rank {local_rank=} {world_size=} \
        {global_group_name=} {global_group_name2=} {sub_group_name=}")

    try:
        success_cnt = 0
        total_cnt = 10
        for k in range(1, total_cnt + 1):
            print(f"[INFO] rank {local_rank}, round {k} start\n")
            nelems = 6 * k
            torch.manual_seed(int(time.time())+ k * 1000)
            in_tensor = torch.rand(1, nelems, device='npu', dtype=torch.float32) * 10
            print(f"{in_tensor=}")
            out_tensor = torch.zeros(nelems * world_size, dtype=torch.float32).npu()
            gold_tensor = torch.zeros(nelems * world_size, dtype=torch.float32).npu()
            for i in range(0, world_size):
                gold_tensor[i * nelems: (i + 1) * nelems] = in_tensor
            dist.all_gather_into_tensor(out_tensor, in_tensor)
            print(f"{out_tensor=}")
            if not torch.allclose(gold_tensor, out_tensor, rtol=1e-4, atol=1e-8):
                print(f"[ERROR] rank {local_rank}, round {k} all gather result not correct\n")
                break
            else:
                print(f"[SUCCESS] rank {local_rank}, round {k} all gather reuslt correct\n")
                success_cnt += 1
        print(f"{success_cnt}/{total_cnt} tests run success")
    finally:
        dist.destroy_process_group(global_group)
        del global_group

    if not zbccl_uninit():
        print("zbccl uninit failed.")


if __name__ == "__main__":
    test_init_zbccl_pg()
