import os
import time
import random
import torch
import torch.distributed as dist
import torch_npu
from zbccl import zbccl_init, zbccl_uninit, zbccl_set_logger_level

torch_npu.npu.config.allow_internal_format = True

def test_init_zbccl_pg():
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"] or 4)
    gpu_id = local_rank

    zbccl_set_logger_level(0)
    local_mem = 256 * 1024 * 1024
    if not zbccl_init(world_size, gpu_id, local_rank, local_mem):
        print(f"zbccl_init failed on rank {local_rank}.")
        return
    else:
        print(f"zbccl_init success on rank {local_rank}\n")

    # init process group
    dist.init_process_group("zbccl", rank=local_rank, world_size=world_size)
    global_group = dist.group.WORLD
    backend = global_group._get_backend(torch.device("npu", local_rank))
    global_group_name = backend.get_zbccl_comm_name()

    # create a group with same ranks of global group
    global_group2 = dist.new_group(list(range(world_size)), backend="zbccl")
    backend2 = global_group2._get_backend(torch.device("npu", local_rank))
    global_group_name2 = backend2.get_zbccl_comm_name()

    # create a sub group
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

    sleep_time = random.uniform(0, 4)
    time.sleep(sleep_time)
    dist.barrier()
    print(f"after barrier rank={local_rank} sleep={sleep_time}s finish at time={time.time()}")

    dist.destroy_process_group(global_group)
    del global_group

    if not zbccl_uninit():
        print("zbccl uninit failed.")


if __name__ == "__main__":
    test_init_zbccl_pg()
