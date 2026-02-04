import os
import sys
import time
import torch
import torch.distributed as dist
from torch.distributed import ReduceOp
import torch_npu
import numpy as np
from ml_dtypes import bfloat16
from zbccl import zbccl_init, zbccl_uninit, zbccl_set_logger_level

torch_npu.npu.config.allow_internal_format = True

def test_zbccl_allreduce():
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"] or 2)
    test_type = os.environ["TEST_TYPE"] or "int"
    current_dir = os.getenv("CURRENT_DIR", ".")
    case_num = int(os.environ["CASE_NUM"] or 16)
    os.environ["ASCEND_LAUNCH_BLOCKING"] = "1"
    gpu_id = local_rank

    type_map = {
        "int": np.int32,
        "int32_t": np.int32,
        "float16_t": np.float16,
        "float": np.float32,
        "bfloat16_t": np.float16,
    }
    data_type = type_map.get(test_type, 'int')

    torch_type_map = {
        "int": torch.int32,
        "int32_t": torch.int32,
        "float16_t": torch.float16,
        "float": torch.float32,  
        "bfloat16_t": torch.bfloat16
    }

    tensor_data_type = torch_type_map.get(test_type, 'int')

    zbccl_set_logger_level(0)
    mem_128M = 256 * 1024 * 1024
    if not zbccl_init(world_size, gpu_id, local_rank, mem_128M):
        print(f"zbccl_init failed on rank {local_rank}.")
        return
    else:
        print(f"zbccl_init success on rank {local_rank}\n")

    group = dist.init_process_group("zbccl", rank=local_rank, world_size=world_size)
    print(f"init zbccl group success on rank {local_rank=} {world_size=}")
    try:
        ret = 0
        for i in range(0, case_num):
            data_len = 6 * (2 ** i)
            golden_dir = f"allreduce_{data_len}_{world_size}"
            data = np.fromfile(f"{current_dir}/golden/{golden_dir}/input_gm_{local_rank}.bin", dtype=data_type)
            tensor = torch.from_numpy(data).to(tensor_data_type).npu()
            gold_data = np.fromfile(f"{current_dir}/golden/{golden_dir}/golden.bin", dtype=data_type)
            gold_tensor = torch.from_numpy(gold_data).to(tensor_data_type).npu()
            dist.all_reduce(tensor, op=ReduceOp.SUM)
            if not torch.allclose(gold_tensor, tensor, rtol=1e-4, atol=1e-8):
                print(f"[ERROR] rank {local_rank}, case {i} allreduce result not correct\n")
                ret = 1
                break
        if ret == 0:
            print(f"[INFO] rank {local_rank}, allreduce run all case success\n")
    finally:
        dist.destroy_process_group(group)

    if not zbccl_uninit():
        print("zbccl uninit failed.")


if __name__ == "__main__":
    test_zbccl_allreduce()
