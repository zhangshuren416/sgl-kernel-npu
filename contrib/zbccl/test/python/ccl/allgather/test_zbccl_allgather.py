import os
import sys
import time
import torch
import torch.distributed as dist
import torch_npu
import numpy as np
from zbccl import zbccl_init, zbccl_uninit, zbccl_set_logger_level

torch_npu.npu.config.allow_internal_format = True

def test_zbccl_allgather():
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"] or 2)
    test_type = os.environ["TEST_TYPE"] or "int"
    current_dir = os.getenv("CURRENT_DIR", ".")
    case_num = int(os.environ["CASE_NUM"] or 16)
    os.environ["ASCEND_LAUNCH_BLOCKING"] = "1"
    device_id = local_rank

    type_map = {
        "int": np.int32,
        "int32_t": np.int32,
        "float16_t": np.float16,
        "float": np.float32,
        "bfloat16_t": np.float16
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
    local_mem = 256 * 1024 * 1024
    if not zbccl_init(world_size, device_id, local_rank, local_mem):
        print(f"zbccl_init failed on rank {local_rank}.")
        return
    else:
        print(f"zbccl_init success on rank {local_rank}\n")

    group = dist.init_process_group("zbccl", rank=local_rank, world_size=world_size)
    print(f"init zbccl group success on rank {local_rank=} {world_size=}")
    enable_profiling = os.getenv("ENABLE_PROFILING", "0") == "1"
    if enable_profiling:
        prof_cnt = 0
        experimental_config = torch_npu.profiler._ExperimentalConfig(
            aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
            profiler_level=torch_npu.profiler.ProfilerLevel.Level2,
            l2_cache=False,
            data_simplification=False,
        )
        profiling_path = f"{current_dir}/profiling/"
        prof = torch_npu.profiler.profile(
            activities=[
                torch_npu.profiler.ProfilerActivity.CPU,
                torch_npu.profiler.ProfilerActivity.NPU,
            ],
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                profiling_path
            ),
            schedule=torch_npu.profiler.schedule(
                    wait=1, warmup=1, active=10, repeat=1, skip_first=1
                ),
                record_shapes=True,
                profile_memory=True,
                with_stack=False,
                with_flops=False,
                with_modules=False,
                experimental_config=experimental_config,
        )
    try:
        ret = 0
        prof_cnt = 0
        if enable_profiling:
            prof.start()
        for i in range(0, case_num):
            for k in range(0, 50):
                if enable_profiling and prof_cnt > 1:
                    prof.step()
                data_len = 6 * (2 ** i)
                golden_dir = f"allgather_{data_len}_{world_size}"
                data = np.fromfile(f"{current_dir}/golden/{golden_dir}/input_gm_{local_rank}.bin", dtype=data_type)
                in_tensor = torch.from_numpy(data).to(tensor_data_type).npu()
                gold_data = np.fromfile(f"{current_dir}/golden/{golden_dir}/golden.bin", dtype=data_type)
                gold_tensor = torch.from_numpy(gold_data).to(tensor_data_type).npu()
                out_tensor = torch.zeros(data_len * world_size, dtype=tensor_data_type).npu()
                dist.all_gather_into_tensor(out_tensor, in_tensor)
                prof_cnt = prof_cnt + 1
                if not torch.allclose(gold_tensor, out_tensor, rtol=1e-4, atol=1e-8):
                    print(f"[ERROR] rank {local_rank}, case {i} allgather result not correct\n")
                    ret = 1
                    break
        if ret == 0:
            print(f"[INFO] rank {local_rank}, allgather run all case success\n")
        if enable_profiling:
            torch.npu.synchronize()
            prof.stop()
    finally:
        dist.destroy_process_group(group)

    if not zbccl_uninit():
        print("zbccl uninit failed.")


if __name__ == "__main__":
    test_zbccl_allgather()
