import sys
from pathlib import Path

import torch
import torch_npu
import pickle
import os
import zbccl
from zbccl import record_memory_history, dump_snapshot


def init():
    # This will allocate memory in the device using the new allocator
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    zbccl.zbccl_set_logger_level(0)
    mem_128M = 128 * 1024 * 1024
    if not zbccl.zbccl_init(world_size, local_rank, mem_128M):
        print(f"zbccl_init failed on rank {local_rank}.")
        exit(-1)
    else:
        print(f"zbccl_init success on rank {local_rank}")


def malloc(size, stream):
    with torch.npu.stream(stream):
        return torch.npu.caching_allocator_alloc(size)


def free(addr):
    torch.npu.caching_allocator_delete(addr)


def load_pickle_snapshot(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data


if __name__ == '__main__':
    ori_pickle_path = Path(sys.argv[1])
    new_pickle_path = f"{ori_pickle_path.stem}_replay{ori_pickle_path.suffix}"

    addr_map = {}
    stream_map = {}

    ori_snapshot = load_pickle_snapshot(ori_pickle_path.resolve())
    device_traces = next((l for l in ori_snapshot['device_traces'] if len(l) > 2000), None)
    #device_traces = ori_snapshot['device_traces'][8]

    init()  # init zbccl(including switch to dma/sma)

    record_memory_history("all", sys.maxsize)

    for idx, te in enumerate(device_traces):
        action = te['action']
        size = te['size']
        stream_id = te['stream']
        addr = te['addr']

        if action == "alloc" or action == 'empty_cache' or action == 'free_completed':
            if action == 'alloc':
                stream = stream_map.setdefault(stream_id, torch.npu.Stream())
                addr_map[addr] = malloc(size, stream)
            elif action == 'free_completed':
                free(addr_map[addr])
                del addr_map[addr]
            elif action == 'empty_cache':
                torch.npu.empty_cache()

    new_snapshot = dump_snapshot()
    record_memory_history(None, 1)
    with open(new_pickle_path, 'wb') as f:
        pickle.dump(new_snapshot, f)

