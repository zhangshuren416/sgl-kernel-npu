import sys
from pathlib import Path

import torch
import torch_npu
import pickle
from zbccl import switch_to_allocator, init_shmem, record_memory_history, dump_snapshot


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
    switch_to_allocator()
    torch.npu.set_device(0)
    torch.npu.init()
    init_shmem(0, 1, 30 * (1024 ** 3), 0, 'tcp://127.0.0.1:3399',True)
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

