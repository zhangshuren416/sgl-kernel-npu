
from .zbccl_module import zbccl_init, zbccl_uninit
from .zbccl_module import switch_to_allocator, zbccl_get_shmem_base_addr
from .zbccl_module import mem_get_info
from .zbccl_module import __version__

from zbccl.zbccl.allocator import record_memory_history, dump_snapshot, simulate_init
from zbccl.zbccl import ProcessGroupZBCCL
from zbccl.zbccl import ZBCCLBootstrapType, zbccl_set_logger_level
import torch, torch_npu

from zbccl.zbccl.deepep_adaptor import Config
from .zbccl_buffer import Buffer
from .zbccl_utils import EventOverlap

__all__ = [
    "switch_to_allocator",
    "zbccl_get_shmem_base_addr",
    "record_memory_history",
    "dump_snapshot",
    "get_heap_stats",
    "zbccl_init",
    "zbccl_uninit",
    "ZBCCLBootstrapType",
    "zbccl_set_logger_level",
    "simulate_init",
    "__version__"
]

def _new_process_helper(dist_backend_opts, pg_options):
    store = dist_backend_opts.store
    group_rank = dist_backend_opts.group_rank
    group_size = dist_backend_opts.group_size

    if pg_options is None or not isinstance(pg_options, ProcessGroupZBCCL.Options):
        pg_options = ProcessGroupZBCCL.Options()

    pg_options.is_high_priority_stream = False
    pg_options.op_timeout = dist_backend_opts.timeout
    pg_options.global_ranks_in_group = dist_backend_opts.global_ranks_in_group
    pg_options.group_id = dist_backend_opts.group_id
    return ProcessGroupZBCCL(store, group_rank, group_size, pg_options)

torch.distributed.Backend.register_backend("zbccl", lambda dist_backend_opts, pg_options:
    _new_process_helper(dist_backend_opts, pg_options), extended_api=True, devices=["npu"])