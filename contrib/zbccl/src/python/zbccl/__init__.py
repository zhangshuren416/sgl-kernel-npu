
from .zbccl_module import zbccl_init, zbccl_uninit, ZBCCLBootstrapType
from .zbccl_module import switch_to_allocator, init_shmem

from zbccl.zbccl.allocator import record_memory_history, dump_snapshot
from zbccl.zbccl import npu_adaptor

__all__ = [
    "switch_to_allocator",
    "init_shmem",
    "record_memory_history",
    "dump_snapshot"
]

zbccl_init(1, ZBCCLBootstrapType.MEMFABRIC)