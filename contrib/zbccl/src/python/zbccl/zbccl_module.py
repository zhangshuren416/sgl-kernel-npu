import ctypes
from pathlib import Path
import torch
import torch_npu
from enum import Enum
from typing import Optional
from zbccl.zbccl import ZBCCLBootstrapType, ZBCCLBootstrapOption, zbccl_bootstrap, zbccl_unbootstrap

CURRENT_DIR = Path(__file__).resolve().parent
ZBCCL_LIB = list(CURRENT_DIR.glob("zbccl.*.so"))[0]


def zbccl_init(world_size: int,
               rank_id: int,
               device_mem_size: int,
               bootstrap_type: ZBCCLBootstrapType = ZBCCLBootstrapType.BOOT_BY_MEMFABRIC,
               start_config_server: bool = False,
               data_op_type: int = 0,
               ccl_meta_space_size: int = 1,
               ccl_group_cap: int = 128,
               flags: int = 0,
               ip_port : str = "tcp://127.0.0.1:6789"):
    '''
    Initialize zbccl library

    :param physicalMemoryFraction: proportion of device memory managed by zbccl
    :param bootstrap: gva memory bootstrap backend
    :return: 0 if
    '''
    # bootstrap
    opt = ZBCCLBootstrapOption()
    opt.flags = flags
    opt.btType = bootstrap_type
    opt.ipPort = ip_port
    opt.worldSize = world_size
    opt.rankId = rank_id
    opt.startConfigServer = start_config_server
    opt.deviceMemorySize = device_mem_size
    opt.dataOperationType = data_op_type
    opt.cclMetaSpaceSize = ccl_meta_space_size
    opt.cclGroupCap = ccl_group_cap
    zbccl_bootstrap(opt)

    # init mem allocator

    # init ccl

    return True


def zbccl_uninit(flags: int = 0):
    '''
    Un-initialize zbccl library
    :return:
    '''

    # un-init ccl

    # un-init allocator

    # un-init bootstrap
    zbccl_unbootstrap(flags)

    return True


def switch_to_allocator():
    new_alloc = torch_npu.npu.memory.NPUPluggableAllocator(ZBCCL_LIB,
                                                           "zbccl_pluggable_malloc", "zbccl_pluggable_free")
    # Swap the current allocator
    torch_npu.npu.memory.change_current_allocator(new_alloc)
    zbccl_allocator = ctypes.CDLL(ZBCCL_LIB)

    init_fn = ctypes.cast(getattr(zbccl_allocator, "zbccl_pluggable_init"), ctypes.c_void_p).value
    empty_fn = ctypes.cast(getattr(zbccl_allocator, "zbccl_pluggable_empty_cache"), ctypes.c_void_p).value
    record_stream_fn = ctypes.cast(getattr(zbccl_allocator, "zbccl_pluggable_record_stream"), ctypes.c_void_p).value
    erase_stream_fn = ctypes.cast(getattr(zbccl_allocator, "zbccl_pluggable_erase_stream"), ctypes.c_void_p).value
    # begin_allocate_to_pool_fn = ctypes.cast(getattr(zbccl_allocator, "zbccl_pluggable_begin_allocate_to_pool"), ctypes.c_void_p).value
    # end_allocate_to_pool_fn = ctypes.cast(getattr(zbccl_allocator, "zbccl_pluggable_end_allocate_to_pool"), ctypes.c_void_p).value
    # release_pool_fn = ctypes.cast(getattr(zbccl_allocator, "zbccl_pluggable_release_pool"), ctypes.c_void_p).value

    new_alloc.allocator().set_init_fn(init_fn)
    new_alloc.allocator().set_reset_fn(empty_fn)
    new_alloc.allocator().set_record_stream_fn(record_stream_fn)
    new_alloc.allocator().set_erase_stream_fn(erase_stream_fn)
    # new_alloc.allocator().set_begin_allocate_to_pool_fn(begin_allocate_to_pool_fn)
    # new_alloc.allocator().set_end_allocate_to_pool_fn(end_allocate_to_pool_fn)
    # new_alloc.allocator().set_release_pool_fn(release_pool_fn)


def init_shmem(my_rank, n_ranks, local_mem_size, meta_size, ip_port, is_simulation=False):
    zbccl_allocator = ctypes.CDLL(ZBCCL_LIB)
    # 设置函数原型
    zbccl_allocator.zbccl_inner_init_shmem.argtypes = [
        ctypes.c_int,      # my_rank
        ctypes.c_int,      # n_ranks
        ctypes.c_uint64,   # local_mem_size
        ctypes.c_uint64,   # meta_size
        ctypes.c_char_p,   # ip_port
        ctypes.c_bool      # is_simulation
    ]
    zbccl_allocator.zbccl_inner_init_shmem.restype = None

    zbccl_allocator.zbccl_inner_init_shmem(
        ctypes.c_int(my_rank),                    # my_rank
        ctypes.c_int(n_ranks),                    # n_ranks
        ctypes.c_uint64(local_mem_size),          # local_mem_size
        ctypes.c_uint64(meta_size),               # meta_size
        ip_port.encode('utf-8'),                  # ip_port
        ctypes.c_bool(is_simulation)              # is_simulation
    )


def zbccl_get_shmem_base_addr():
    zbccl_allocator = ctypes.CDLL(ZBCCL_LIB)
    zbccl_allocator.zbccl_get_shmem_base_addr.restype = ctypes.c_void_p
    return zbccl_allocator.zbccl_get_shmem_base_addr()