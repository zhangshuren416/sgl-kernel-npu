import ctypes
from pathlib import Path
import os
import torch
import torch_npu
from enum import Enum
from typing import Optional
from zbccl.zbccl import ZBCCLBootstrapType, ZBCCLBootstrapOption
from zbccl.zbccl import zbccl_version, zbccl_bootstrap, zbccl_unbootstrap, zbccl_comm_destroy_all
from zbccl.zbccl.allocator import get_heap_stats

CURRENT_DIR = Path(__file__).resolve().parent
ZBCCL_LIB = list(CURRENT_DIR.glob("zbccl.*.so"))[0]

__version__ = zbccl_version()

def zbccl_init(world_size: int,
               rank_id: int,
               device_mem_size: int,
               bootstrap_type: ZBCCLBootstrapType = ZBCCLBootstrapType.BOOT_BY_MEMFABRIC,
               start_config_server: bool = False,
               data_op_type: int = 0,
               ccl_meta_space_size: int = 1024,
               ccl_group_cap: int = 128,
               flags: int = 0,
               ip_port : str = "tcp://127.0.0.1:6789"):
    '''
    Initialize zbccl library

    :param world_size: size of ranks to init zbccl
    :param rank_id: current rank id
    :param bootstrap_type: under memory bootstrap type, memfabric support only
    :param start_config_server: whether to start config server
    :param data_op_type: data operator type
    :param ccl_meta_space_size: collective communication meta space size, unit is KB
    :param ccl_group_cap: number of collective communication
    :param flag: reserve flag
    :param ip_port: bootstrap used ip port
    :return: 0 if success else error code
    '''

    # get env of MemFabric home
    mem_fabric_lib_path = os.environ.get("MEMFABRIC_HYBRID_LIBRARY_PATH")
    if mem_fabric_lib_path is None:
        # try to import memfabric from python package
        import memfabric_hybrid as mf
        mem_fabric_lib_path = mf.get_lib_path()
        if mem_fabric_lib_path is not None:
            os.environ["MEMFABRIC_HYBRID_LIBRARY_PATH"] = mem_fabric_lib_path
            print(f"Set MEMFABRIC_HYBRID_LIBRARY_PATH to {mem_fabric_lib_path}")


    # init mem allocator, switch before set_device
    switch_to_allocator()
    torch.npu.set_device(rank_id)

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

    # init ccl

    return True


def zbccl_uninit(flags: int = 0):
    '''
    Un-initialize zbccl library
    :return:
    '''

    # un-init ccl
    zbccl_comm_destroy_all(flags)

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


def zbccl_get_shmem_base_addr():
    print("zbccl_get_shmem_base_addr is deprecated, using zbccl_init instead")
    zbccl_allocator = ctypes.CDLL(ZBCCL_LIB)
    zbccl_allocator.zbccl_get_shmem_base_addr.restype = ctypes.c_void_p
    return zbccl_allocator.zbccl_get_shmem_base_addr()


def mem_get_info():
    # this api is used to fulfill torch.npu.mem_get_info functions when sma take control of memory
    return get_heap_stats()[1] - get_heap_stats()[0], get_heap_stats()[1]
