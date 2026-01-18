import os

from enum import Enum


class ZBCCLBootstrapType(Enum):
    MEMFABRIC = 0,
    ACLSHMEM = 1,


def zbccl_init(physicalMemoryFraction: float, bootstrap: ZBCCLBootstrapType):
    '''
    Initialize zbccl library

    :param physicalMemoryFraction: proportion of device memory managed by zbccl
    :param bootstrap: gva memory bootstrap backend
    :return: 0 if
    '''

    # init mem allocator

    # init process group

    return None


def zbccl_uninit():
    pass
