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
    # bootstrap

    # init mem allocator

    # init ccl

    return None


def zbccl_uninit():
    '''
    Un-initialize zbccl library
    :return:
    '''

    # un-init ccl

    # un-init allocator

    # un-init bootstrap

    return None
