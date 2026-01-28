/*
Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
ZBCCL is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:
     http://license.coscl.org.cn/MulanPSL2

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
*/
#ifndef ZBCCL_KERNEL_BARRIER_H
#define ZBCCL_KERNEL_BARRIER_H

#include "kernel_operator.h"
#include "zbccl_kernel_def.h"
#include "zbccl_kernel_utils.h"


template<typename T>
ZBCCL_KERNEL T zbccl_load(__gm__ T *addr)
{
    return *((__gm__ T *)addr);
}

template<typename T>
ZBCCL_KERNEL void zbccl_store(__gm__ T *addr, T value)
{
    *((__gm__ T *)addr) = value;
}

ZBCCL_KERNEL void zbccl_single_set(__gm__ uint64_t *addr, uint64_t val)
{
    zbccl_store(addr, val);
    dcciCacheline((__gm__ uint8_t *) addr);
}

ZBCCL_KERNEL void zbccl_single_wait_until_eq(__gm__ uint64_t *syncAddr, uint64_t cmp_val)
{
    uint64_t cur_val;
    do {
        dcciCacheline((__gm__ uint8_t *)syncAddr);
        cur_val = *syncAddr;
    } while(!(cur_val == cmp_val || cur_val == (cmp_val + 1)));
}

ZBCCL_KERNEL void zbccl_barrier_npu(uint16_t rankId, uint16_t groupSize, uint64_t localSize, __gm__ uint64_t *counterAddress, GM_ADDR output)
{
    int vecId = AscendC::GetBlockIdx();
    int vecSize = AscendC::GetBlockNum() * AscendC::GetTaskRation();

    int k = 8;
    k = k < groupSize ? k : groupSize;
    k = k < vecSize ? k : vecSize;

    __gm__ uint64_t *sync_counter = reinterpret_cast<__gm__ uint64_t *>(counterAddress);

    uint64_t count = zbccl_load(sync_counter) + 1;
    if (vecId == rankId % vecSize) {
        zbccl_single_set(sync_counter, count);
    }

    for (int i = vecId; i < groupSize; i += k) {
        __gm__ uint64_t *target_addr = (__gm__ uint64_t *)zbccl_ptr((__gm__ void *)counterAddress, rankId, i, localSize);
        zbccl_single_wait_until_eq(target_addr, count);
    }

    zbccl_store(sync_counter, count);
}

ZBCCL_KERNEL void zbccl_barrier_all(uint16_t rankId, uint16_t groupSize, uint64_t localSize, __gm__ uint64_t *counterAddress, GM_ADDR output)
{
    // AscendC::SyncAll<false>();

    if ASCEND_IS_AIV {
        zbccl_barrier_npu(rankId, groupSize, localSize, counterAddress, output);
    }

    // AscendC::SyncAll<false>();
}

#endif