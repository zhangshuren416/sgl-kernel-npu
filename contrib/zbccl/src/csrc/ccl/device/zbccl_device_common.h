/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * ZBCCL is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */
#ifndef ZBCCL_DEVICE_COMMON_H
#define ZBCCL_DEVICE_COMMON_H

#include <cstdint>
#include "kernel_operator.h"
#include "shmem_api.h"

#define AICORE_FORCE_INLINE __attribute__((always_inline)) __aicore__ __inline__

namespace zbccl {

constexpr uint32_t DATA_ADDR_SIZE = 8;
constexpr uint32_t DATA_ADDR_INTERVAL = 8;
constexpr uint32_t FLAG_INTERVAL = 16;

AICORE_FORCE_INLINE void dcciCacheline(__gm__ uint8_t *addr)
{
    using namespace AscendC;
    GlobalTensor<uint8_t> global;
    global.SetGlobalBuffer(addr);

    // Important: add hint to avoid dcci being optimized by compiler
    __asm__ __volatile__("");
    DataCacheCleanAndInvalid<uint8_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(global);
    __asm__ __volatile__("");
}

AICORE_FORCE_INLINE uint64_t GetDataAddr(__gm__ void *metaAddr, uint32_t rank, uint32_t groupSize)
{
    uint32_t dataAddrOffset = rank * DATA_ADDR_INTERVAL;
    __gm__ uint64_t* dataGmAddr = (__gm__ uint64_t*)metaAddr + dataAddrOffset;
    dcciCacheline((__gm__ uint8_t *)dataGmAddr);
    return *dataGmAddr;
}

AICORE_FORCE_INLINE void SetDataAddr(__gm__ void *metaAddr, uint64_t val, uint32_t rank, uint32_t groupSize)
{
    uint32_t dataAddrOffset = rank * DATA_ADDR_INTERVAL;
    __gm__ uint64_t* dataGmAddr = (__gm__ uint64_t*)metaAddr + dataAddrOffset;
    *dataGmAddr = val;
    dcciCacheline((__gm__ uint8_t *)dataGmAddr);
}

AICORE_FORCE_INLINE int32_t GetFlag(__gm__ void *metaAddr, uint32_t rank, uint32_t groupSize)
{
    uint32_t dataAddrLength = groupSize * DATA_ADDR_SIZE * DATA_ADDR_INTERVAL;
    __gm__ int32_t* flagAddr = (__gm__ int32_t*)((__gm__ uint8_t*)metaAddr + dataAddrLength) + rank * FLAG_INTERVAL;
    dcciCacheline((__gm__ uint8_t *)flagAddr);
    return *flagAddr;
}

AICORE_FORCE_INLINE void SetFlag(__gm__ void *metaAddr, int32_t val, uint32_t rank, uint32_t groupSize)
{
    uint32_t dataAddrLength = groupSize * DATA_ADDR_SIZE * DATA_ADDR_INTERVAL;
    __gm__ int32_t* flagAddr = (__gm__ int32_t*)((__gm__ uint8_t*)metaAddr + dataAddrLength) + rank * FLAG_INTERVAL;
    *flagAddr = val;
    dcciCacheline((__gm__ uint8_t *)flagAddr);
}

AICORE_FORCE_INLINE void InitDataAddrAndFlag(__gm__ void *metaAddr, __gm__ void* inputAddr, uint32_t aivIndex, uint32_t rank, uint32_t groupSize)
{
    if (aivIndex < groupSize) {
        SetFlag(metaAddr, 0, aivIndex, groupSize);
    }
    shmem_barrier_all();
    if (aivIndex < groupSize) {
        uint64_t dataAddr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(inputAddr));
        SetDataAddr(shmem_ptr(metaAddr, aivIndex), dataAddr, rank, groupSize);
        SetFlag(shmem_ptr(metaAddr, aivIndex), 1, rank, groupSize);
    }
}

} // namespace zbccl 


#endif