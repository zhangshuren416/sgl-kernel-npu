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
#ifndef ZBCCL_KERNEL_REDUCE_SCATTER_H
#define ZBCCL_KERNEL_REDUCE_SCATTER_H

#include <cstdint>
#include "kernel_operator.h"
#include "shmem_api.h"
#include "zbccl_def.h"

#define AICORE_FORCE_INLINE __attribute__((always_inline)) __aicore__ __inline__

namespace zbccl {
namespace ccl {

constexpr uint32_t UB_DMA_MAX_SIZE = 190 * 1024;
constexpr uint32_t DATA_ADDR_SIZE = 8;
constexpr uint32_t DATA_ADDR_INTERVAL = 8;
constexpr uint32_t FLAG_INTERVAL = 16;

template <typename T>
AICORE_FORCE_INLINE void SetAtomicOp(uint32_t atomicOp)
{
    switch (atomicOp) {
        case 0:
            AscendC::SetAtomicAdd<T>();
            break;
        case 2:
            AscendC::SetAtomicMax<T>();
            break;
        case 3:
            AscendC::SetAtomicMin<T>();
            break;
        default:
            AscendC::SetAtomicNone();
            break;
    }
}

template <typename T>
inline __aicore__ T CeilDiv(const T dividend, const T divisor)
{
    return (divisor == 0) ? 0 : ((dividend + divisor - 1) / divisor);
}

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

template <typename T>
class ZeroBuffReduceScatterKernel
{
public:
    __aicore__ inline ZeroBuffReduceScatterKernel() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR metaAddr, AscendC::TPipe* pipe,
                                uint32_t rank, uint32_t rankSize, uint32_t totalLength,
                                uint32_t magic, uint64_t fftsAddr, uint32_t atomicOp)
    {
        shmemx_set_ffts_config(fftsAddr);
        this->atomicOp = atomicOp;
        this->magic = magic;
        this->rank = rank;

        const uint32_t aivNum = AscendC::GetBlockNum();
        const uint32_t aivIndex = AscendC::GetBlockIdx();

        uint32_t coreGroupNum = aivNum;
        uint32_t lenPerRank = totalLength;

        corePerRank = coreGroupNum / rankSize;
        coreRankIdx = aivIndex % corePerRank;
        coreTargetRank = aivIndex / corePerRank;

        InitDataAddrAndFlag((__gm__ void*)metaAddr, (__gm__ void*)x, aivIndex, rank, rankSize);
        int32_t addrReadyFlag;
        do {
            addrReadyFlag = GetFlag((__gm__ void*)metaAddr, coreTargetRank, rankSize);
        } while (addrReadyFlag != 1);

        uint64_t inputAddr = GetDataAddr((__gm__ void*)metaAddr, coreTargetRank, rankSize);
        GM_ADDR inputPtr = (GM_ADDR)inputAddr;

        uint32_t lenPerRankAlignToCore = CeilDiv(lenPerRank, corePerRank) * corePerRank;
        uint32_t formerLength = lenPerRankAlignToCore / corePerRank;
        uint32_t tailLength = lenPerRank / corePerRank;
        uint32_t formerNum = lenPerRank % corePerRank;
        uint32_t tailNum = corePerRank - formerNum;
        uint32_t xOffset;
        uint32_t yOffset;

        if (coreRankIdx < formerNum) {
            lenPerCore = formerLength;
            xOffset = rank * lenPerRank + coreRankIdx * formerLength;
            yOffset = coreRankIdx * formerLength;
        } else {
            lenPerCore = tailLength;
            xOffset = rank * lenPerRank +
                formerNum * formerLength + (coreRankIdx - formerNum) * tailLength;
            yOffset = formerNum * formerLength + (coreRankIdx - formerNum) * tailLength;
        }

        xGm.SetGlobalBuffer((__gm__ T *)inputPtr + xOffset, lenPerCore);
        yGm.SetGlobalBuffer((__gm__ T *)y + yOffset, lenPerCore);
        if (lenPerCore * sizeof(T) > UB_DMA_MAX_SIZE) {
            pipe->InitBuffer(bindQueue, 1, UB_DMA_MAX_SIZE);
        } else {
            pipe->InitBuffer(bindQueue, 1, lenPerCore * sizeof(T));
        }
    }

    __aicore__ inline void Process()
    {
#ifdef __DAV_C220_VEC__

        uint32_t leftCopySize = lenPerCore * sizeof(T);
        AscendC::DataCopyPadExtParams<T> padParams;
        SetAtomicOp<T>(atomicOp);
        uint32_t times = 0;
        uint32_t preCopyNum = UB_DMA_MAX_SIZE / sizeof(T);

        do {
            uint32_t curCopySize = (leftCopySize > UB_DMA_MAX_SIZE) ? UB_DMA_MAX_SIZE : leftCopySize;
            AscendC::LocalTensor<T> xLocal = bindQueue.AllocTensor<T>();
            AscendC::DataCopyExtParams dataCopyParams(1, curCopySize, 0, 0, 0);
            AscendC::DataCopyPad(xLocal, xGm[times * preCopyNum], dataCopyParams, padParams);
            bindQueue.EnQue(xLocal);
            xLocal = bindQueue.DeQue<T>();
            AscendC::DataCopyPad(yGm[times * preCopyNum], xLocal, dataCopyParams);
            bindQueue.FreeTensor(xLocal);
            leftCopySize = (leftCopySize > UB_DMA_MAX_SIZE) ? leftCopySize - UB_DMA_MAX_SIZE : 0;
            times++;
        } while (leftCopySize > 0);

        AscendC::SetAtomicNone();
        // Sync Ensure Corresponding Tasks Done.
        shmem_quiet();
        shmem_barrier_all();
#endif
    }

private:
    AscendC::TQueBind<AscendC::TPosition::VECIN, AscendC::TPosition::VECIN,1> bindQueue;
    AscendC::GlobalTensor<T> xGm;
    AscendC::GlobalTensor<T> yGm;
    uint32_t rank;
    uint32_t atomicOp;
    uint32_t lenPerCore;
    uint32_t coreTargetRank;
    uint32_t coreRankIdx;
    uint32_t corePerRank;
    uint32_t magic;
};

}
}
#endif // ZBCCL_KERNEL_REDUCE_SCATTER_H