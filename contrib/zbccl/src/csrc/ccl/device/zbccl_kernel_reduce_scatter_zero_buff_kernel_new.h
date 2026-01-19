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
#ifndef REDUCE_SCATTER_ZERO_BUFF_KERNEL_H
#define REDUCE_SCATTER_ZERO_BUFF_KERNEL_H

#include <cstdint>
#include "kernel_operator.h"
#include "shmem_api.h"
#include "zbccl_device_common.h"
#include "zbccl_defines.h"
#include "zbccl_functions.h"

namespace zbccl {
namespace reduce_scatter_zero_buff {


constexpr int64_t SYNC_FLAG_INTERVAL = 16;
constexpr uint32_t UB_DMA_MAX_SIZE = 190 * 1024;


template <typename T>
SHMEM_DEVICE void SetAtomicOp(uint32_t atomicOp)
{
    switch ((enum ReduceOp)atomicOp) {
        case ReduceOp::REDUCE_SUM:
            AscendC::SetAtomicAdd<T>();
            break;
        case ReduceOp::REDUCE_MAX:
            AscendC::SetAtomicMax<T>();
            break;
        case ReduceOp::REDUCE_MIN:
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


template <typename T>
class ZeroBuffReduceScatterKernelNew
{
public:
    __aicore__ inline ZeroBuffReduceScatterKernelNew() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR gva, AscendC::TPipe* pipe,
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
        uint32_t lenPerRank = totalLength / rankSize;

        // core_target_rank = [0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3] core_rank_idx = [0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3]
        corePerRank = coreGroupNum / rankSize;
        coreRankIdx = aivIndex % corePerRank;
        coreTargetRank = aivIndex / corePerRank;

        GM_ADDR metaAddr = gva + aivNum * SYNC_FLAG_INTERVAL * sizeof(int32_t);
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

        uint32_t gvaDataOffset = aivNum * SYNC_FLAG_INTERVAL;
        gvaSyncOffset = aivIndex * SYNC_FLAG_INTERVAL;
        xGm.SetGlobalBuffer((__gm__ T *)inputPtr + xOffset, lenPerCore);
        yGm.SetGlobalBuffer((__gm__ T *)y + yOffset, lenPerCore);
        gvaSyncGm.SetGlobalBuffer((__gm__ int32_t *)gva, gvaDataOffset);
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
    AscendC::GlobalTensor<int32_t> gvaSyncGm;
    uint32_t rank;
    uint32_t atomicOp;
    uint32_t lenPerCore;
    uint32_t coreTargetRank;
    uint32_t coreRankIdx;
    uint32_t corePerRank;
    uint32_t gvaSyncOffset;
    uint32_t magic;
};



}
}
#endif // REDUCE_SCATTER_ZERO_BUFF_KERNEL_H