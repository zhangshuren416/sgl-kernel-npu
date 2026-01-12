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
#ifndef ALL_REDUCE_ZERO_BUFF_KERNEL_H
#define ALL_REDUCE_ZERO_BUFF_KERNEL_H

#include <cstdint>
#include "kernel_operator.h"
#include "shmem_api.h"
#include "zbccl_defines.h"
#include "zbccl_functions.h"

namespace zbccl {
namespace allreduce {


constexpr int64_t SYNC_FLAG_INTERVAL = 16;
constexpr uint32_t UB_DMA_MAX_SIZE = 192 * 1024;


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
class ZeroBuffAllReduceKernel
{
public:
    __aicore__ inline ZeroBuffAllReduceKernel() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR gva,
                                uint32_t rank, uint32_t rankSize, uint32_t totalLength,
                                uint32_t magic, uint64_t fftsAddr, uint32_t atomicOp = 0)
    {
        shmemx_set_ffts_config(fftsAddr);
        this->atomicOp = atomicOp;
        this->magic = magic;
        this->rank = rank;

        const uint32_t aivNum = AscendC::GetBlockNum();
        const uint32_t aivIndex = AscendC::GetBlockIdx();

        uint32_t coreGroupNum = aivNum;
        uint32_t lenPerRank = totalLength;

        // core_target_rank = [0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3] core_rank_idx = [0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3]
        corePerRank = coreGroupNum / rankSize;
        coreRankIdx = aivIndex % corePerRank;
        coreTargetRank = aivIndex / corePerRank;

        uint32_t lenPerRankAlignToCore = CeilDiv(lenPerRank, corePerRank) * corePerRank;
        uint32_t formerLength = lenPerRankAlignToCore / corePerRank;
        uint32_t tailLength = lenPerRank / corePerRank;
        uint32_t formerNum = lenPerRank % corePerRank;
        uint32_t tailNum = corePerRank - formerNum;
        uint32_t xOffset;
        uint32_t yOffset;

        if (coreRankIdx < formerNum) {
            lenPerCore = formerLength;
            xOffset = coreRankIdx * formerLength;
            yOffset = coreRankIdx * formerLength;
        } else {
            lenPerCore = tailLength;
            xOffset = formerNum * formerLength + (coreRankIdx - formerNum) * tailLength;
            yOffset = formerNum * formerLength + (coreRankIdx - formerNum) * tailLength;
        }

        uint32_t gvaDataOffset = aivNum * SYNC_FLAG_INTERVAL;
        gvaSyncOffset = aivIndex * SYNC_FLAG_INTERVAL;
        xGm.SetGlobalBuffer((__gm__ T *)x + xOffset, lenPerCore);
        yGm.SetGlobalBuffer((__gm__ T *)y + yOffset, lenPerCore);
        gvaSyncGm.SetGlobalBuffer((__gm__ int32_t *)gva, gvaDataOffset);
    }

    __aicore__ inline void Process()
    {
#ifdef __DAV_C220_VEC__
        // AsdopsBuffer<ArchType::ASCEND_V220> buf;
        // AscendC::LocalTensor<T> tmpBuff = buf.GetBuffer<BufferType::ASCEND_UB, T>(64);
        AscendC::LocalTensor<T> tmpBuff(AscendC::TPosition::VECIN, 64, UB_DMA_MAX_SIZE);

        const __gm__ int32_t *gvaSyncGmAddr = gvaSyncGm.GetPhyAddr();
        __gm__ int32_t *coreGvaSyncGmAddr = (__gm__ int32_t *)gvaSyncGmAddr + gvaSyncOffset;
        shmemx_signal_op(coreGvaSyncGmAddr, magic, SHMEM_SIGNAL_SET, rank);
        __gm__ int32_t * waitAddr = (__gm__ int32_t *)shmem_ptr((__gm__ int32_t *)gvaSyncGmAddr, coreTargetRank);
        int32_t waitOffset = (rank * corePerRank + coreRankIdx) * SYNC_FLAG_INTERVAL;
        shmem_signal_wait_until(waitAddr + waitOffset, SHMEM_CMP_EQ, magic);

        // [AllReduce Step 1] local input gm -> output gm.
        SetAtomicOp<T>(atomicOp);
        AscendC::PipeBarrier<PIPE_ALL>();

        shmem_mte_get_mem_nbi(yGm, xGm, tmpBuff, lenPerCore, coreTargetRank, EVENT_ID0);

        AscendC::SetAtomicNone();
        AscendC::PipeBarrier<PIPE_ALL>();
        // Sync Ensure Corresponding Tasks Done.
        shmem_quiet();
        shmemi_barrier_core_soft();
#endif
    }

private:
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
#endif // ALL_REDUCE_ZERO_BUFF_KERNEL_H