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
#include "zbccl_defines.h"
#include "zbccl_functions.h"

namespace zbccl {
namespace reducescatter {

constexpr int64_t SYNC_FLAG_INTERVAL = 16;
constexpr uint32_t UB_DMA_MAX_SIZE = 190 * 1024;
constexpr int64_t GVA_BUFF_MAX_SIZE = 100 * 1024 * 1024;
constexpr uint32_t BIG_DATA_SIZE = 2 * 1024 * 1024;

template <typename T>
__aicore__ inline void SetAtomicOp(uint32_t atomicOp)
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
class ReduceScatterKernel
{
public:
    __aicore__ inline ReduceScatterKernel() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR gva,
                                uint32_t rank, uint32_t rankSize, uint32_t totalLength, uint32_t curLength,
                                uint32_t magic, uint64_t fftsAddr, bool isSmall, uint32_t atomicOp = 0)
    {
        this->rank = rank;
        this->rankSize = rankSize;
        this->magic = magic;
        this->atomicOp = atomicOp;
        this->totalLength = totalLength;
        this->fftsAddr = fftsAddr;
        this->smallFlag = isSmall;

        const uint32_t aivNum = AscendC::GetBlockNum();
        this->aivIndex = AscendC::GetBlockIdx();

        if (this->smallFlag) {
            this->coreGroupNum = aivNum;
            this->elePerRank = totalLength / rankSize;
            this->lenPerRank = totalLength / rankSize;
        } else {
            this->coreGroupNum = aivNum / 2;
            this->elePerRank = totalLength / rankSize;
            this->lenPerRank = curLength / rankSize;
        }

        // core_target_rank = [0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3] core_rank_idx = [0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3]
        this->corePerRank = this->coreGroupNum / rankSize;
        this->coreTargetRank = this->aivIndex / this->corePerRank;
        this->coreRankIdx = this->aivIndex % this->corePerRank;

        // len: length of single process loop
        uint32_t lenPerRankAlignToCore = CeilDiv(this->lenPerRank, this->corePerRank) * this->corePerRank;
        formerLength = lenPerRankAlignToCore / this->corePerRank;
        tailLength = lenPerRank / this->corePerRank;
        formerNum = this->lenPerRank % this->corePerRank;
        tailNum = this->corePerRank - formerNum;

        if (this->coreRankIdx < formerNum) {
            this->lenPerCore = formerLength;
            xOffset = coreTargetRank * elePerRank + this->coreRankIdx * formerLength;
            yOffset = this->coreRankIdx * formerLength;
        } else {
            this->lenPerCore = tailLength;
            xOffset = coreTargetRank * elePerRank + formerNum * formerLength +
                        (this->coreRankIdx - formerNum) * tailLength;
            yOffset = formerNum * formerLength + (this->coreRankIdx - formerNum) * tailLength;
        }

        gvaDataOffset = aivNum * SYNC_FLAG_INTERVAL;
        gvaSyncOffset = this->aivIndex * SYNC_FLAG_INTERVAL;
        xGm.SetGlobalBuffer((__gm__ T *)x + xOffset, this->lenPerCore);
        yGm.SetGlobalBuffer((__gm__ T *)y + yOffset, this->lenPerCore);
        gvaGm.SetGlobalBuffer((__gm__ T *)((__gm__ int32_t *)gva + gvaDataOffset), GVA_BUFF_MAX_SIZE / sizeof(T));
        gvaSyncGm.SetGlobalBuffer((__gm__ int32_t *)gva, gvaDataOffset);
    }

    __aicore__ inline void Process()
    {
        shmemx_set_ffts_config(fftsAddr);
        if (smallFlag) {
            CopySmallData();
        } else {
            CopyBigData();
        }
    }

    __aicore__ inline void RunBigDataOp(GM_ADDR input, GM_ADDR output, GM_ADDR gva,
        uint32_t rank, uint32_t rankSize, uint32_t totalLength, uint32_t magic, uint64_t fftsAddr, uint32_t reduceOp)
    {
        const int64_t maxGvaNum = GVA_BUFF_MAX_SIZE / sizeof(T);
        uint32_t maxCountPerLoop = (uint32_t)maxGvaNum;
        uint32_t times = (totalLength + maxCountPerLoop - 1) / maxCountPerLoop;
        uint32_t leftNum = totalLength;
        uint32_t curInputOffset = 0;
        uint32_t curOutputOffset = 0;
        for (uint32_t i = 0; i < times; i++) {
            uint32_t curLen = leftNum > maxCountPerLoop ? maxCountPerLoop : leftNum;
            uint32_t curOffset = curLen / rankSize;
            Init(input + curInputOffset * sizeof(T), output + curOutputOffset * sizeof(T), gva, rank, rankSize,
                    totalLength, curLen, (magic + i) * 1024, fftsAddr, false, reduceOp);
            AscendC::PipeBarrier<PIPE_ALL>();
            shmemx_barrier_all_vec();
            Process();
            leftNum -= curLen;
            curInputOffset += curOffset;
            curOutputOffset += curOffset;
            AscendC::PipeBarrier<PIPE_ALL>();
            shmemx_barrier_all_vec();
        }
    }

private:
    __aicore__ inline void CopySmallData()
    {
#ifdef __DAV_C220_VEC__
        const uint32_t ubSize = UB_DMA_MAX_SIZE;
        uint32_t gvaCopyInOffset;
        uint32_t gvaCopyOutOffset;

        const __gm__ int32_t *gvaSyncGmAddr = gvaSyncGm.GetPhyAddr();
        __gm__ int32_t *coreGvaSyncGmAddr = (__gm__ int32_t *)gvaSyncGmAddr + gvaSyncOffset;

        AscendC::LocalTensor<T> tmpBuff(AscendC::TPosition::VECIN, 64, ubSize / sizeof(T));

        // data move parameters
        if (coreRankIdx < formerNum) {
            gvaCopyInOffset = coreTargetRank * elePerRank + coreRankIdx * formerLength;
            gvaCopyOutOffset = rank * elePerRank + coreRankIdx * formerLength;
        } else {
            gvaCopyInOffset =
                coreTargetRank * elePerRank + formerNum * formerLength + (coreRankIdx - formerNum) * tailLength;
            gvaCopyOutOffset = rank * elePerRank + formerNum * formerLength + (coreRankIdx - formerNum) * tailLength;
        }

        // [ReduceScatter Step 1] local input gm -> symmetric mem.
        shmem_mte_put_mem_nbi(gvaGm[gvaCopyInOffset], xGm, tmpBuff, lenPerCore, rank, EVENT_ID0);

        // Sync Ensure Corresponding Tasks Done.
        shmem_quiet();
        shmemi_barrier_core_soft();

        shmemx_signal_op(coreGvaSyncGmAddr, magic, SHMEM_SIGNAL_SET, rank);
        __gm__ int32_t * waitAddr = (__gm__ int32_t *)shmem_ptr((__gm__ int32_t *)gvaSyncGmAddr, coreTargetRank);
        int32_t waitOffset = (rank * corePerRank + coreRankIdx) * SYNC_FLAG_INTERVAL;
        shmem_signal_wait_until(waitAddr + waitOffset, SHMEM_CMP_EQ, magic);

        // [ReduceScatter Step 2] symmetric mem -> local output & reduce.
        
        SetAtomicOp<T>(atomicOp);
        AscendC::PipeBarrier<PIPE_ALL>();

        shmem_mte_get_mem_nbi(yGm, gvaGm[gvaCopyOutOffset], tmpBuff, lenPerCore, coreTargetRank, EVENT_ID0);

        AscendC::SetAtomicNone();
        AscendC::PipeBarrier<PIPE_ALL>();
#endif
    }

    __aicore__ inline void CopyBigData()
    {
#ifdef __DAV_C220_VEC__

        const uint32_t ubSize = UB_DMA_MAX_SIZE;
        uint32_t gvaCopyInOffset;
        uint32_t gvaCopyOutOffset;

        const __gm__ int32_t *gvaSyncGmAddr = gvaSyncGm.GetPhyAddr();
        __gm__ int32_t *coreGvaSyncGmAddr = (__gm__ int32_t *)gvaSyncGmAddr + gvaSyncOffset;

        if (coreRankIdx < formerNum) {
            gvaCopyInOffset = coreTargetRank * lenPerRank + coreRankIdx * formerLength;
            gvaCopyOutOffset = rank * lenPerRank + coreRankIdx * formerLength;
        } else {
            gvaCopyInOffset =
                coreTargetRank * lenPerRank + formerNum * formerLength + (coreRankIdx - formerNum) * tailLength;
            gvaCopyOutOffset = rank * lenPerRank + formerNum * formerLength + (coreRankIdx - formerNum) * tailLength;
        }

        // 0-half core copy data to local symmetric mem, half-all core copy remote data from symmetric mem.
        // GM to SymmPtr
        // TODO copy local rank data to output addr directly

        if (aivIndex < coreGroupNum) {
            AscendC::LocalTensor<T> tmpBuff(AscendC::TPosition::VECIN, (1024 + 32), ubSize / sizeof(T));
            uint32_t copyNum = ubSize / sizeof(T);
            uint32_t leftCopySize = lenPerCore * sizeof(T);

            int64_t times = 0;
            int64_t flag = 0;
            // todo ub align
            while (leftCopySize >= ubSize) {
                shmem_mte_put_mem_nbi(gvaGm[gvaCopyInOffset + times * copyNum],
                                      xGm[times * copyNum],
                                      tmpBuff, copyNum, rank, EVENT_ID0);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
                times += 1;
                flag = times + magic;
                shmemx_signal_op(coreGvaSyncGmAddr, flag, SHMEM_SIGNAL_SET, rank);

                AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID0);

                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);

                leftCopySize -= ubSize;
            }
            if (leftCopySize <= 0) {
                return;
            }
            shmem_mte_put_mem_nbi(gvaGm[gvaCopyInOffset + times * copyNum],
                                  xGm[times * copyNum],
                                  tmpBuff, leftCopySize / sizeof(T), rank, EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
            times += 1;
            flag = times + magic;
            shmemx_signal_op(coreGvaSyncGmAddr, flag, SHMEM_SIGNAL_SET, rank);
            return;
        }

        CpGvaToOutput(gvaCopyOutOffset);
#endif
    }

    __aicore__ inline void CpGvaToOutput(uint32_t gvaCopyOutOffset)
    {
        coreRankIdx = (aivIndex - coreGroupNum) % corePerRank;
        coreTargetRank = (aivIndex - coreGroupNum) / corePerRank;

        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
        uint32_t copiedCount = 0;
        AscendC::LocalTensor<int32_t> ctrlFlag(AscendC::TPosition::VECIN, 32, 512);
        uint32_t totalSize = lenPerCore * sizeof(T);
        gvaSyncOffset = (rank * corePerRank + coreRankIdx) * SYNC_FLAG_INTERVAL;
        while (true) {
            shmem_get_int32_mem_nbi(ctrlFlag, gvaSyncGm[gvaSyncOffset], 1, coreTargetRank);
            AscendC::PipeBarrier<PIPE_ALL>();

            if ((ctrlFlag.GetValue(0) >> 10) != (magic >> 10)) {
                continue;
            }

            int32_t readyNum = ctrlFlag.GetValue(0) - magic;
            if (readyNum <= 0 || copiedCount >= readyNum) {
                continue;
            }

            uint32_t gvaSendOffset = copiedCount * UB_DMA_MAX_SIZE / sizeof(T);
            uint32_t outputRecvOffset = copiedCount * UB_DMA_MAX_SIZE / sizeof(T);
            uint32_t curLoopNumLeft = (readyNum - copiedCount) * UB_DMA_MAX_SIZE / sizeof(T);
            if (readyNum * UB_DMA_MAX_SIZE > totalSize) {
                curLoopNumLeft = (totalSize - copiedCount * UB_DMA_MAX_SIZE) / sizeof(T);
            }

            SetAtomicOp<T>(atomicOp);
            AscendC::PipeBarrier<PIPE_ALL>();
            CpGM2GM(gvaCopyOutOffset, gvaSendOffset, outputRecvOffset, curLoopNumLeft);
            AscendC::SetAtomicNone();
            AscendC::PipeBarrier<PIPE_ALL>();
            copiedCount = readyNum;
            if (copiedCount * UB_DMA_MAX_SIZE >= totalSize) {
                break;
            }
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    }

    __aicore__ inline void CpGM2GM(uint32_t gvaOutOffset, uint32_t sendOffset, uint32_t outputOffset, uint32_t count)
    {
        uint32_t copyUbSize = UB_DMA_MAX_SIZE / 2;
        uint32_t copyUbNum = copyUbSize / sizeof(T);
        AscendC::LocalTensor<T> pingBuff(AscendC::TPosition::VECIN, 1024 + 32, copyUbNum);
        AscendC::LocalTensor<T> pongBuff(AscendC::TPosition::VECIN, 96 * 1024 + 32, copyUbNum);
        AscendC::LocalTensor<T> ubBuff;

        int pingpongId = 0;
        for (uint32_t i = 0; count > 0; i++) {
            AscendC::TEventID EVENT_ID = pingpongId == 0 ? EVENT_ID0 : EVENT_ID1;
            ubBuff = pingpongId == 0 ? pingBuff : pongBuff;
            uint32_t copyNum = count > copyUbNum ? copyUbNum : count;
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
            // todo ub align pad

            shmem_mte_get_mem_nbi(yGm[outputOffset], gvaGm[gvaOutOffset + sendOffset],
                                    ubBuff, copyNum, coreTargetRank, EVENT_ID);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);

            sendOffset += copyNum;
            outputOffset += copyNum;
            count -= copyNum;
            pingpongId = 1 - pingpongId;
        }
    }

private:
    AscendC::GlobalTensor<T> xGm;
    AscendC::GlobalTensor<T> yGm;
    AscendC::GlobalTensor<T> gvaGm;
    AscendC::GlobalTensor<int32_t> gvaSyncGm;
    uint32_t rank;
    uint32_t rankSize;
    uint32_t totalLength;
    uint32_t magic;
    uint64_t fftsAddr;
    uint32_t atomicOp;
    uint32_t coreGroupNum;
    uint32_t corePerRank;
    uint32_t lenPerCore;
    uint32_t lenPerRank;
    uint32_t coreTargetRank;
    uint32_t coreRankIdx;
    uint32_t elePerRank;
    uint32_t xOffset;
    uint32_t yOffset;
    uint32_t gvaSyncOffset;
    uint32_t gvaDataOffset;
    uint32_t formerNum;
    uint32_t tailNum;
    uint32_t formerLength;
    uint32_t tailLength;
    int64_t aivIndex;
    bool smallFlag;
};
}
}
#endif  // ZBCCL_KERNEL_REDUCE_SCATTER_H
