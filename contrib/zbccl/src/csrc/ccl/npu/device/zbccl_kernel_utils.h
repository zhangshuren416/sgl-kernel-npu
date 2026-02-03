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
#ifndef ZBCCL_KERNEL_UTILS_H
#define ZBCCL_KERNEL_UTILS_H

#include "kernel_operator.h"
#include "zbccl_comm_host_device_struct.h"

#define ZBCCL_KERNEL __attribute__((always_inline)) __aicore__ __inline__

constexpr int64_t FLAG_SIZE = 8;
constexpr int64_t UB_DMA_MAX_SIZE = 190 * 1024;
using namespace AscendC;

template <typename T>
ZBCCL_KERNEL void SetAtomicOp(uint32_t atomicOp)
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
ZBCCL_KERNEL T CeilDiv(const T dividend, const T divisor)
{
    return (divisor == 0) ? 0 : ((dividend + divisor - 1) / divisor);
}

ZBCCL_KERNEL void dcciCacheline(__gm__ uint8_t *addr)
{
    using namespace AscendC;
    GlobalTensor<uint8_t> global;
    global.SetGlobalBuffer(addr);

    // Important: add hint to avoid dcci being optimized by compiler
    __asm__ __volatile__("");
    DataCacheCleanAndInvalid<uint8_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(global);
    __asm__ __volatile__("");
}

ZBCCL_KERNEL __gm__ void *zbccl_ptr(__gm__ void *ptr, int curPe, int dstPe, uint64_t localMemSize, 
                                    __gm__ uint16_t *peerGroupRank2WorldRank)
{
    int worldDstPe = static_cast<int>(*((__gm__ uint16_t *)(peerGroupRank2WorldRank + dstPe)));
    int worldCurPe = static_cast<int>(*((__gm__ uint16_t *)(peerGroupRank2WorldRank + curPe)));
    uint64_t curPtr = reinterpret_cast<uint64_t>(ptr);
    uint64_t dstPtr = curPtr + (worldDstPe - worldCurPe) * localMemSize;
    return reinterpret_cast<__gm__ void *>(dstPtr);
}

ZBCCL_KERNEL void Barrier(__gm__ void *paramAddr, uint16_t rankId, uint16_t groupSize, uint64_t localDeviceMemSize,
                               __gm__ uint16_t *peerGroupRank2WorldRank)
{
    AscendC::SyncAll<true>();
    __gm__ uint64_t *ctrlFlagsGM;
    AscendC::LocalTensor<uint64_t> localSetTensor(AscendC::TPosition::VECIN, 8*64 + 32, 2);
    AscendC::LocalTensor<uint64_t> localCheckTensor(AscendC::TPosition::VECIN, 9*64 + 32, 2);
    AscendC::DataCopyExtParams copyParams = {1U, static_cast<uint32_t>(64), 0, 0, 0};
    GlobalTensor<uint64_t> globalTensor;
    if (GetBlockIdx() == 0) {
        PipeBarrier<PIPE_ALL>();
        // rankId = 0, targetRank = 1, rankId = 1, targetRank = 0,
        for (int i = 1; i < groupSize; i++) {
            uint32_t targetRank = (rankId + i) % groupSize;
            auto ptr = zbccl_ptr((__gm__ uint64_t *)(paramAddr), rankId, targetRank, localDeviceMemSize, peerGroupRank2WorldRank);
            globalTensor.SetGlobalBuffer((__gm__ uint64_t *)ptr, BARRIER_FLAG_SIZE * groupSize);
            localSetTensor.SetValue(0, 1212);
            AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
            AscendC::DataCopyPad(globalTensor[rankId * BARRIER_FLAG_SIZE], localSetTensor, copyParams);
        }
        PipeBarrier<PIPE_ALL>();
        for (int i = 1; i < groupSize; i++) { //rankId = 1, targetRank = 0
            uint32_t targetRank = (rankId + i) % groupSize;
            globalTensor.SetGlobalBuffer((__gm__ uint64_t *)paramAddr, BARRIER_FLAG_SIZE * groupSize);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            while (true) {
                AscendC::DataCopyPadExtParams<uint64_t> copyExtParams{false, 0U, 0U, 0U};
                AscendC::DataCopyPad(localCheckTensor, globalTensor[targetRank * BARRIER_FLAG_SIZE], copyParams, copyExtParams);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
                if (localCheckTensor.GetValue(0) == 1212) {
                    break;
                }
            }
        }
        PipeBarrier<PIPE_ALL>();
        for (int i = 1; i < groupSize; i++) {
            uint32_t targetRank = (rankId + i) % groupSize;
            globalTensor.SetGlobalBuffer((__gm__ uint64_t *)paramAddr, BARRIER_FLAG_SIZE * groupSize);
            localSetTensor.SetValue(0, 0);
            AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
            AscendC::DataCopyExtParams copyParams = {1U, static_cast<uint32_t>(64), 0, 0, 0};
            AscendC::DataCopyPad(globalTensor[targetRank * BARRIER_FLAG_SIZE], localSetTensor, copyParams);
        }
    }
    AscendC::SyncAll<true>();
}

template<typename T>
ZBCCL_KERNEL T zbccl_load(__gm__ T *addr)
{
    dcciCacheline((__gm__ uint8_t *) addr);
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

ZBCCL_KERNEL void zbccl_barrier_npu(uint16_t rankId, uint16_t groupSize, uint64_t localSize,
                                    __gm__ uint64_t *counterAddress, __gm__ uint64_t *barrierAddress, 
                                    __gm__ uint16_t *peerGroupRank2WorldRank)
{
    int vecId = AscendC::GetBlockIdx();
    int vecSize = AscendC::GetBlockNum() * AscendC::GetTaskRation();

    int k = 8;
    k = k < groupSize ? k : groupSize;
    k = k < vecSize ? k : vecSize;

    uint64_t count = zbccl_load<uint64_t>(counterAddress) + 1;
    if (vecId == rankId % vecSize) {
        zbccl_single_set(barrierAddress, count);
    }

    for (int i = vecId; i < groupSize; i += k) {
        __gm__ uint64_t *target_addr = (__gm__ uint64_t *)zbccl_ptr((__gm__ void *)barrierAddress, rankId, i, localSize, 
                                                                    peerGroupRank2WorldRank);
        zbccl_single_wait_until_eq(target_addr, count);
    }

    if (vecId == rankId % vecSize) {
        zbccl_single_set(counterAddress, count);
    }
}

ZBCCL_KERNEL void zbccl_barrier_all(uint16_t rankId, uint16_t groupSize, uint64_t localSize,
                                    __gm__ uint64_t *counterAddress, __gm__ uint64_t *barrierAddress,
                                    __gm__ uint16_t *peerGroupRank2WorldRank)
{
    AscendC::SyncAll<true>();

    if ASCEND_IS_AIV {
        zbccl_barrier_npu(rankId, groupSize, localSize, counterAddress, barrierAddress, peerGroupRank2WorldRank);
    }

    AscendC::SyncAll<true>();
}

ZBCCL_KERNEL void zbccl_barrier_all(__gm__ CommGroupInfo *groupInfo)
{
    zbccl_barrier_all(groupInfo->myGroupRank, groupInfo->groupSize, groupInfo->localDeviceMemSize,
                      (__gm__ uint64_t *)&groupInfo->counter, (__gm__ uint64_t *)&groupInfo->barrier, 
                      (__gm__ uint16_t *)&groupInfo->peerGroupRank2WorldRank);
}

template <AscendC::HardEvent event>
ZBCCL_KERNEL void SyncFunc()
{
    int32_t eventID = static_cast<int32_t>(GetTPipePtr()->FetchEventID(event));
    AscendC::SetFlag<event>(eventID);
    AscendC::WaitFlag<event>(eventID);
}

template <AscendC::HardEvent event>
ZBCCL_KERNEL void SyncFunc(int32_t eventID)
{
    AscendC::SetFlag<event>(eventID);
    AscendC::WaitFlag<event>(eventID);
}

template<typename T>
ZBCCL_KERNEL void CpGM2GM(AscendC::GlobalTensor<T> outputGT, AscendC::GlobalTensor<T> inputGT, uint64_t count)
{
    uint32_t copyUbSize = UB_DMA_MAX_SIZE;
    uint32_t copyUbNum = copyUbSize / sizeof(T);
    AscendC::LocalTensor<T> buf(AscendC::TPosition::VECIN, 1024 + 32, copyUbNum);

    uint64_t curOffset = 0;
    AscendC::DataCopyPadExtParams<T> copyExtParams;
    while (count > 0) {
        uint64_t curCount = count > copyUbNum ? copyUbNum : count;
        AscendC::DataCopyExtParams copyParams(1, curCount * sizeof(T), 0, 0, 0);

        AscendC::DataCopyPad(buf, inputGT[curOffset], copyParams, copyExtParams);
        SyncFunc<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
        AscendC::DataCopyPad(outputGT[curOffset], buf, copyParams);
        if (count > copyUbNum) {
            SyncFunc<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        }
        count -= curCount;
        curOffset += curCount;
    }
    return;
}

#endif // ZBCCL_KERNEL_UTILS_H