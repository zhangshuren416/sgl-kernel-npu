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
#define ZBCCL_CORE_BARRIER_SHIFT 2

typedef int32_t zbccl_barrier_bit[ZBCCL_SCALAR_CACHELINE_SIZE / sizeof(int32_t)];

constexpr int64_t FLAG_SIZE = 8;
constexpr int64_t UB_PAD_COUNT = 4;
constexpr int64_t UB_ALIGN_SIZE = 32;
constexpr int64_t UB_BUFF_INTERVAL = 64;
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

ZBCCL_KERNEL __gm__ void *zbccl_ptr(__gm__ void *ptr, int curPe, int dstPe,
                                    uint64_t localSize, __gm__ uint16_t *peerRanks)
{
    int worldDstPe = static_cast<int>(*((__gm__ uint16_t *)(peerRanks + dstPe)));
    int worldCurPe = static_cast<int>(*((__gm__ uint16_t *)(peerRanks + curPe)));
    uint64_t curPtr = reinterpret_cast<uint64_t>(ptr);
    uint64_t dstPtr = curPtr + (worldDstPe - worldCurPe) * localSize;
    return reinterpret_cast<__gm__ void *>(dstPtr);
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

template<typename T>
ZBCCL_KERNEL void zbccl_single_set(__gm__ T *addr, T val)
{
    zbccl_store(addr, val);
    dcciCacheline((__gm__ uint8_t *) addr);
}

template<typename T>
ZBCCL_KERNEL void zbccl_single_wait_until_eq(__gm__ T *syncAddr, T cmp_val)
{
    T cur_val;
    do {
        dcciCacheline((__gm__ uint8_t *)syncAddr);
        cur_val = *syncAddr;
    } while(!(cur_val == cmp_val || cur_val == (cmp_val + 1)));
}

ZBCCL_KERNEL void zbccl_barrier_core_soft(uint16_t rankId, uint16_t groupSize, uint64_t localSize,
                                          __gm__ uint8_t *counterAddress, __gm__ uint8_t *barrierAddress)
{
    if ASCEND_IS_AIC {
        return;
    }

    int32_t block_idx = AscendC::GetBlockIdx();
    int32_t block_dim = AscendC::GetBlockNum() * AscendC::GetTaskRation();
    auto coreBarrier = (__gm__ zbccl_barrier_bit *)barrierAddress;
    auto coreCounter = (__gm__ zbccl_barrier_bit *)counterAddress;
    int32_t count = zbccl_load<int32_t>((__gm__ int32_t *)coreCounter) + 1;

    int32_t shift = 1;
    int32_t offset = 0;
    while (shift < block_dim) {
        int32_t next = (block_idx + shift) % block_dim;
        zbccl_single_set((__gm__ int32_t *)(coreBarrier + next * ZBCCL_AIV_MAX_EXP_NUM + offset), count);
        zbccl_single_wait_until_eq((__gm__ int32_t *)(coreBarrier + block_idx * ZBCCL_AIV_MAX_EXP_NUM + offset), count);
        shift *= ZBCCL_CORE_BARRIER_SHIFT;
        offset++;
    }
    zbccl_store((__gm__ int32_t *)coreCounter, count);
}

ZBCCL_KERNEL void zbccl_barrier_core(uint16_t rankId, uint16_t groupSize, uint64_t localSize,
                                     __gm__ uint8_t *counterAddress, __gm__ uint8_t *barrierAddress)
{
#ifdef __CCE_AICORE_ENABLE_MIX__
    AscendC::SyncAll<true>();
#else
    zbccl_barrier_core_soft(rankId, groupSize, localSize, counterAddress, barrierAddress);
#endif
}

ZBCCL_KERNEL void zbccl_barrier_all(uint16_t rankId, uint16_t groupSize, uint64_t localSize,
                                    __gm__ uint64_t *counterAddress, __gm__ uint64_t *barrierAddress,
                                    __gm__ uint8_t *coreCounterAddress, __gm__ uint8_t *coreBarrierAddress,
                                    __gm__ uint16_t *peerRanks)
{
    int vecId = AscendC::GetBlockIdx();
    int vecSize = AscendC::GetBlockNum() * AscendC::GetTaskRation();
    uint64_t count = zbccl_load<uint64_t>(counterAddress) + 1;
    int k = 8;
    k = k < groupSize ? k : groupSize;
    k = k < vecSize ? k : vecSize;

    zbccl_barrier_core(rankId, groupSize, localSize, coreCounterAddress, coreBarrierAddress);

    if ASCEND_IS_AIV {
        if (vecId == rankId) {
            zbccl_single_set(barrierAddress, count);
        }
        for (int i = vecId; i < groupSize; i += k) {
            __gm__ void *dst = zbccl_ptr((__gm__ void *)barrierAddress, rankId, i, localSize, peerRanks);
            __gm__ uint64_t *target_addr = (__gm__ uint64_t *)dst;
            zbccl_single_wait_until_eq(target_addr, count);
        }
        if (vecId == rankId) {
            zbccl_single_set(counterAddress, count);
        }
    }

    zbccl_barrier_core(rankId, groupSize, localSize, coreCounterAddress, coreBarrierAddress);
}



template <AscendC::HardEvent event>
ZBCCL_KERNEL void SyncFunc(int32_t eventID)
{
    AscendC::SetFlag<event>(eventID);
    AscendC::WaitFlag<event>(eventID);
}

ZBCCL_KERNEL void SetMetaValue(__gm__ uint64_t *ptr, uint32_t rankId, uint64_t value, uint16_t groupSize,
                               AscendC::LocalTensor<uint64_t> localTensor)
{
    GlobalTensor<uint64_t> globalTensor;
    globalTensor.SetGlobalBuffer((__gm__ uint64_t *)ptr, groupSize * FLAG_SIZE);
    localTensor.SetValue(0, value);
    SyncFunc<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
    AscendC::DataCopy(globalTensor[rankId * FLAG_SIZE], localTensor, UB_PAD_COUNT);
}

ZBCCL_KERNEL void WaitMetaValue(__gm__ uint64_t *ptr, uint32_t rankId, uint64_t value, uint16_t groupSize,
                               AscendC::LocalTensor<uint64_t> localTensor)
{
    GlobalTensor<uint64_t> globalTensor;
    globalTensor.SetGlobalBuffer((__gm__ uint64_t *)ptr, groupSize * FLAG_SIZE);
    SyncFunc<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    while (true) {
        AscendC::DataCopy(localTensor, globalTensor[rankId * FLAG_SIZE], UB_PAD_COUNT);
        SyncFunc<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
        if (localTensor.GetValue(0) == value) {
            break;
        }
    }
}

ZBCCL_KERNEL void GetMetaValue(__gm__ uint64_t *ptr, uint32_t rankId, uint16_t groupSize,
                               AscendC::LocalTensor<uint64_t> localTensor)
{
    GlobalTensor<uint64_t> globalTensor;
    globalTensor.SetGlobalBuffer((__gm__ uint64_t *)ptr, groupSize * FLAG_SIZE);
    SyncFunc<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    AscendC::DataCopy(localTensor, globalTensor[rankId * FLAG_SIZE], UB_PAD_COUNT);
}

ZBCCL_KERNEL void Barrier(__gm__ void *paramAddr, uint16_t myGroupRank, uint16_t groupSize,
                          uint64_t localDeviceMemSize, __gm__ uint16_t *peerGroupRank2WorldRank)
{
    AscendC::SyncAll<true>();
    const uint64_t barrierMagic = 1024;
    const uint64_t BarrierOffset = 512;
    const int64_t aivIndex = AscendC::GetBlockIdx();
    if (aivIndex < groupSize) {
        auto ptr = zbccl_ptr((__gm__ uint64_t *)(paramAddr), myGroupRank, aivIndex,
                             localDeviceMemSize, peerGroupRank2WorldRank);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::LocalTensor<uint64_t> localSetTensor(AscendC::TPosition::VECIN, BarrierOffset + UB_ALIGN_SIZE, UB_PAD_COUNT);
        SetMetaValue((__gm__ uint64_t *)ptr, myGroupRank, barrierMagic, groupSize, localSetTensor);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::LocalTensor<uint64_t> localCheckTensor(AscendC::TPosition::VECIN,
                                                        BarrierOffset + UB_BUFF_INTERVAL + UB_ALIGN_SIZE, UB_PAD_COUNT);
        WaitMetaValue((__gm__ uint64_t *)(paramAddr), aivIndex, barrierMagic, groupSize, localCheckTensor);
        AscendC::PipeBarrier<PIPE_ALL>();
        SetMetaValue((__gm__ uint64_t *)(paramAddr), aivIndex, 0, groupSize, localSetTensor);
    }
    AscendC::SyncAll<true>();
}

template<typename T>
ZBCCL_KERNEL void CpGM2GM(AscendC::GlobalTensor<T> outputGT, AscendC::GlobalTensor<T> inputGT, uint64_t count)
{
    uint32_t copyUbSize = UB_DMA_MAX_SIZE;
    uint32_t copyUbNum = copyUbSize / sizeof(T);
    const uint64_t CpOffset = 1024;
    AscendC::LocalTensor<T> buf(AscendC::TPosition::VECIN, CpOffset + UB_ALIGN_SIZE, copyUbNum);
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