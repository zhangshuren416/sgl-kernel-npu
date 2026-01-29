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

constexpr int64_t FLAG_SIZE = 16;
constexpr int64_t UB_DMA_MAX_SIZE = 190 * 1024;


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

ZBCCL_KERNEL __gm__ void *zbccl_ptr(__gm__ void *ptr, int curPe, int dstPe, uint64_t localMemSize)
{
    uint64_t curPtr = reinterpret_cast<uint64_t>(ptr);
    uint64_t dstPtr = curPtr + (dstPe - curPe) * localMemSize;
    return reinterpret_cast<__gm__ void *>(dstPtr);
}

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

ZBCCL_KERNEL void ExchangeInputAddr(GM_ADDR inputGM, GM_ADDR metaGM, uint16_t groupSize, uint16_t myGroupRank,
                                    uint64_t flagMagic, uint64_t localDeviceMemSize)
{
    const int64_t aivNum = AscendC::GetBlockNum();
    const int64_t aivIndex = AscendC::GetBlockIdx();

    int64_t addrOffset = myGroupRank * FLAG_SIZE;

    AscendC::LocalTensor<uint64_t> inputBuff(AscendC::TPosition::VECIN, 32, 1);
    inputBuff(0) = reinterpret_cast<uint64_t>(inputGM);
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);

    AscendC::LocalTensor<uint64_t> flagbuff(AscendC::TPosition::VECIN, 96, 1);
    flagbuff(0) = flagMagic;
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
    AscendC::GlobalTensor<uint64_t> metaAddrTensor;

    if (aivIndex < groupSize) {
        // write addr
        auto exchangeAddr = reinterpret_cast<__gm__ CommGroupInfo *>(metaGM)->myAddressExchangeGva;
        auto ptr = zbccl_ptr((__gm__ uint64_t *)(exchangeAddr), myGroupRank, aivIndex, localDeviceMemSize);
        metaAddrTensor.SetGlobalBuffer((__gm__ uint64_t *)ptr, groupSize * FLAG_SIZE * 2);  // size ??
        AscendC::DataCopyExtParams copyParams = {1U, static_cast<uint32_t>(64), 0, 0, 0};
        AscendC::DataCopyPad(metaAddrTensor[addrOffset], inputBuff, copyParams);

        //write flag
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopyPad(metaAddrTensor[addrOffset + groupSize * FLAG_SIZE], flagbuff, copyParams);
    }
}

template<typename T>
ZBCCL_KERNEL void CpGM2GM(AscendC::GlobalTensor<T> outputGT, AscendC::GlobalTensor<T> inputGT, uint64_t count)
{
    uint32_t copyUbSize = UB_DMA_MAX_SIZE / 2;
    uint32_t copyUbNum = copyUbSize / sizeof(T);
    AscendC::LocalTensor<T> pingBuff(AscendC::TPosition::VECIN, 1024 + 32, copyUbNum);
    AscendC::LocalTensor<T> pongBuff(AscendC::TPosition::VECIN, 96 * 1024 + 32, copyUbNum);

    uint64_t curOffset = 0;
    AscendC::DataCopyPadExtParams<T> copyExtParams;
    uint8_t pingpongId = 0;
    while (count > 0) {
        AscendC::TEventID EVENT_ID = pingpongId == 0 ? EVENT_ID0 : EVENT_ID1;
        AscendC::LocalTensor<T> buf = pingpongId == 0 ? pingBuff : pongBuff;
        uint64_t curCount = count > copyUbNum ? copyUbNum : count;
        AscendC::DataCopyExtParams copyParams(1, curCount * sizeof(T), 0, 0, 0);

        AscendC::DataCopyPad(buf, inputGT[curOffset], copyParams, copyExtParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        AscendC::DataCopyPad(outputGT[curOffset], buf, copyParams);
        if (count > copyUbNum) {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
        }
        count -= curCount;
        curOffset += curCount;
        pingpongId = 1 - pingpongId;
    }
    return;
}

template <AscendC::HardEvent event>
ZBCCL_KERNEL void SyncFunc()
{
    int32_t eventID = static_cast<int32_t>(GetTPipePtr()->FetchEventID(event));
    AscendC::SetFlag<event>(eventID);
    AscendC::WaitFlag<event>(eventID);
}

#endif // ZBCCL_KERNEL_UTILS_H