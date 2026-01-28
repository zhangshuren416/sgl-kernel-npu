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
#include "zbccl_kernel_def.h"

constexpr int64_t FLAG_SIZE = 16;
constexpr int64_t UB_DMA_MAX_SIZE = 190 * 1024;

ZBCCL_KERNEL __gm__ void *zbccl_ptr(__gm__ void *ptr, int curPe, int dstPe, uint64_t localMemSize)
{
    uint64_t curPtr = reinterpret_cast<uint64_t>(ptr);
    uint64_t dstPtr = curPtr + (dstPe - curPe) * localMemSize;
    return reinterpret_cast<__gm__ void *>(dstPtr);
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

#endif // ZBCCL_KERNEL_UTILS_H