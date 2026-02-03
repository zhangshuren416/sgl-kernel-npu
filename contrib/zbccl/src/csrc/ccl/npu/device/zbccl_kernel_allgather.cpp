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

#include "kernel_operator.h"
#include "zbccl_def.h"
#include "zbccl_kernel_utils.h"

class AllGatherKernel
{
public:
    ZBCCL_KERNEL AllGatherKernel() {}

    ZBCCL_KERNEL void SetFlagValue(__gm__ uint64_t *ptr, uint32_t rankId ,uint64_t value)
    {
        GlobalTensor<uint64_t> globalTensor;
        AscendC::LocalTensor<uint64_t> localSetTensor(AscendC::TPosition::VECIN, 32, 1);
        globalTensor.SetGlobalBuffer((__gm__ uint64_t *)ptr, FLAG_SIZE * groupSize); 
        localSetTensor.SetValue(0, value); 
        SyncFunc<AscendC::HardEvent::S_MTE3>(EVENT_ID0); 
        AscendC::DataCopyPad(globalTensor[rankId * FLAG_SIZE], localSetTensor, copyParams); 
    }

    ZBCCL_KERNEL void WaitFlagValue(__gm__ uint64_t *ptr, uint32_t rankId, uint64_t value)
    {
        GlobalTensor<uint64_t> globalTensor;
        AscendC::LocalTensor<uint64_t> localCheckTensor(AscendC::TPosition::VECIN, 64 + 32, 1);
        globalTensor.SetGlobalBuffer((__gm__ uint64_t *)ptr, FLAG_SIZE * groupSize); 
        SyncFunc<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0); 
        while (true) { 
            AscendC::DataCopyPad(localCheckTensor, globalTensor[rankId * FLAG_SIZE], copyParams, copyExtParams); 
            SyncFunc<AscendC::HardEvent::MTE2_S>(EVENT_ID0); 
            if (localCheckTensor.GetValue(0) == value) { 
                break; 
            }
        }
    }

    ZBCCL_KERNEL void Barrier(GM_ADDR metaGM)
    { 
        AscendC::SyncAll<true>(); 
        auto paramAddr = reinterpret_cast<__gm__ CommGroupInfo *>(metaGM)->myParamDataGva; 
        if (GetBlockIdx() == 0) { 
            PipeBarrier<PIPE_ALL>(); 
            // myGroupRank = 0, targetRank = 1, myGroupRank = 1, targetRank = 0, 
            for (int i = 1; i < groupSize; i++) { 
                uint32_t targetRank = (myGroupRank + i) % groupSize; 
                auto ptr = zbccl_ptr((__gm__ uint64_t *)(paramAddr), myGroupRank, targetRank, localDeviceMemSize, peerGroupRank2WorldRank);
                SetFlagValue((__gm__ uint64_t *)ptr, myGroupRank ,1212);
            } 
            PipeBarrier<PIPE_ALL>(); 
            for (int i = 1; i < groupSize; i++) { //myGroupRank = 1, targetRank = 0 
                uint32_t targetRank = (myGroupRank + i) % groupSize; 
                WaitFlagValue((__gm__ uint64_t *)(paramAddr), targetRank, 1212);
            } 
            PipeBarrier<PIPE_ALL>(); 
            for (int i = 1; i < groupSize; i++) { 
                uint32_t targetRank = (myGroupRank + i) % groupSize;
                SetFlagValue((__gm__ uint64_t *)(paramAddr), targetRank, 0); 
            } 
        } 
        AscendC::SyncAll<true>(); 
    }

    ZBCCL_KERNEL void ExchangeInputAddr(GM_ADDR inputGM, GM_ADDR metaGM, uint64_t flagMagic)
    {
        const int64_t aivNum = AscendC::GetBlockNum();
        const int64_t aivIndex = AscendC::GetBlockIdx();
        int64_t addrOffset = myGroupRank * FLAG_SIZE;

        AscendC::LocalTensor<uint64_t> inputBuff(AscendC::TPosition::VECIN, 2*64 + 32, 1);
        inputBuff(0) = reinterpret_cast<uint64_t>(inputGM);
        SyncFunc<AscendC::HardEvent::S_MTE3>(EVENT_ID0);

        AscendC::LocalTensor<uint64_t> flagbuff(AscendC::TPosition::VECIN, 3*64 + 32, 1);
        flagbuff(0) = flagMagic;
        SyncFunc<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::GlobalTensor<uint64_t> metaAddrTensor;

        if (aivIndex < groupSize) {
            // write addr
            auto exchangeAddr = reinterpret_cast<__gm__ CommGroupInfo *>(metaGM)->myAddressExchangeGva;
            auto ptr = zbccl_ptr((__gm__ uint64_t *)(exchangeAddr), myGroupRank, aivIndex, localDeviceMemSize, peerGroupRank2WorldRank);
            metaAddrTensor.SetGlobalBuffer((__gm__ uint64_t *)ptr, inputAddrSize * 2); 
            AscendC::DataCopyPad(metaAddrTensor[addrOffset], inputBuff, copyParams);

            //write flag
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::DataCopyPad(metaAddrTensor[addrOffset + inputAddrSize], flagbuff, copyParams);
        }
    }

    template<typename T>
    ZBCCL_KERNEL void Process(GM_ADDR input, GM_ADDR output, GM_ADDR metaGM, uint64_t elements)
    {
#ifdef __DAV_C220_VEC__
        __gm__ CommGroupInfo *comm = reinterpret_cast<__gm__ CommGroupInfo *>(metaGM);
        groupSize = comm->groupSize;
        myGroupRank = comm->myGroupRank;
        localDeviceMemSize = comm->localDeviceMemSize;
        peerGroupRank2WorldRank = reinterpret_cast<__gm__ uint16_t *>(comm->peerGroupRank2WorldRank);
        copyParams = {1U, static_cast<uint32_t>(64), 0, 0, 0};
        inputAddrSize = groupSize * FLAG_SIZE;
        const int64_t aivNum = AscendC::GetBlockNum() * AscendC::GetTaskRation();
        const int64_t aivIndex = AscendC::GetBlockIdx();

        AscendC::LocalTensor<uint64_t> flagbuff(AscendC::TPosition::VECIN, 4*64 + 32, 1);
        flagbuff(0) = 0;
        SyncFunc<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::GlobalTensor<uint64_t> metaAddrTensor;
        int64_t addrOffset = myGroupRank * FLAG_SIZE;
        if (aivIndex < groupSize) {
            auto exchangeAddr = reinterpret_cast<__gm__ CommGroupInfo *>(metaGM)->myAddressExchangeGva;
            auto ptr = zbccl_ptr((__gm__ uint64_t *)(exchangeAddr), myGroupRank, aivIndex, localDeviceMemSize, peerGroupRank2WorldRank);
            metaAddrTensor.SetGlobalBuffer((__gm__ uint64_t *)ptr, inputAddrSize * 2); 
            AscendC::DataCopyPad(metaAddrTensor[addrOffset + inputAddrSize], flagbuff, copyParams);
        }

        //zbccl_barrier_all(myGroupRank, groupSize, localDeviceMemSize, counterAddress, barrierAddress, peerGroupRank2WorldRank);
        Barrier(metaGM);

        uint64_t flagMagic = 1024;
        ExchangeInputAddr(input, metaGM, flagMagic);

        // data move parameters
        const int64_t corePerRank = aivNum / groupSize; // 16 / 16 = 1
        const int64_t coreRankIdx = aivIndex % corePerRank; // 0
        const int64_t x = aivIndex / corePerRank; // 0,1,2...15

        uint64_t flag = 0;
        AscendC::LocalTensor<uint64_t> flagBuff(AscendC::TPosition::VECIN, 5*64 + 32, 1);
        flagBuff(0) = 0;
        SyncFunc<AscendC::HardEvent::S_MTE2>(EVENT_ID0);
        auto exchangeAddr = reinterpret_cast<__gm__ CommGroupInfo *>(metaGM)->myAddressExchangeGva;
        metaAddrTensor.SetGlobalBuffer((__gm__ uint64_t *)exchangeAddr, inputAddrSize * 2);

        while (flag != flagMagic) {
            SyncFunc<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            AscendC::DataCopyPad(flagBuff, metaAddrTensor[inputAddrSize + x * FLAG_SIZE], copyParams, copyExtParams);
            SyncFunc<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
            flag = flagBuff.GetValue(0);
        }

        uint32_t numPerCore = elements / corePerRank;
        uint32_t outputOffset = x * elements + coreRankIdx * numPerCore;
        uint32_t inputOffset = coreRankIdx * numPerCore;
        if (coreRankIdx == corePerRank - 1) {
            numPerCore = elements - (corePerRank - 1) * numPerCore;
        }

        AscendC::LocalTensor<uint64_t> inputAddrBuff(AscendC::TPosition::VECIN, 6*64 + 32, 1);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopyPad(inputAddrBuff, metaAddrTensor[x * FLAG_SIZE], copyParams, copyExtParams);
        SyncFunc<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
        AscendC::GlobalTensor<T> outputGT;
        outputGT.SetGlobalBuffer((__gm__ T *)output, elements * groupSize);
        AscendC::GlobalTensor<T> inputGT;
        inputGT.SetGlobalBuffer((__gm__ T *)inputAddrBuff.GetValue(0), elements);

        AscendC::PipeBarrier<PIPE_ALL>();
        CpGM2GM(outputGT[outputOffset], inputGT[inputOffset], numPerCore);
#endif
    }

private:
    AscendC::DataCopyExtParams copyParams;
    AscendC::DataCopyPadExtParams<uint64_t> copyExtParams;
    uint16_t groupSize;
    uint16_t myGroupRank;
    uint64_t localDeviceMemSize;
    uint16_t inputAddrSize;
    __gm__ uint16_t *peerGroupRank2WorldRank;
};

extern "C" __global__ __aicore__
void ZBCCLAllGatherInner(GM_ADDR input, GM_ADDR output, size_t elements, int dataType, GM_ADDR metaAddr)
{
    AllGatherKernel op;
    zbccl_datatype_t ZBCCL_DATA_TYPE = static_cast<zbccl_datatype_t>(dataType);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);

    switch (ZBCCL_DATA_TYPE){
        case zbccl_datatype_t::ZBCCL_DATA_TYPE_INT8:
            op.Process<int8_t>(input, output, metaAddr, elements);
            break;
        case zbccl_datatype_t::ZBCCL_DATA_TYPE_INT16:
            op.Process<int16_t>(input, output, metaAddr, elements);
            break;
        case zbccl_datatype_t::ZBCCL_DATA_TYPE_INT32:
            op.Process<int32_t>(input, output, metaAddr, elements);
            break;
        case zbccl_datatype_t::ZBCCL_DATA_TYPE_FP16:
            op.Process<float16_t>(input, output, metaAddr, elements);
            break;
        case zbccl_datatype_t::ZBCCL_DATA_TYPE_FP32:
            op.Process<float>(input, output, metaAddr, elements);
            break;
        case zbccl_datatype_t::ZBCCL_DATA_TYPE_INT64:
            op.Process<int64_t>(input, output, metaAddr, elements);
            break;
        case zbccl_datatype_t::ZBCCL_DATA_TYPE_UINT64:
            op.Process<uint64_t>(input, output, metaAddr, elements);
            break;
        case zbccl_datatype_t::ZBCCL_DATA_TYPE_UINT8:
            op.Process<uint8_t>(input, output, metaAddr, elements);
            break;
        case zbccl_datatype_t::ZBCCL_DATA_TYPE_UINT16:
            op.Process<uint16_t>(input, output, metaAddr, elements);
            break;
        case zbccl_datatype_t::ZBCCL_DATA_TYPE_UINT32:
            op.Process<uint32_t>(input, output, metaAddr, elements);
            break;
        case zbccl_datatype_t::ZBCCL_DATA_TYPE_FP64:
            op.Process<float64_t>(input, output, metaAddr, elements);
            break;
        case zbccl_datatype_t::ZBCCL_DATA_TYPE_BFP16:
            op.Process<bfloat16_t>(input, output, metaAddr, elements);
            break;
        default:
            break;
    }
}

int32_t ZBCCLOpAllGather(const void *sendBuff, void *recvBuff, size_t sendCount, zbccl_datatype_t dataType,
                           aclrtStream stream, const CommGroupInfo &groupInfo)
{
    int32_t blockDim = 16;
    int dataTypeInt = static_cast<int>(dataType);
    uint8_t *metaAddr = reinterpret_cast<uint8_t *>(groupInfo.myMetaGva);
    uint8_t *realSendBuff = reinterpret_cast<uint8_t *>(const_cast<void *>(sendBuff));
    uint8_t *realRecvBuff = reinterpret_cast<uint8_t *>(recvBuff);

    ZBCCLAllGatherInner<<<blockDim, nullptr, stream>>>(realSendBuff, realRecvBuff, sendCount, dataTypeInt, metaAddr);
    return 0;
}