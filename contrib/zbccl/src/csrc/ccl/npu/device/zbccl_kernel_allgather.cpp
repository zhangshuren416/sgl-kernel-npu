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

    template<typename T>
    ZBCCL_KERNEL void Init(GM_ADDR input, GM_ADDR output, GM_ADDR metaGM, uint64_t elements)
    {
#ifdef __DAV_C220_VEC__
        this->comm = reinterpret_cast<__gm__ CommGroupInfo *>(metaGM);
        this->groupSize = comm->groupSize;
        this->myGroupRank = comm->myGroupRank;
        this->localDeviceMemSize = comm->localDeviceMemSize;
        this->inputAddrSize = groupSize * FLAG_SIZE;
        this->peerGroupRank2WorldRank = reinterpret_cast<__gm__ uint16_t *>(comm->peerGroupRank2WorldRank);
        this->exchangeAddr = comm->myAddressExchangeGva;
        this->paramAddr = comm->myParamDataGva; 
        this->aivNum = AscendC::GetBlockNum() * AscendC::GetTaskRation();
        this->input = input;
        this->output = output;
        this->elements = elements;
#endif
    }

    ZBCCL_KERNEL void ExchangeInputAddr(__gm__ void *inputGM, __gm__ CommGroupInfo *comm, uint64_t flagMagic, int64_t aivIndex)
    {
        int64_t addrOffset = myGroupRank * FLAG_SIZE;
        AscendC::LocalTensor<uint64_t> inputInBuff(AscendC::TPosition::VECIN, UB_ALIGN_SIZE, UB_PAD_COUNT);
        AscendC::LocalTensor<uint64_t> flagInBuff(AscendC::TPosition::VECIN, UB_BUFF_INTERVAL + UB_ALIGN_SIZE, UB_PAD_COUNT);

        if (aivIndex < groupSize) {
            // write addr
            auto exchangeAddr = comm->myAddressExchangeGva;
            auto ptr = zbccl_ptr((__gm__ uint64_t *)(exchangeAddr), myGroupRank, aivIndex, 
                                 localDeviceMemSize, peerGroupRank2WorldRank);
            SetMetaValue((__gm__ uint64_t *)ptr, myGroupRank, reinterpret_cast<uint64_t>(inputGM), groupSize, inputInBuff);

            //write exchangeFlag
            AscendC::PipeBarrier<PIPE_ALL>();
            SetMetaValue((__gm__ uint64_t *)ptr + inputAddrSize, myGroupRank, flagMagic, groupSize, flagInBuff);
        }
    }

    template<typename T>
    ZBCCL_KERNEL void Process()
    {
#ifdef __DAV_C220_VEC__
        const int64_t aivIndex = AscendC::GetBlockIdx();
        AscendC::LocalTensor<uint64_t> flagClearBuff(AscendC::TPosition::VECIN, 2*UB_BUFF_INTERVAL + UB_ALIGN_SIZE, UB_PAD_COUNT);
        if (aivIndex < groupSize) {
            auto ptr = zbccl_ptr((__gm__ uint64_t *)(exchangeAddr), myGroupRank, 
                            aivIndex, localDeviceMemSize, peerGroupRank2WorldRank);
            SetMetaValue((__gm__ uint64_t *)ptr + inputAddrSize, myGroupRank, 0, groupSize, flagClearBuff);
        }

        //zbccl_barrier_all(myGroupRank, groupSize, localDeviceMemSize, counterAddress, barrierAddress, peerGroupRank2WorldRank);
        Barrier((__gm__ uint64_t *)paramAddr, myGroupRank, groupSize, localDeviceMemSize, peerGroupRank2WorldRank);

        uint64_t flagMagic = 1024;
        ExchangeInputAddr(input, comm, flagMagic, aivIndex);

        // tiling parameters
        const int64_t corePerRank = aivNum / groupSize;
        const int64_t coreRankIdx = aivIndex % corePerRank;
        const int64_t offset = aivIndex / corePerRank;
        uint64_t exchangeFlag = 0;
        uint32_t numPerCore = elements / corePerRank;
        uint32_t outputOffset = offset * elements + coreRankIdx * numPerCore;
        uint32_t inputOffset = coreRankIdx * numPerCore;
        if (coreRankIdx == corePerRank - 1) {
            numPerCore = elements - (corePerRank - 1) * numPerCore;
        }

        AscendC::LocalTensor<uint64_t> flagOutBuff(AscendC::TPosition::VECIN, 3*UB_BUFF_INTERVAL + UB_ALIGN_SIZE, UB_PAD_COUNT);
        WaitMetaValue((__gm__ uint64_t *)exchangeAddr + inputAddrSize, offset, flagMagic, groupSize, flagOutBuff);

        AscendC::LocalTensor<uint64_t> inputOutBuff(AscendC::TPosition::VECIN, 4*UB_BUFF_INTERVAL + UB_ALIGN_SIZE, UB_PAD_COUNT);
        AscendC::PipeBarrier<PIPE_ALL>();
        GetMetaValue((__gm__ uint64_t *)exchangeAddr, offset, groupSize, inputOutBuff);
        SyncFunc<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
        AscendC::GlobalTensor<T> outputGT;
        outputGT.SetGlobalBuffer((__gm__ T *)output, elements * groupSize);
        AscendC::GlobalTensor<T> inputGT;
        inputGT.SetGlobalBuffer((__gm__ T *)inputOutBuff.GetValue(0), elements);

        AscendC::PipeBarrier<PIPE_ALL>();
        CpGM2GM(outputGT[outputOffset], inputGT[inputOffset], numPerCore);
#endif
    }

private:
    uint16_t groupSize;
    uint16_t myGroupRank;
    uint64_t localDeviceMemSize;
    uint16_t inputAddrSize;
    int64_t aivNum;
    uint64_t elements;
    uintptr_t exchangeAddr;
    uintptr_t paramAddr;
    __gm__ void *input;
    __gm__ void *output;
    __gm__ CommGroupInfo *comm;
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
            op.Init<int8_t>(input, output, metaAddr, elements);
            op.Process<int8_t>();
            break;
        case zbccl_datatype_t::ZBCCL_DATA_TYPE_INT16:
            op.Init<int16_t>(input, output, metaAddr, elements);
            op.Process<int16_t>();
            break;
        case zbccl_datatype_t::ZBCCL_DATA_TYPE_INT32:
            op.Init<int32_t>(input, output, metaAddr, elements);
            op.Process<int32_t>();
            break;
        case zbccl_datatype_t::ZBCCL_DATA_TYPE_FP16:
            op.Init<float16_t>(input, output, metaAddr, elements);
            op.Process<float16_t>();
            break;
        case zbccl_datatype_t::ZBCCL_DATA_TYPE_FP32:
            op.Init<float>(input, output, metaAddr, elements);
            op.Process<float>();
            break;
        case zbccl_datatype_t::ZBCCL_DATA_TYPE_INT64:
            op.Init<int64_t>(input, output, metaAddr, elements);
            op.Process<int64_t>();
            break;
        case zbccl_datatype_t::ZBCCL_DATA_TYPE_UINT64:
            op.Init<uint64_t>(input, output, metaAddr, elements);
            op.Process<uint64_t>();
            break;
        case zbccl_datatype_t::ZBCCL_DATA_TYPE_UINT8:
            op.Init<uint8_t>(input, output, metaAddr, elements);
            op.Process<uint8_t>();
            break;
        case zbccl_datatype_t::ZBCCL_DATA_TYPE_UINT16:
            op.Init<uint16_t>(input, output, metaAddr, elements);
            op.Process<uint16_t>();
            break;
        case zbccl_datatype_t::ZBCCL_DATA_TYPE_UINT32:
            op.Init<uint32_t>(input, output, metaAddr, elements);
            op.Process<uint32_t>();
            break;
        case zbccl_datatype_t::ZBCCL_DATA_TYPE_FP64:
            op.Init<float64_t>(input, output, metaAddr, elements);
            op.Process<float64_t>();
            break;
        case zbccl_datatype_t::ZBCCL_DATA_TYPE_BFP16:
            op.Init<bfloat16_t>(input, output, metaAddr, elements);
            op.Process<bfloat16_t>();;
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