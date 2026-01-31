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
    ZBCCL_KERNEL void Process(GM_ADDR input, GM_ADDR output, GM_ADDR metaGM, uint64_t elements)
    {
#ifdef __DAV_C220_VEC__
        __gm__ CommGroupInfo *comm = reinterpret_cast<__gm__ CommGroupInfo *>(metaGM);
        uint16_t groupSize = comm->groupSize;
        uint16_t myGroupRank = comm->myGroupRank;
        uint64_t localDeviceMemSize = comm->localDeviceMemSize;

        Barrier(metaGM, myGroupRank, groupSize, localDeviceMemSize);

        uint64_t flagMagic = 1024;
        ExchangeInputAddr(input, metaGM, groupSize, myGroupRank, flagMagic, localDeviceMemSize);

        __gm__ uint32_t *exchange = reinterpret_cast<__gm__ uint32_t *>(comm->myAddressExchangeGva);

        const int64_t aivNum = AscendC::GetBlockNum() * AscendC::GetTaskRation();
        const int64_t aivIndex = AscendC::GetBlockIdx();

        // data move parameters
        const int64_t corePerRank = aivNum / groupSize; // 4
        const int64_t coreRankIdx = aivIndex % corePerRank; // 0,1,2,3
        const int64_t x = aivIndex / corePerRank; //0,1

        uint64_t flag = 0;
        AscendC::LocalTensor<uint64_t> flagBuff(AscendC::TPosition::VECIN, 2*64 + 32, 1);
        flagBuff(0) = 0;
        AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID0);

        auto exchangeAddr = reinterpret_cast<__gm__ CommGroupInfo *>(metaGM)->myAddressExchangeGva;
        AscendC::GlobalTensor<uint64_t> meta_addr_tensor;
        meta_addr_tensor.SetGlobalBuffer((__gm__ uint64_t *)exchangeAddr, groupSize * FLAG_SIZE * 2);
        AscendC::DataCopyPadExtParams<uint64_t> copyExtParams{false, 0U, 0U, 0U};
        AscendC::DataCopyExtParams copyParams{1U, static_cast<uint32_t>(64), 0, 0, 0};

        while (flag != flagMagic) {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            AscendC::DataCopyPad(flagBuff, meta_addr_tensor[groupSize * FLAG_SIZE + x * FLAG_SIZE], copyParams, copyExtParams);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
            flag = flagBuff.GetValue(0);
        }

        AscendC::LocalTensor<uint64_t> input_addr_buff(AscendC::TPosition::VECIN, 3*64 + 32, 1);
        AscendC::GlobalTensor<T> outputGT;
        outputGT.SetGlobalBuffer((__gm__ T *)output, elements * groupSize);

        uint32_t numPerCore = elements / corePerRank;
        uint32_t outputOffset = x * elements + coreRankIdx * numPerCore;
        uint32_t inputOffset = coreRankIdx * numPerCore;
        if (coreRankIdx == corePerRank - 1) {
            numPerCore = elements - (corePerRank - 1) * numPerCore;
        }

        // get input addr
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopyPad(input_addr_buff, meta_addr_tensor[x * FLAG_SIZE], copyParams, copyExtParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);

        AscendC::GlobalTensor<T> inputGT;
        inputGT.SetGlobalBuffer((__gm__ T *)input_addr_buff.GetValue(0), elements);

        AscendC::PipeBarrier<PIPE_ALL>();
        CpGM2GM(outputGT[outputOffset], inputGT[inputOffset], numPerCore);
#endif
    }

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
    int32_t blockDim = 24;
    int dataTypeInt = static_cast<int>(dataType);
    uint8_t *metaAddr = reinterpret_cast<uint8_t *>(groupInfo.myMetaGva);
    uint8_t *realSendBuff = reinterpret_cast<uint8_t *>(const_cast<void *>(sendBuff));
    uint8_t *realRecvBuff = reinterpret_cast<uint8_t *>(recvBuff);

    ZBCCLAllGatherInner<<<blockDim, nullptr, stream>>>(realSendBuff, realRecvBuff, sendCount, dataTypeInt, metaAddr);
    return 0;
}