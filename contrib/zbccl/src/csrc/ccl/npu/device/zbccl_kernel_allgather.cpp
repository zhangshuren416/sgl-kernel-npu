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

#include "zbccl_kernel_allgather.h"

extern "C" __global__ __aicore__
void allgather(GM_ADDR input, GM_ADDR output, size_t elements, int dataType, GM_ADDR metaAddr)
{
    AllGatherKernel op;
    zbccl_datatype_t ZBCCL_DATA_TYPE = static_cast<zbccl_datatype_t>(dataType);

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

int32_t ZBCCL_OP_AllGather(const void *sendBuff, void *recvBuff, size_t sendCount, zbccl_datatype_t dataType,
                           aclrtStream stream, const CommGroupInfo &groupInfo)
{
    int32_t blockDim = 24;
    int dataTypeInt = static_cast<int>(dataType);
    uint8_t *metaAddr = reinterpret_cast<uint8_t *>(groupInfo.myMetaGva);
    uint8_t *realSendBuff = reinterpret_cast<uint8_t *>(const_cast<void *>(sendBuff));
    uint8_t *realRecvBuff = reinterpret_cast<uint8_t *>(recvBuff);

    allgather<<<blockDim, nullptr, stream>>>(realSendBuff, realRecvBuff, sendCount, dataTypeInt, metaAddr);
    return 0;
}