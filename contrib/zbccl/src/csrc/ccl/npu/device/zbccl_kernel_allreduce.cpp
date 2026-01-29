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
#include "zbccl_kernel_allreduce.h"


using namespace zbccl;
using namespace zbccl::ccl;

extern "C" __global__ __aicore__ void ZeroBuffAllReduce(
    GM_ADDR input, GM_ADDR output, GM_ADDR gva,
    uint64_t fftsAddr, uint32_t dataType, uint32_t totalLength,
    uint32_t rank, uint32_t groupSize, uint32_t reduceOp)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::SetSyncBaseAddr(fftsAddr);
    uint32_t magic = 1;
    AscendC::TPipe pipe;
    zbccl_datatype_t zbcclDataType = static_cast<zbccl_datatype_t>(dataType);
    switch (zbcclDataType) {
        case zbccl_datatype_t::ZBCCL_DATA_TYPE_INT8: {
            ZeroBuffAllReduceKernel<int8_t> op;
            op.Init(input, output, gva, &pipe, rank, groupSize, totalLength, magic, reduceOp);
            op.Process();
            break;
        }
        case zbccl_datatype_t::ZBCCL_DATA_TYPE_INT16: {
            ZeroBuffAllReduceKernel<int16_t> op;
            op.Init(input, output, gva, &pipe, rank, groupSize, totalLength, magic, reduceOp);
            op.Process();
            break;
        }
        case zbccl_datatype_t::ZBCCL_DATA_TYPE_INT32: {
            ZeroBuffAllReduceKernel<int32_t> op;
            op.Init(input, output, gva, &pipe, rank, groupSize, totalLength, magic, reduceOp);
            op.Process();
            break;
        }
        case zbccl_datatype_t::ZBCCL_DATA_TYPE_FP32: {
            ZeroBuffAllReduceKernel<float> op;
            op.Init(input, output, gva, &pipe, rank, groupSize, totalLength, magic, reduceOp);
            op.Process();
            break;
        }
        case zbccl_datatype_t::ZBCCL_DATA_TYPE_FP16: {
            ZeroBuffAllReduceKernel<float16_t> op;
            op.Init(input, output, gva, &pipe, rank, groupSize, totalLength, magic, reduceOp);
            op.Process();
            break;
        }
        case zbccl_datatype_t::ZBCCL_DATA_TYPE_BFP16: {
            ZeroBuffAllReduceKernel<bfloat16_t> op;
            op.Init(input, output, gva, &pipe, rank, groupSize, totalLength, magic, reduceOp);
            op.Process();
            break;
        }
        default:
            return;
    }
}

int32_t ZBCCLAllReduce(const void *inp, void *out, size_t numel, zbccl_datatype_t dataType,
                       aclrtStream stream, zbccl_reduce_op_t reduceOp, const CommGroupInfo &groupInfo)
{
    /* define the block dim */
    uint32_t blockDim = 16;
    uint32_t dataTypeNum = static_cast<uint32_t>(dataType);
    uint32_t reduceOpNum = static_cast<uint32_t>(reduceOp);

    // Prepare FFTS address
    uint64_t fftsAddr = groupInfo.fftsConfig;
    uint16_t rank = groupInfo.myGroupRank;
    uint16_t groupSize = groupInfo.groupSize;
    uint8_t* metaAddr = reinterpret_cast<uint8_t *>(groupInfo.myMetaGva);
    uint8_t* input = reinterpret_cast<uint8_t *>(const_cast<void *>(inp));
    uint8_t* output = reinterpret_cast<uint8_t *>(out);

    ZeroBuffAllReduce<<<blockDim, nullptr, stream>>>(input, output, metaAddr, fftsAddr, dataTypeNum, numel, rank, groupSize, reduceOpNum);

    return 0;
}
