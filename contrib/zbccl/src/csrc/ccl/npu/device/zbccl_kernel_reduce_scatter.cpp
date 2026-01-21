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
#include "zbccl_kernel_reduce_scatter.h"


using namespace zbccl;
using namespace zbccl::ccl;

extern "C" __global__ __aicore__ void ZeroBuffReduceScatter(
    GM_ADDR input, GM_ADDR output, GM_ADDR gva,
    uint64_t fftsAddr, uint32_t dataType, uint32_t totalLength,
    uint32_t rank, uint32_t groupSize, uint32_t reduceOp)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    uint32_t magic = 1;
    AscendC::TPipe pipe;
    zbccl_datatype_t zbcclDataType = static_cast<zbccl_datatype_t>(dataType);
    switch (zbcclDataType) {
        case zbccl_datatype_t::ZBCCL_DATA_TYPE_INT8: {
            ZeroBuffReduceScatterKernel<int8_t> op;
            op.Init(input, output, gva, &pipe, rank, groupSize, totalLength, magic, fftsAddr, reduceOp);
            op.Process();
            break;
        }
        case zbccl_datatype_t::ZBCCL_DATA_TYPE_INT16: {
            ZeroBuffReduceScatterKernel<int16_t> op;
            op.Init(input, output, gva, &pipe, rank, groupSize, totalLength, magic, fftsAddr, reduceOp);
            op.Process();
            break;
        }
        case zbccl_datatype_t::ZBCCL_DATA_TYPE_INT32: {
            ZeroBuffReduceScatterKernel<int32_t> op;
            op.Init(input, output, gva, &pipe, rank, groupSize, totalLength, magic, fftsAddr, reduceOp);
            op.Process();
            break;
        }
        case zbccl_datatype_t::ZBCCL_DATA_TYPE_FP32: {
            ZeroBuffReduceScatterKernel<float> op;
            op.Init(input, output, gva, &pipe, rank, groupSize, totalLength, magic, fftsAddr, reduceOp);
            op.Process();
            break;
        }
        case zbccl_datatype_t::ZBCCL_DATA_TYPE_FP16: {
            ZeroBuffReduceScatterKernel<float16_t> op;
            op.Init(input, output, gva, &pipe, rank, groupSize, totalLength, magic, fftsAddr, reduceOp);
            op.Process();
            break;
        }
        case zbccl_datatype_t::ZBCCL_DATA_TYPE_BFP16: {
            ZeroBuffReduceScatterKernel<bfloat16_t> op;
            op.Init(input, output, gva, &pipe, rank, groupSize, totalLength, magic, fftsAddr, reduceOp);
            op.Process();
            break;
        }
        default:
            return;
    }
}