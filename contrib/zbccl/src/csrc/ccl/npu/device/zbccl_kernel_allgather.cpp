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

extern "C" __global__ __aicore__ void allgather(GM_ADDR input, GM_ADDR output, GM_ADDR metaAddr, uint32_t elements, int dataType, 
    uint64_t ffts, uint16_t groupSize, uint16_t rank)
{
    AllGatherKernel op;
    shmemx_set_ffts_config(ffts);
    zbccl_datatype_t ZBCCL_DATA_TYPE = static_cast<zbccl_datatype_t>(dataType);
    switch (ZBCCL_DATA_TYPE){
        case zbccl_datatype_t::ZBCCL_DATA_TYPE_INT8:
            op.Process<int8_t>(input, output, metaAddr, elements, groupSize, rank);
            break;
        case zbccl_datatype_t::ZBCCL_DATA_TYPE_INT16:
            op.Process<int16_t>(input, output, metaAddr, elements, groupSize, rank);
            break;
        case zbccl_datatype_t::ZBCCL_DATA_TYPE_INT32:
            op.Process<int32_t>(input, output, metaAddr, elements, groupSize, rank);
            break;
        case zbccl_datatype_t::ZBCCL_DATA_TYPE_FP16:
            op.Process<float16_t>(input, output, metaAddr, elements, groupSize, rank);
            break;
        case zbccl_datatype_t::ZBCCL_DATA_TYPE_FP32:
            op.Process<float>(input, output, metaAddr, elements, groupSize, rank);
            break;
        case zbccl_datatype_t::ZBCCL_DATA_TYPE_INT64:
            op.Process<int64_t>(input, output, metaAddr, elements, groupSize, rank);
            break;
        case zbccl_datatype_t::ZBCCL_DATA_TYPE_UINT64:
            op.Process<uint64_t>(input, output, metaAddr, elements, groupSize, rank);
            break;
        case zbccl_datatype_t::ZBCCL_DATA_TYPE_UINT8:
            op.Process<uint8_t>(input, output, metaAddr, elements, groupSize, rank);
            break;
        case zbccl_datatype_t::ZBCCL_DATA_TYPE_UINT16:
            op.Process<uint16_t>(input, output, metaAddr, elements, groupSize, rank);
            break;
        case zbccl_datatype_t::ZBCCL_DATA_TYPE_UINT32:
            op.Process<uint32_t>(input, output, metaAddr, elements, groupSize, rank);
            break;
        case zbccl_datatype_t::ZBCCL_DATA_TYPE_FP64:
            op.Process<float64_t>(input, output, metaAddr, elements, groupSize, rank);
            break;
        case zbccl_datatype_t::ZBCCL_DATA_TYPE_BFP16:
            op.Process<bfloat16_t>(input, output, metaAddr, elements, groupSize, rank);
            break;
        default:
            break;
    }
    
}