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
#ifndef ZBCCL_OP_REDUCESCATTER_H
#define ZBCCL_OP_REDUCESCATTER_H

#include "dl_cann_api.h"
#include "zbccl_common_includes.h"
#include "aclrtlaunch_ZeroBuffReduceScatter.h"

namespace zbccl {
namespace ccl {

int32_t ZBCCLReduceScatter(const void *inp, void *out, size_t recvNumel, zbccl_datatype_t dataType,
                           aclrtStream stream, zbccl_reduce_op_t reduceOp, ZBCommMetaInfo &metaInfo)
{
    /* define the block dim */
    uint32_t blockDim = 16;
    uint32_t dataTypeNum = static_cast<uint32_t>(dataType);
    uint32_t reduceOpNum = static_cast<uint32_t>(reduceOp);

    // Prepare FFTS address
    uint64_t fftsAddr = metaInfo.fftsConfig;
    // fixme set output empty in kernel
    AclrtMemset(out, recvNumel, 0, recvNumel);
    uint16_t rank = metaInfo.myGroupRank;
    uint16_t groupSize = metaInfo.groupSize;
    uint8_t* metaAddr = reinterpret_cast<uint8_t *>(metaInfo.myMetaDataGva);

    /* launch the kernel function via ACLRT_LAUNCH_KERNEL */
    ACLRT_LAUNCH_KERNEL(ZeroBuffReduceScatter)(blockDim, stream, inp, out, metaAddr,
                                            fftsAddr, dataTypeNum, recvNumel, rank, groupSize, reduceOpNum);

    return 0;
}

}  // namespace ccl
}  // namespace zbccl
#endif  // ZBCCL_OP_REDUCESCATTER_H