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
#ifndef ZBCCL_OP_ALLGATHER_H
#define ZBCCL_OP_ALLGATHER_H

#include "dl_cann_api.h"
#include "aclrtlaunch_allgather.h"
#include "zbccl.h"
#include "zbccl_op.h"
#include "zbccl_communicator.h"

namespace zbccl {
namespace ccl {

ZResult ZBCCL_OP_AllGather(const void *sendBuff, void *recvBuff, size_t sendCount,
                           zbccl_datatype_t dataType, aclrtStream stream, const ZBCommMetaInfo &metaInfo)
{
    int32_t blockDim = 24;
    int dataTypeInt = static_cast<int>(dataType);
    uint8_t* metaAddr = reinterpret_cast<uint8_t *>(metaInfo.myMetaDataGva);
    uint16_t rank = metaInfo.myGroupRank;
    uint16_t groupSize = metaInfo.groupSize;
    uint64_t fftsAddr = metaInfo.fftsConfig;
    uint64_t localDeviceMemSize = metaInfo.localDeviceMemSize;

    ACLRT_LAUNCH_KERNEL(allgather)(blockDim, stream, const_cast<void *>(sendBuff), recvBuff, metaAddr, sendCount,
                                   dataTypeInt, fftsAddr, groupSize, rank, localDeviceMemSize);
    return Z_OK;
}
} // namespace zccl

}  // namespace zbccl

#endif  // ZBCCL_OP_ALLGATHER_H
