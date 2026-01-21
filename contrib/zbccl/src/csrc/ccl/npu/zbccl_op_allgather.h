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

ZResult AllGather(const void *sendBuff, void *recvBuff, size_t sendCount, zbccl_datatype_t dataType, aclrtStream stream, 
    ZBCommMetaInfo metaInfo)
{
    int32_t blockDim = 24;
    uint64_t fftsAddr = shmemx_get_ffts_config();
    int dataTypeInt = static_cast<int>(dataType);

    void *myParamDataGva = reinterpret_cast<void *>(metaInfo.myParamDataGva());
    ZBCCL_CHECK_S(AclrtMemcpy(metaInfo.myParamDataGva, sizeof(ZBCommMetaInfo), &metaInfo, sizeof(ZBCommMetaInfo), 
                              ACL_MEMCPY_HOST_TO_DEVICE) == Z_OK, Z_RT_ERROR);

    ACLRT_LAUNCH_KERNEL(allgather)(blockDim, stream, sendBuff, recvBuff, myParamDataGva, sendCount, dataTypeInt, fftsAddr);
    return Z_OK;
}
} // namespace zccl

}  // namespace zbccl

#endif  // ZBCCL_OP_ALLGATHER_H
