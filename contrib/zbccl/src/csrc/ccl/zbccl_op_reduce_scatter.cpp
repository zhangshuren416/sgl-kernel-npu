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

#include "aclrtlaunch_ShmemReduceScatter.h"
#include "zbccl_op_reduce_scatter.h"
#include "zbccl_op_reduce_scatter_tiling.h"

namespace zbccl {

constexpr uint32_t BLOCK_NUM_SMALL_DATA = 8;
constexpr uint32_t BLOCK_NUM_LARGE_DATA = 16;
constexpr uint32_t BIG_DATA_THRESHOLD = 2 * 1024 * 1024;
constexpr uint32_t GVA_BUFF_MAX_SIZE = 100 * 1024 * 1024;
constexpr uint32_t SYNC_FLAG_INTERVAL = 16;

ZBCCL_API int ZcclReduceScatter(uint8_t *inp, uint8_t *out,
    size_t inpNumel, ZCCLDataType dataType, int teamId, aclrtStream stream, uint32_t reduceOp)
{
    /* define the block dim */
    uint32_t blockDim = 0;

    // get team info
    uint32_t rank = shmem_team_my_pe(teamId);
    uint32_t rankSize = shmem_team_n_pes(teamId);

    size_t typeSize = GetSizeFromTypeEnum(dataType);
    if (inpNumel * typeSize < BIG_DATA_THRESHOLD) {
        blockDim = BLOCK_NUM_SMALL_DATA;
    } else {
        blockDim = BLOCK_NUM_LARGE_DATA;
    }
    uint32_t dataTypeNum = static_cast<uint32_t>(dataType);

    size_t tilingSize = sizeof(ReduceScatterTilingData);
    void* tilingPtr = nullptr;
    void* tilingDevicePtr = nullptr;
    CHECK_ACL(aclrtMallocHost(&tilingPtr, tilingSize));
    CHECK_ACL(aclrtMalloc(&tilingDevicePtr, tilingSize, ACL_MEM_MALLOC_HUGE_FIRST));
    get_tiling(reinterpret_cast<ReduceScatterTilingData *>(tilingPtr), inpNumel, dataType, rankSize, blockDim);
    CHECK_ACL(aclrtMemcpy(tilingDevicePtr, tilingSize, tilingPtr, tilingSize, ACL_MEMCPY_HOST_TO_DEVICE));

    // Prepare FFTS address
    uint64_t fftsAddr = shmemx_get_ffts_config();
    // allocate gva buffer
    size_t gvaSize = blockDim * SYNC_FLAG_INTERVAL * sizeof(int32_t) + GVA_BUFF_MAX_SIZE;
    void *ptr = shmem_malloc(gvaSize);
    aclrtMemset(ptr, gvaSize, 0, gvaSize);
    // set output empty
    size_t outputSize = inpNumel / rankSize;
    aclrtMemset(out, outputSize, 0, outputSize);

    /* launch the kernel function via ACLRT_LAUNCH_KERNEL */
    ACLRT_LAUNCH_KERNEL(ShmemReduceScatter)(blockDim, stream, inp, out, (uint8_t *)ptr,
                                            fftsAddr, dataTypeNum, inpNumel, teamId, reduceOp, tilingDevicePtr);
    shmem_free(ptr);
    return 0;
}

}
