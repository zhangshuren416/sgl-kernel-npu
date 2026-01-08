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

#include "zbccl_op_reduce_scatter_tiling.h"
#include "acl/acl.h"
#include "zbccl_defines.h"
#include "zbccl_functions.h"
#include "tiling/platform/platform_ascendc.h"
#include "shmem_api.h"

namespace zbccl {

constexpr uint32_t BIG_DATA_THRESHOLD = 2 * 1024 * 1024;
constexpr uint32_t GVA_BUFF_MAX_SIZE = 100 * 1024 * 1024;
constexpr uint32_t SYNC_FLAG_INTERVAL = 16;
void get_tiling(ReduceScatterTilingData *tilingPtr, uint32_t inputLength, ZCCLDataType dataType, uint32_t groupSize,
                uint32_t coreNum)
{
    auto ascendc_platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    int32_t max_aiv_core = static_cast<int32_t>(ascendc_platform->GetCoreNumAiv());

    size_t dataTypeSize = GetSizeFromTypeEnum(dataType);
    bool isSmall;
    uint32_t coreGroupNum = 0;
    uint32_t lenPerRank = 0;
    uint32_t curLenPerRank = 0;
    uint32_t maxLenPerRank = 0;
    uint32_t lastLenPerRank = 0;

    uint32_t coreFormerLength = 0;
    uint32_t coreFormerNum = 0;
    uint32_t coreTailLength = 0;
    uint32_t coreTailNum = 0;

    uint32_t lastLoopCoreFormerLength = 0;
    uint32_t lastLoopCoreFormerNum = 0;
    uint32_t lastLoopCoreTailLength = 0;
    uint32_t lastLoopCoreTailNum = 0;

    if (inputLength < BIG_DATA_THRESHOLD) {
        isSmall = true;
        coreGroupNum = coreNum;
        lenPerRank = inputLength / groupSize;
        curLenPerRank = lenPerRank;
    } else {
        isSmall = false;
        coreGroupNum = coreNum / 2;
        lenPerRank = inputLength / groupSize;
        uint32_t maxLenPerLoop = GVA_BUFF_MAX_SIZE / dataTypeSize;
        if (inputLength <= maxLenPerLoop) {
            maxLenPerRank = inputLength / groupSize;
            lastLenPerRank = inputLength / groupSize;
        } else {
            maxLenPerRank = maxLenPerLoop / groupSize;
            lastLenPerRank = (inputLength % maxLenPerLoop == 0) ? maxLenPerRank : inputLength % maxLenPerLoop;
        }
    }

    uint32_t corePerRank = coreGroupNum / groupSize;
    uint32_t curLenPerRankAlign = 0;
    if (curLenPerRank > 0) {
        curLenPerRankAlign = (curLenPerRank + corePerRank - 1) / corePerRank * corePerRank;
        coreFormerLength = curLenPerRankAlign / corePerRank;
        coreTailLength = curLenPerRank / corePerRank;
        coreFormerNum = curLenPerRank % corePerRank;
        coreTailNum = corePerRank - coreFormerNum;
    } else {
        if (maxLenPerRank > 0) {
            curLenPerRankAlign = (maxLenPerRank + corePerRank - 1) / corePerRank * corePerRank;
            coreFormerLength = curLenPerRankAlign / corePerRank;
            coreTailLength = maxLenPerRank / corePerRank;
            coreFormerNum = maxLenPerRank % corePerRank;
            coreTailNum = corePerRank - coreFormerNum;
        }
        if (lastLenPerRank > 0) {
            curLenPerRankAlign = (lastLenPerRank + corePerRank - 1) / corePerRank * corePerRank;
            lastLoopCoreFormerLength = curLenPerRankAlign / corePerRank;
            lastLoopCoreTailLength = lastLenPerRank / corePerRank;
            lastLoopCoreFormerNum = lastLenPerRank % corePerRank;
            lastLoopCoreTailNum = corePerRank - lastLoopCoreFormerNum;
        }
        
    }

    tilingPtr->corePerRank = corePerRank;
    tilingPtr->lenPerRank = lenPerRank;
    tilingPtr->curLenPerRank = curLenPerRank;
    tilingPtr->maxLenPerRank = maxLenPerRank;
    tilingPtr->lastLenPerRank = lastLenPerRank;

    tilingPtr->coreFormerLength = coreFormerLength;
    tilingPtr->coreFormerNum = coreFormerNum;
    tilingPtr->coreTailLength = coreTailLength;
    tilingPtr->coreTailNum = coreTailNum;

    tilingPtr->lastLoopCoreFormerLength = lastLoopCoreFormerLength;
    tilingPtr->lastLoopCoreFormerNum = lastLoopCoreFormerNum;
    tilingPtr->lastLoopCoreTailLength = lastLoopCoreTailLength;
    tilingPtr->lastLoopCoreTailNum = lastLoopCoreTailNum;

    tilingPtr->smallFlag = isSmall;
    tilingPtr->gvaDataOffset = coreNum * SYNC_FLAG_INTERVAL;
    tilingPtr->gvaDataLength = GVA_BUFF_MAX_SIZE / dataTypeSize;
    tilingPtr->gvaSyncLength = coreNum * SYNC_FLAG_INTERVAL;

}
}
