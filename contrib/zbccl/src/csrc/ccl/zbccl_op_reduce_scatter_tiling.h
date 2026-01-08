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
#ifndef ZBCCL_OP_REDUCE_SCATTER_TILING_H
#define ZBCCL_OP_REDUCE_SCATTER_TILING_H

#include <cstdint>
#include "zbccl_defines.h"

namespace zbccl {

struct ReduceScatterTilingData {
    uint32_t corePerRank;
    uint32_t lenPerRank;
    uint32_t curLenPerRank;
    uint32_t maxLenPerRank;
    uint32_t lastLenPerRank;

    uint32_t coreFormerLength;
    uint32_t coreFormerNum;
    uint32_t coreTailLength;
    uint32_t coreTailNum;

    uint32_t lastLoopCoreFormerLength;
    uint32_t lastLoopCoreFormerNum;
    uint32_t lastLoopCoreTailLength;
    uint32_t lastLoopCoreTailNum;

    uint32_t gvaDataOffset;
    uint32_t gvaDataLength;
    uint32_t gvaSyncLength;
    bool smallFlag;
};

void get_tiling(ReduceScatterTilingData *tilingPtr, uint32_t inputLength, ZCCLDataType dataType,
                                   uint32_t groupSize, uint32_t coreNum);
}

#endif  // ZBCCL_OP_REDUCE_SCATTER_TILING_H
