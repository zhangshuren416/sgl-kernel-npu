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
#include "dma_common.h"
#include <sstream>
#include <string>

bool OptionsManager::IsHcclZeroCopyEnable = false;
bool OptionsManager::CheckForceUncached = false;

std::string ZBCCLFormatErrorCode(int32_t errorCode)
{
    // if (c10_npu::option::OptionsManager::IsCompactErrorOutput()) {
    //     return "";
    // }
    std::ostringstream oss;
    // int deviceIndex = -1;
    // c10_npu::GetDevice(&deviceIndex);
    // auto rank_id = c10_npu::option::OptionsManager::GetRankId();
    oss << "\n[ERROR] CODE" << static_cast<int>(errorCode);

    return oss.str();
}
