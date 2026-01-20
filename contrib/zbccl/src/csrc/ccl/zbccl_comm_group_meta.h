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
#ifndef ZBCCL_COMM_GROUP_META_H
#define ZBCCL_COMM_GROUP_META_H

#include "zbccl_common_includes.h"

namespace zbccl {
namespace ccl {
class GroupMetaArranger
{
public:
    static GroupMetaArranger &Instance()
    {
        static GroupMetaArranger gInstance;
        return gInstance;
    }

    static constexpr uint64_t OPERATE_PARAM_SIZE = 64 * 1024L; /* 64KB */

public:
    GroupMetaArranger() = default;
    ~GroupMetaArranger() = default;

    ZResult Initialize(const ZBCCLInitStateExt &extraState) noexcept;
    void UnInitialize() noexcept;

    uint64_t GetSingleMetaSpaceSize() const noexcept;
    uint64_t GetAddressExchangeSpaceSize() const noexcept;

    ZResult CurrentGroup(uint32_t &index, uintptr_t &groupMetaGVA);
    void Move2NextGroup();

private:
    ZResult Verify() noexcept;

private:
    uintptr_t myMetaGVA_ = 0;
    uint64_t totalMetaSpaceSize_ = 0;
    uint64_t singleMetaSpaceSize_ = 0;
    uint16_t cclGroupCap_ = 0;

private:
    static std::atomic<uint32_t> gGroupIndex;
};

inline uint64_t GroupMetaArranger::GetSingleMetaSpaceSize() const noexcept
{
    return singleMetaSpaceSize_;
}

inline uint64_t GroupMetaArranger::GetAddressExchangeSpaceSize() const noexcept
{
    uint64_t tmpSingleSize = singleMetaSpaceSize_;
    /* translate to bytes */
    tmpSingleSize = tmpSingleSize * 1024;
    return tmpSingleSize - OPERATE_PARAM_SIZE;
}
}  // namespace ccl
}  // namespace zbccl

#endif  // ZBCCL_COMM_GROUP_META_H
