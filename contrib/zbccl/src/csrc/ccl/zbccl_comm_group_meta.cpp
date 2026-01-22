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
#include "zbccl_comm_group_meta.h"

namespace zbccl {
namespace ccl {
std::atomic<uint32_t> GroupMetaArranger::gGroupIndex{0};

ZResult GroupMetaArranger::Initialize(const ZBCCLInitStateExt &extraState) noexcept
{
    if (myMetaGVA_ != 0) {
        ZBCCL_LOG_DEBUG("Group meta arranger already initialized");
        return Z_OK;
    }

    myMetaGVA_ = reinterpret_cast<uintptr_t>(extraState.myCCLMetaDeviceGva);
    totalMetaSpaceSize_ = extraState.metaSizeOfDevice;
    /* translate to bytes from KB */
    singleMetaSpaceSize_ = static_cast<uint64_t>(extraState.cclMetaSpaceSize) * 1024;
    cclGroupCap_ = extraState.cclGroupCap;

    auto result = Verify();
    if (result != Z_OK) {
        UnInitialize();
        ZBCCL_LOG_ERROR("Initialize group meta arranger failed, result: " << result);
        return result;
    }

    ZBCCL_LOG_DEBUG("Initialized group meta arranger successfully");

    return Z_OK;
}

void GroupMetaArranger::UnInitialize() noexcept
{
    myMetaGVA_ = 0;
    totalMetaSpaceSize_ = 0;
    singleMetaSpaceSize_ = 0;
    cclGroupCap_ = 0;

    ZBCCL_LOG_DEBUG("Un-initialized group meta arranger successfully");
}

ZResult GroupMetaArranger::Verify() noexcept
{
    uint64_t tmpSize = singleMetaSpaceSize_ * cclGroupCap_;
    if (totalMetaSpaceSize_ >= tmpSize) {
        return Z_OK;
    }

    ZBCCL_LOG_ERROR("Size of meta space is less than single space multiple group count cap. Total meta space size: "
                    << totalMetaSpaceSize_ << " bytes, single group space size: " << singleMetaSpaceSize_
                    << " bytes, group count cap: " << cclGroupCap_);
    return Z_ERROR;
}

ZResult GroupMetaArranger::CurrentGroup(uint32_t &index, uintptr_t &groupMetaGVA)
{
    auto currentIndex = gGroupIndex.load();
    if (currentIndex >= cclGroupCap_) {
        ZBCCL_LOG_DEBUG("Get group index failed as out of bound, current index: " << currentIndex << ", cclGroupCap_: "
                                                                                  << cclGroupCap_);
        return Z_ERROR;
    }

    index = currentIndex;
    groupMetaGVA = myMetaGVA_ + singleMetaSpaceSize_ * currentIndex;

    ZBCCL_LOG_DEBUG("Got group index: " << index << ", group meta GVA: " << groupMetaGVA);

    return Z_OK;
}

void GroupMetaArranger::Move2NextGroup()
{
    ++gGroupIndex;
}
}  // namespace ccl
}  // namespace zbccl