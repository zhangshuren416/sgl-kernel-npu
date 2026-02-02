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
#include "zbccl_comm_group_id.h"
#include "zbccl_bootstrap_default.h"

namespace zbccl {
namespace ccl {
ZResult AutoReleaseGroupId::AcquireUniqueGroupId(uint32_t maxGroupCount, uint32_t &uniqueId)
{
    auto bootstrap = bootstrap::Bootstrap::Get();
    ZBCCL_ASSERT_RETURN(bootstrap != nullptr, Z_NOT_BOOTSTRAPPED);

    return bootstrap->AcquireCommGroupId(maxGroupCount, uniqueId);
}

ZResult AutoReleaseGroupId::ReleaseUniqueGroupId(uint32_t uniqueId)
{
    auto bootstrap = bootstrap::Bootstrap::Get();
    ZBCCL_ASSERT_RETURN(bootstrap != nullptr, Z_NOT_BOOTSTRAPPED);

    return bootstrap->ReleaseCommGroupId(uniqueId);
}

ZResult AutoReleaseGroupId::AllGatherExchangeInfo(const std::string &groupName, const CommGroupExchangeInfo &myInfo,
                                                  std::vector<CommGroupExchangeInfo> &allInfo, uint32_t groupSize)
{
    auto bootstrap = bootstrap::Bootstrap::Get();
    ZBCCL_ASSERT_RETURN(bootstrap != nullptr, Z_NOT_BOOTSTRAPPED);

    ZBCCL_ASSERT_RETURN(allInfo.capacity() >= groupSize, Z_INVALID_PARAM);

    auto sendBuf = static_cast<char *>(static_cast<void *>(const_cast<CommGroupExchangeInfo *>(&myInfo)));
    auto recvBuf = static_cast<char *>(static_cast<void *>(allInfo.data()));
    return bootstrap->SubGroupAllGather(groupName, groupSize, myInfo.myGroupRankId, sendBuf,
                                        sizeof(CommGroupExchangeInfo), recvBuf,
                                        sizeof(CommGroupExchangeInfo) * groupSize);
}

AutoReleaseGroupId::~AutoReleaseGroupId()
{
    if (uniqueGroupId_ == UINT16_MAX) {
        return;
    }

    Release();
}

ZResult AutoReleaseGroupId::Acquire()
{
    ZResult result = Z_OK;
    std::lock_guard<std::mutex> guard(mutex_);
    uint32_t tmpGroupId = UINT16_MAX;

    ZBCCL_LOG_DEBUG(*this);

    CommGroupExchangeInfo exchangeInfo;
    /* only rank 0 need to acquire id by bootstrap */
    if (rankId_ == 0) {
        ZBCCL_LOG_DEBUG("Try to acquire id by bootstrap for rank " << rankId_);
        result = AcquireUniqueGroupId(maxGroupCount_, tmpGroupId);
        if (result != Z_OK) {
            ZBCCL_LOG_ERROR("Acquire unique id with bootstrap failed, result: " << result);
            return result;
        }

        exchangeInfo.groupId = tmpGroupId;
    }

    /* set exchange info */
    exchangeInfo.myWorldRankId = workRankId_;
    exchangeInfo.myGroupRankId = rankId_;

    /* clean */
    gatheredGroupInfo_.clear();
    gatheredGroupInfo_.reserve(rankSize_);

    ZBCCL_LOG_DEBUG("Exchange id by bootstrap for rank " << rankId_);
    /* do all gather to exchange id and rank info */
    result = AllGatherExchangeInfo(groupName_, exchangeInfo, gatheredGroupInfo_, rankSize_);
    if (result != Z_OK) {
        ZBCCL_LOG_ERROR("Exchange info with bootstrap failed, result: " << result);
        return result;
    } else {
        tmpGroupId = gatheredGroupInfo_[0].groupId;
    }

    uniqueGroupId_ = static_cast<uint16_t>(tmpGroupId);
    ZBCCL_LOG_DEBUG("Acquired unique id " << uniqueGroupId_);

    return Z_OK;
}

void AutoReleaseGroupId::Release()
{
    std::lock_guard<std::mutex> guard(mutex_);
    ZBCCL_LOG_DEBUG("Try to release uniqueId " << uniqueGroupId_);
    ReleaseUniqueGroupId(uniqueGroupId_);
    uniqueGroupId_ = -1;
}

void AutoReleaseGroupId::MoveIdAndGatheredInfo(AutoReleaseGroupId &other) noexcept
{
    std::lock_guard<std::mutex> guard(mutex_);
    if (uniqueGroupId_ != UINT16_MAX) {
        ZBCCL_LOG_DEBUG("Try to release uniqueId " << uniqueGroupId_);
        ReleaseUniqueGroupId(uniqueGroupId_);
    }

    ZBCCL_LOG_DEBUG("Move id, other.id " << other.uniqueGroupId_ << ", myId: " << uniqueGroupId_);
    uniqueGroupId_ = other.uniqueGroupId_;
    other.uniqueGroupId_ = UINT16_MAX;

    gatheredGroupInfo_.swap(other.gatheredGroupInfo_);
}

}  // namespace ccl
}  // namespace zbccl