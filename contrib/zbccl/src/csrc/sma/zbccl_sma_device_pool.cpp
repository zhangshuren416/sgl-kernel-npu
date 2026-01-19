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
#include "zbccl_sma_device_pool.h"

namespace zbccl {
namespace sma {
namespace device {

static bool DeviceBlockCompareBySize(const DeviceBlock *a, const DeviceBlock *b) {
    if (a->stream_ != b->stream_) {
        return (uintptr_t)a->stream_ < (uintptr_t)b->stream_;
    }

    if (a->size_ != b->size_) {
        return a->size_ < b->size_;
    }

    return a->ptr_ < b->ptr_;
}

// DeviceBlock
bool DeviceBlock::isSplit() const
{
    return (prev_ != nullptr) || (next_ != nullptr);
}

void DeviceBlock::splice(DeviceBlock *before, DeviceBlock *after)
{
    if (before) {
        ZBCCL_CHECK_S(before->next_ == after, "block split check failed :", Z_INVALID_PTR);
        before->next_ = this;
    }
    prev_ = before;
    if (after) {
        ZBCCL_ASSERT_S(after->prev_ == before, "block split check failed :", Z_INVALID_PTR);
        after->prev_ = this;
    }
    next_ = after;
}

// DeviceBlockPool
void DeviceBlockPool::eraseBlock(DeviceBlockType block_type, DeviceBlock *block) {
    if (block_type == BT_SMALL)
        small_blocks_.erase(block);
    else
        large_blocks_.erase(block);
}

void DeviceBlockPool::insertBlock(DeviceBlockType block_type, DeviceBlock *block) {
    if (block_type == BT_SMALL)
        small_blocks_.insert(block);
    else
        large_blocks_.insert(block);
}

// DeviceAllocParams
inline int32_t DeviceAllocParams::device() const
{
    return search_key_.deviceId_;
}

inline aclrtStream DeviceAllocParams::stream() const
{
    return search_key_.stream_;
}

inline size_t DeviceAllocParams::size() const
{
    return search_key_.size_;
}

}  // namespace device
}  // namespace sma
}  // namespace zbccl
