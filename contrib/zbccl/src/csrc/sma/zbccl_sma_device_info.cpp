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
#include "zbccl_sma_device_info.h"

namespace zbccl {
namespace sma {
namespace device {

void DeviceInfoObserver::recordTrace(TraceAction action, int64_t addr, size_t size, aclrtStream stream, int device) {
    if (!record_history_) {
        return;
    }
    if (device >= snapshots_.size()) {
        snapshots_.resize(device + 1);
    }

    int32_t curr_device = -1;
    c10_npu::GetDevice(&curr_device);
    if (curr_device != device){
        std::lock_guard<std::recursive_mutex> lock(mutex_);
        ZBCCL_ASSERT_S(curr_device == device, "Un-support cases on device A use device B sma pool");
    }

    auto te = TraceInfo(action, device, addr, size, stream);

    if (snapshots_[device].trace_infos_.size() < max_trace_len_) {
        snapshots_[device].trace_infos_.emplace_back(te);
    } else {
        snapshots_[device].trace_infos_[trace_next_++] = te;
        if (trace_next_ == max_trace_len_) {
            trace_next_ = 0;
        }
    }
}

void DeviceInfoObserver::takeSnapshot(const std::vector<const DeviceBlock *>& all_blocks, int device) {
    if (device >= snapshots_.size()) {
        snapshots_.resize(device + 1);
    }
    if (snapshots_[device].seg_infos_.size() > 0) {
        ZBCCL_LOG_WARN("already have snapshot record, this action is skipped, [TODO] support multiply snapshots");
        return;
    }

    int32_t curr_device = -1;
    c10_npu::GetDevice(&curr_device);
    if (curr_device != device){
        std::lock_guard<std::recursive_mutex> lock(mutex_);
        ZBCCL_LOG_WARN("Un-support cases on device A use device B sma pool");
    }

    uint64_t total_active = 0;
    for (const DeviceBlock * const head_block : all_blocks) {
        // we report one segment for each continuous range of memory
        if (head_block->prev_) {
            continue;
        }
        snapshots_[device].seg_infos_.emplace_back();
        SegmentInfo &segment_info = snapshots_[device].seg_infos_.back();
        segment_info.device_ = head_block->deviceId_;
        segment_info.address_ = reinterpret_cast<int64_t>(head_block->ptr_);
        segment_info.stream_ = head_block->stream_;
        segment_info.is_large_ = (head_block->block_type_ == BT_BIG);
        segment_info.is_private_ = head_block->pool_->is_private_;
        const DeviceBlock *block = head_block;
        while (block != nullptr) {
            segment_info.blocks_.emplace_back();
            BlockInfo &block_info = segment_info.blocks_.back();

            block_info.size_ = block->size_;
            block_info.requested_size_ = block->requested_size_;
            block_info.allocated_ = block->allocated_;
            block_info.active_ = block->allocated_ || (block->event_count_ > 0);

            segment_info.total_size_ += block_info.size_;
            if (block_info.allocated_) {
                segment_info.allocated_size_ += block_info.size_;
            }
            if (block_info.active_) {
                segment_info.active_size_ += block_info.size_;
                segment_info.requested_size_ += block_info.requested_size_;
            }
            block = block->next_;
        }
        total_active += segment_info.active_size_;
    }

    std::sort(snapshots_[device].seg_infos_.begin(), snapshots_[device].seg_infos_.end(),
              [](const SegmentInfo &a, const SegmentInfo &b) { return a.address_ < b.address_; });

    recordTrace(TraceAction::SNAPSHOT, 0, total_active, nullptr, device);
}

const SnapshotDeviceInfo& DeviceInfoObserver::dumpSnapshot(int device) {
    return snapshots_[device];
}


void DeviceInfoObserver::recordHistory(bool record_history, int64_t max_size) {
    record_history_ = record_history;
    max_trace_len_ = max_size;
}

}  // namespace device
}  // namespace sma
}  // namespace zbccl
