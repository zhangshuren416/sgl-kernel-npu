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
#ifndef ZBCCL_SMA_DEVICE_POOL_H
#define ZBCCL_SMA_DEVICE_POOL_H

#include "zbccl_sma_common.h"
#include "zbccl_sma_config.h"

namespace zbccl {
namespace sma {
namespace device {

using StreamSet = class ska::flat_hash_set<c10_npu::NPUStream>;

struct DeviceBlockPool;

/**
 * @brief DeviceBlock
 */
struct DeviceBlock
{
    int32_t deviceId_{-1};              // device Id
    aclrtStream stream_{nullptr};       // allocation stream
    StreamSet stream_uses_;             // streams on which the block was used
    size_t size_{0};                    // block size in bytes
    size_t requested_size_{0};          // memory originally requested
    DeviceBlockPool *pool_{nullptr};    // owning block pool
    DeviceBlockType block_type_{BT_SMALL}; // block type in pool
    void *ptr_{0};                      // memory address
    DeviceBlock *prev_{nullptr};        // prev block if split from a larger allocation
    DeviceBlock *next_{nullptr};        // next block if split from a larger allocation
    int32_t event_count_{0};            // number of outstanding events
    bool allocated_{false};             // in-use flag
    int64_t gc_count_{0};          // get_free_blocks_call_count when DeviceBlock is inserted

    bool is_safe_{true};                 // whether this block have corresponded dataPtr
    std::shared_ptr<c10::GatheredContext> context_when_allocated_;

    DeviceBlock(int32_t device, aclrtStream stream, size_t size, DeviceBlockPool *pool, void *ptr, DeviceBlockType block_type)
        : deviceId_(device), stream_(stream), size_(size), requested_size_(0), pool_(pool), ptr_(ptr), block_type_(block_type)
    {}
    // constructor for search key
    DeviceBlock(int32_t device, aclrtStream stream, size_t size) : deviceId_(device), stream_(stream), size_(size) {}

    ~DeviceBlock() = default;

    bool isSplit() const;
    void splice(DeviceBlock *before, DeviceBlock *after);
};

/**
 * @brief Comparator function of device block in blockSets
 */
using Comparison = bool (*)(const DeviceBlock *, const DeviceBlock *);
static bool DeviceBlockCompareBySize(const DeviceBlock *a, const DeviceBlock *b);

/**
 * @brief DeviceBlockPool
 */
class DeviceBlockPool
{
public:
    std::set<DeviceBlock *, Comparison> small_blocks_;
    std::set<DeviceBlock *, Comparison> large_blocks_;

    // following params applied for privatePool(or graphPool) cases
    bool is_private_{false};
    // Number of live graphs using this pool
    int use_count{ 1 };
    // Number of unfreed npuMallocs made for this pool. When use_count and
    // npuMalloc_count drop to zero, we can delete this PrivatePool from
    // graph_pools.
    int npuMalloc_count{ 0 };

    DeviceBlockPool(bool is_private = false)
            : small_blocks_(DeviceBlockCompareBySize),
              large_blocks_(DeviceBlockCompareBySize),
              is_private_(is_private)
    {}

    void eraseBlock(DeviceBlockType block_type, DeviceBlock *block);
    void insertBlock(DeviceBlockType block_type, DeviceBlock *block);
};

/**
 * @brief DeviceAllocParams
 */
struct DeviceAllocParams {
    DeviceAllocParams(int32_t device, size_t size, aclrtStream stream, DeviceBlockPool *pool, size_t alloc_size, DeviceBlockType block_type)
            : search_key_(device, stream, size), pool_(pool), alloc_size_(alloc_size), block_type_(block_type)
    {}

    int32_t device() const;
    aclrtStream stream() const;
    size_t size() const;

    DeviceBlock search_key_;
    // since all value pass of DeviceBlockPool using ptr, we direct use pool_ instead of p_pool_
    DeviceBlockPool *pool_;
    size_t alloc_size_;
    DeviceBlock *block_{nullptr};
    DeviceBlockType block_type_;
    ZResult result_{Z_OK};
};

}  // namespace device
}  // namespace sma
}  // namespace zbccl

#endif  // ZBCCL_SMA_DEVICE_POOL_H
