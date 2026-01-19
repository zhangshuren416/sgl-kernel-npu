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
#include "zbccl_sma.h"

namespace zbccl {
namespace sma {

SecondaryMemoryAllocator::SecondaryMemoryAllocator() {
}

void SecondaryMemoryAllocator::add_allocated_block(device::DeviceBlock *block) {
    std::lock_guard<std::mutex> lock(mutex_);
    allocated_blocks_[block->ptr_] = block;
}

device::DeviceBlock *SecondaryMemoryAllocator::get_allocated_block(void *ptr, bool remove) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = allocated_blocks_.find(ptr);
    if (it == allocated_blocks_.end()) {
        return nullptr;
    }
    device::DeviceBlock *block = it->second;
    if (remove) {
        allocated_blocks_.erase(it);
    }
    return block;
}

bool SecondaryMemoryAllocator::initialized() {
    return !device_allocator_.empty();
}

void SecondaryMemoryAllocator::cleanEvent() {
    int count = static_cast<int>(device_allocator_.size());
    for (int i = 0; i < count; i++) {
        device_allocator_[i]->releaseAndFreeEvents();
    }
}

bool SecondaryMemoryAllocator::checkBlockIsSafe(const c10::DataPtr &ptr) {
    if (!ptr.get()) {
        return true;
    }
    /*
    if (ptr.get_deleter() != &local_raw_delete) {
        return true;
    }*/
    device::DeviceBlock *block = get_allocated_block(ptr.get());
    ZBCCL_ASSERT_S(block != nullptr, "No allocated block can be found", Z_INVALID_PTR);
    return block->is_safe_;
}

void SecondaryMemoryAllocator::markAllBlockUnsafe(int device) {
    return device_allocator_[device]->markAllBlockUnsafe();
}

void SecondaryMemoryAllocator::updateBlockToSafe(const c10::DataPtr &ptr) {
    if (!ptr.get()) {
        return;
    }
    /*
    if (ptr.get_deleter() != &local_raw_delete) {
        return;
    }*/
    device::DeviceBlock *block = get_allocated_block(ptr.get());
    ZBCCL_ASSERT_S(block != nullptr, "No allocated block can be found", Z_INVALID_PTR);
    if (!block->is_safe_) {
        ZBCCL_LOG_INFO("Triggers to refresh the data of the unsafe memory block and remove the unsafe flag");
    }
    block->is_safe_ = true;
}

ZResult SecondaryMemoryAllocator::Initialize(zbccl_allocator_options *options, int32_t device_count) noexcept {
    int size = static_cast<int>(device_allocator_.size());
    if (size < device_count) {
        device_allocator_.resize(device_count);
        for (auto i = size; i < device_count; ++i) {
            device_allocator_[i] = std::make_unique<device::DeviceSMACachingAllocator>();
        }
    }
    return Z_OK;
}

ZResult SecondaryMemoryAllocator::Allocate(void **devPtr, int device, size_t size, aclrtStream stream) noexcept {
    ZBCCL_ASSERT_S(0 <= device && static_cast<size_t>(device) < device_allocator_.size(),
        "Allocator not initialized for device ", device, ": did you call init?", Z_INVALID_PARAM);
    device::DeviceBlock *block = device_allocator_[device]->malloc(device, size, stream);

    add_allocated_block(block);
    *devPtr = static_cast<void *>(block->ptr_);

    return Z_OK;
}

ZResult SecondaryMemoryAllocator::Free(void *ptr) noexcept {
    if (!ptr) {
        return Z_INVALID_PTR;
    }
    device::DeviceBlock *block = get_allocated_block(ptr, true);
    if (!block) {
        ZBCCL_LOG_WARN("invalid device pointer: " << ptr);
    }
    device_allocator_[block->deviceId_]->free(block);

    return Z_OK;
}

ZResult SecondaryMemoryAllocator::EmptyCache(bool check_error) {
    ZBCCL_LOG_DEBUG("Begin empty cache with check_error = " << check_error);
    int32_t current_device = 0;
    if (check_error) {
        ZBCCL_CHECK_S(c10_npu::GetDevice(&current_device) == ACL_SUCCESS, Z_RT_ERROR);
    } else {
        c10_npu::GetDevice(&current_device);
    }

    int count = static_cast<int>(device_allocator_.size());
    bool free_physical = true;  // actually we do not need free_physical
    for (int i = 0; i < count; i++)
        device_allocator_[i]->emptyCache(i, check_error, free_physical);
    // FIXME skip using GetUsedDevices
    // auto used_devices_list = c10_npu::GetUsedDevices();
    // for (int8_t device_idx : used_devices_list) {
    //     if (check_error) {
    //         NPU_CHECK_ERROR_MOCK(c10_npu::SetDevice(device_idx));
    //     } else {
    //         NPU_CHECK_WARN_MOCK(c10_npu::SetDevice(device_idx));
    //     }
    //     device_allocator[device_idx]->emptyCache(device_idx, check_error, free_physical);
    // }
    if (check_error) {
        ZBCCL_CHECK_S(c10_npu::MaybeSetDevice(current_device) == ACL_SUCCESS, Z_RT_ERROR);
    } else {
        c10_npu::MaybeSetDevice(current_device);
    }
    ZBCCL_LOG_DEBUG("End empty cache with check_error = " << check_error);

    return Z_OK;
}

ZResult SecondaryMemoryAllocator::RecordStream(void *ptr, c10_npu::NPUStream stream) {
    // Empty tensor's storage().data() might be a null ptr. As there is no
    // blocks associated with those tensors, it is fine to do nothing here.
    if (!ptr) {
        return Z_ERROR;
    }

    // If a tensor is not allocated by this instance, simply skip
    // This usually happens when NPU tensors are shared across processes,
    // we have implemented reference counting based sharing mechanism to
    // guarantee tensors won't be accidentally freed by one process while
    // they are still being used in another
    // if (ptr.get_deleter() != &local_raw_delete) {
    //     return;
    // }

    device::DeviceBlock *block = get_allocated_block(ptr);
    // block must not be null reaching here
    ZBCCL_ASSERT_S(block != nullptr, "No allocated block can be found");
    device_allocator_[block->deviceId_]->recordStream(block, stream);

    return Z_OK;
}

ZResult SecondaryMemoryAllocator::EraseStream(void *ptr, c10_npu::NPUStream stream)
{
    if (!ptr) {
        return Z_ERROR;
    }

    // If a tensor is not allocated by this instance, simply skip
    // This usually happens when NPU tensors are shared across processes,
    // we have implemented reference counting based sharing mechanism to
    // guarantee tensors won't be accidentally freed by one process while
    // they are still being used in another
    // if (ptr.get_deleter() != &local_raw_delete) {
    //     // TORCH_NPU_WARN_ONCE("Tensor not is not allocated by DirectMemoryAllocator, skip eraseStream.");
    //     return;
    // }

    device::DeviceBlock *block = get_allocated_block(ptr);
    if (!block) {
        ZBCCL_LOG_ERROR("invalid device pointer: " << ptr);
    }

    if (block->stream_ != c10_npu::getCurrentNPUStream(block->deviceId_).stream(false)) {
        // If the Stream applying for tensor block different from
        // the stream of submiting event wait task in HCCL synchronize()
        // method, the recordSteam can not be erased.
        // New tensor creation may use the block before HCCL op is complete.
        return Z_ERROR;
    }

    device_allocator_[block->deviceId_]->eraseStream(block, stream);

    return Z_OK;
}

}  // namespace sma
}  // namespace zbccl

extern "C" {
ZBCCL_API void *sma_malloc(size_t size, int device, aclrtStream stream) {
    void *ptr = nullptr;
    if (size == 0) {
        return ptr;
    }
    zbccl::sma::SecondaryMemoryAllocator::GetInstance()->Allocate(&ptr, device, size, stream);
    return ptr;
}

ZBCCL_API void sma_free(void *ptr, size_t size, int device, aclrtStream stream) {
    zbccl::sma::SecondaryMemoryAllocator::GetInstance()->Free(ptr);
}

ZBCCL_API void sma_init(int device_count) {
    zbccl::sma::SecondaryMemoryAllocator::GetInstance()->Initialize(nullptr, device_count);
}

ZBCCL_API void sma_empty_cache(bool check_error) {
    zbccl::sma::SecondaryMemoryAllocator::GetInstance()->EmptyCache(check_error);
}

ZBCCL_API void sma_record_stream(void *ptr, c10_npu::NPUStream stream) {
    zbccl::sma::SecondaryMemoryAllocator::GetInstance()->RecordStream(ptr, stream);
}

ZBCCL_API void sma_erase_stream(void *ptr, c10_npu::NPUStream stream) {
    zbccl::sma::SecondaryMemoryAllocator::GetInstance()->EraseStream(ptr, stream);
}

ZBCCL_API void* sma_get_base_addr(int device) {
    int device_i = 0;
    if (device < 0)
        c10_npu::GetDevice(&device_i);
    else
        device_i = device;
    return zbccl::sma::SecondaryMemoryAllocator::GetInstance()->device_allocator_[device_i]->shmem_base_addr_;
}

ZBCCL_API void sma_init_shmem(int my_rank, int n_ranks, uint64_t local_mem_size, uint64_t meta_size, const char *ip_port) {
    std::cout << "sma init: " << my_rank << " " << n_ranks << " " << local_mem_size << " " << meta_size << " " << ip_port << std::endl;
    if (shmem_init_status() != 2) {
        auto status = shmem_set_conf_store_tls(false, nullptr, 0);
        ZBCCL_ASSERT_S(status == shmem_error_code_t::SHMEM_SUCCESS, "[E]shmem shmem_set_conf_store_tls error.");
        shmem_init_attr_t *attributes;
        status = shmem_set_attr(my_rank, n_ranks, local_mem_size, ip_port, &attributes);
        ZBCCL_ASSERT_S(status == shmem_error_code_t::SHMEM_SUCCESS, "[E]shmem shmem_set_attr error.");
        status = shmem_init_attr(attributes);
        ZBCCL_ASSERT_S(status == shmem_error_code_t::SHMEM_SUCCESS, "[E]shmem shmem_init_attr error.");
    }

    void *shmem_base_addr_ = shmem_malloc(local_mem_size);
    int device = 0;
    c10_npu::GetDevice(&device);

    zbccl::sma::SecondaryMemoryAllocator::GetInstance()->device_allocator_[device]->mem_heap_inited_ = true;
    zbccl::sma::SecondaryMemoryAllocator::GetInstance()->device_allocator_[device]->mem_heap_pool_ =
            std::make_shared<zbccl::sma::heap::MemoryHeap>(shmem_base_addr_ + meta_size, local_mem_size - meta_size);
}

}