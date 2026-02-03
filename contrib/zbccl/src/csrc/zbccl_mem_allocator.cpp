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
#include "zbccl_mem_allocator.h"
#include "c10_npu_dma.h"
#include "zbccl_sma.h"
#include "zbccl_sma_config.h"


#ifdef __cplusplus
extern "C" {
#endif

ZBCCL_API int32_t zbccl_sma_init(zbccl_allocator_options_t *options, int32_t flags)
{
    if(!zbccl::sma::SMAConfig::use_sma_allocator()) {
        std::cout << "[ZBCCL] using dma as allocator!" << std::endl;
        dma_init_heap(options->myGva, options->size);
    } else {
        std::cout << "[ZBCCL] using sma as allocator!" << std::endl;
        sma_init_heap(options->myGva, options->size);
    }
    return zbccl::ZResultErrorCode::Z_OK;
}

ZBCCL_API void zbccl_sma_uninit(int32_t flags)
{
    // TODO: get trigger time correctly, maybe callback
    return;
}

ZBCCL_API void zbccl_pluggable_init(int32_t device_count)
{
    if(!zbccl::sma::SMAConfig::use_sma_allocator()) {
        dma_init(device_count);
    } else {
        sma_init(device_count);
    }
}

ZBCCL_API void *zbccl_pluggable_malloc(size_t size, int32_t device, aclrtStream stream)
{
    if(!zbccl::sma::SMAConfig::use_sma_allocator()) {
        return dma_malloc(size, device, stream);
    } else {
        return sma_malloc(size, device, stream);
    }
}

ZBCCL_API void zbccl_pluggable_free(void *ptr, size_t size, int32_t device, aclrtStream stream)
{
    if(!zbccl::sma::SMAConfig::use_sma_allocator()) {
        dma_free(ptr, size, device, stream);
    } else {
        sma_free(ptr, size, device, stream);
    }
}

ZBCCL_API void zbccl_pluggable_empty_cache(bool check_error)
{
    if(!zbccl::sma::SMAConfig::use_sma_allocator()) {
        dma_empty_cache(check_error);
    } else {
        sma_empty_cache(check_error);
    }
}

ZBCCL_API void zbccl_pluggable_record_stream(void *ptr, c10_npu::NPUStream stream)
{
    if(!zbccl::sma::SMAConfig::use_sma_allocator()) {
        dma_record_stream(ptr, stream);
    } else {
        sma_record_stream(ptr, stream);
    }
}

ZBCCL_API void zbccl_pluggable_erase_stream(void *ptr, c10_npu::NPUStream stream)
{
    if(!zbccl::sma::SMAConfig::use_sma_allocator()) {
        dma_erase_stream(ptr, stream);
    } else {
        sma_erase_stream(ptr, stream);
    }
}


ZBCCL_API void zbccl_pluggable_begin_allocate_to_pool(int device, c10_npu::MempoolId_t mempool_id, std::function<bool(aclrtStream)> filter) {
    if(!zbccl::sma::SMAConfig::use_sma_allocator()) {
        dma_begin_allocate_to_pool(device, mempool_id, filter);
    } else {
        sma_begin_allocate_to_pool(device, mempool_id, filter);
    }
}

ZBCCL_API void zbccl_pluggable_end_allocate_to_pool(int device, c10_npu::MempoolId_t mempool_id) {
    if(!zbccl::sma::SMAConfig::use_sma_allocator()) {
        dma_end_allocate_to_pool(device, mempool_id);
    } else {
        sma_end_allocate_to_pool(device, mempool_id);
    }
}

ZBCCL_API void zbccl_pluggable_release_pool(int device, c10_npu::MempoolId_t mempool_id) {
    if(!zbccl::sma::SMAConfig::use_sma_allocator()) {
        dma_release_pool(device, mempool_id);
    } else {
        dma_release_pool(device, mempool_id);
    }
}

#ifdef __cplusplus
}
#endif

pybind11::dict dump_snapshot() {
    if(!zbccl::sma::SMAConfig::use_sma_allocator()) {
        dma_dump_snapshot();
    } else {
        sma_dump_snapshot();
    }
}

void pybind11_allocator(pybind11::module_ &m)
{
    m.doc() = "ZBCCL Allocator Stats API";

    m.def("record_memory_history", [](std::optional<std::string> enabled, int64_t max_entries) {
        if(!zbccl::sma::SMAConfig::use_sma_allocator()) {
            dma_record_memory_history(enabled, max_entries);
        } else {
            sma_record_memory_history(enabled, max_entries);
        }
        return;
    }, "begin record memory with history");

    m.def("get_heap_stats", [](int device) {
        size_t in_used_size = 0;
        size_t total_size = 0;

        if(!zbccl::sma::SMAConfig::use_sma_allocator()) {
          dma_get_heap_stats(in_used_size, total_size, device);
        } else {
          sma_get_heap_stats(in_used_size, total_size, device);
        }

        return std::make_tuple(in_used_size, total_size);
    }, pybind11::arg("device") = -1,
    "get heap stats，return (used_size, total_size)");

    m.def("dump_snapshot", &dump_snapshot, "dump snapshot");
}
