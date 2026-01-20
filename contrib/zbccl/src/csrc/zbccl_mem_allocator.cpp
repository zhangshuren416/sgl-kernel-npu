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

#define USE_C10NPU_DMA  // else USE_ZBCCL_SMA

#ifdef __cplusplus
extern "C" {
#endif

ZBCCL_API int32_t zbccl_sma_init(zbccl_allocator_options_t *options, int32_t flags)
{
    // call zbccl_init wil create zbccl_allocator_options_t, then init zbccl::sma
    // currently we use dma and its direct init from init_shmem
    return zbccl::ZResultErrorCode::Z_OK;
}

ZBCCL_API void zbccl_sma_uninit(int32_t flags)
{
    // TODO: get trigger time correctly, maybe callback
    return;
}

ZBCCL_API void zbccl_pluggable_init(int32_t device_count)
{
#ifdef USE_C10NPU_DMA
    dma_init(device_count);
#else
    sma_init(device_count);
#endif
    return;
}

ZBCCL_API void *zbccl_pluggable_malloc(size_t size, int32_t device, aclrtStream stream)
{
#ifdef USE_C10NPU_DMA
    return dma_malloc(size, device, stream);
#else
    return sma_malloc(size, device, stream);
#endif
}

ZBCCL_API void zbccl_pluggable_free(void *ptr, size_t size, int32_t device, aclrtStream stream)
{
#ifdef USE_C10NPU_DMA
    dma_free(ptr, size, device, stream);
#else
    sma_free(ptr, size, device, stream);
#endif
    return;
}

ZBCCL_API void zbccl_pluggable_empty_cache(bool check_error)
{
#ifdef USE_C10NPU_DMA
    dma_empty_cache(check_error);
#else
    sma_empty_cache(check_error);
#endif
    return;
}

ZBCCL_API void zbccl_pluggable_record_stream(void *ptr, c10_npu::NPUStream stream)
{
#ifdef USE_C10NPU_DMA
    dma_record_stream(ptr, stream);
#else
    sma_record_stream(ptr, stream);
#endif
    return;
}

ZBCCL_API void zbccl_pluggable_erase_stream(void *ptr, c10_npu::NPUStream stream)
{
#ifdef USE_C10NPU_DMA
    dma_erase_stream(ptr, stream);
#else
    sma_erase_stream(ptr, stream);
#endif
    return;
}

ZBCCL_API void *zbccl_get_shmem_base_addr()
{
#ifdef USE_C10NPU_DMA
    return dma_get_base_addr();
#else
    return sma_get_base_addr();
#endif
}

void zbccl_inner_init_shmem(int my_rank, int n_ranks, uint64_t local_mem_size, uint64_t meta_size, const char *ip_port)
{
#ifdef USE_C10NPU_DMA
    dma_init_shmem(my_rank, n_ranks, local_mem_size, meta_size, ip_port);
#else
    sma_init_shmem(my_rank, n_ranks, local_mem_size, meta_size, ip_port);
#endif
    return;
}

#ifdef __cplusplus
}
#endif
