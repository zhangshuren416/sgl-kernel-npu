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

#define USE_C10NPU_DMA

#ifdef __cplusplus
extern "C" {
#endif

int32_t zbccl_sma_init(zbccl_allocator_options *options, int32_t flags) {
    // call zbccl_init wil create zbccl_allocator_options, then init zbccl::sma
    // currently we use dma and its direct init from init_shmem
    return zbccl::ZResultErrorCode::Z_OK;
}

void zbccl_sma_uninit(int32_t flags) {
    // TODO: get trigger time correctly, maybe callback
    return;
}

void zbccl_pluggable_init(int32_t device_count) {
#ifdef USE_C10NPU_DMA
    my_init(device_count);
#endif
    return;
}

void *zbccl_pluggable_malloc(size_t size, int32_t device, aclrtStream stream) {
#ifdef USE_C10NPU_DMA
    return my_malloc(size, device, stream);
#endif
    return nullptr;
}

void zbccl_pluggable_free(void *ptr, size_t size, int32_t device, aclrtStream stream) {
#ifdef USE_C10NPU_DMA
    my_free(ptr, size, device, stream);
#endif
    return;
}

void zbccl_pluggable_empty_cache(bool check_error) {
#ifdef USE_C10NPU_DMA
    my_empty_cache(check_error);
#endif
    return;
}

void zbccl_record_stream(void *ptr, c10_npu::NPUStream stream) {
#ifdef USE_C10NPU_DMA
    my_record_stream(ptr, stream);
#endif
    return;
}

#ifdef __cplusplus
}
#endif
