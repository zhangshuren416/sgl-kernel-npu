# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
# MemFabric_Hybrid is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

# host side files
FILE(GLOB OP_SRCS
    ${ZBCCL_SRC_DIR}/csrc/ccl/zbccl_op.cpp
    ${ZBCCL_SRC_DIR}/csrc/ccl/zbccl_op_reduce_scatter.cpp
    ${ZBCCL_SRC_DIR}/csrc/ccl/zbccl_op_allreduce_zero_buff.cpp
)

# set the so name
set(OP_PLUGIN_NAME zbccl_op)

# kernel side files without workspace
ascendc_library(zbccl_kernel STATIC
    ${ZBCCL_SRC_DIR}/csrc/ccl/device/zbccl_kernel_reduce_scatter.cpp
    ${ZBCCL_SRC_DIR}/csrc/ccl/device/zbccl_kernel_allreduce_zero_buff_kernel.cpp
)

ascendc_include_directories(zbccl_kernel PRIVATE
    ${SHMEM_HOME_PATH}/shmem/include
    ${SHMEM_HOME_PATH}/memfabric_hybrid/include/smem/device
    ${ASCEND_INCLUDE_DIR}
    ${ZBCCL_SRC_DIR}/csrc/common
)

# create shared library libzccl.so
add_library(${OP_PLUGIN_NAME} SHARED ${OP_SRCS})

target_link_libraries(${OP_PLUGIN_NAME} PRIVATE
        zbccl_kernel
        torch_npu
        ascendcl
        tiling_api
        register
        platform
        ascendalog
        dl
        shmem
        mf_smem
        mf_hybm_core
)

target_link_directories(${OP_PLUGIN_NAME} PRIVATE
        ${TORCH_DIR}/lib
        ${TORCH_NPU_DIR}/lib
        ${SHMEM_HOME_PATH}/shmem/lib
        ${SHMEM_HOME_PATH}/memfabric_hybrid/lib
)

target_include_directories(${OP_PLUGIN_NAME} PRIVATE
        ${ZBCCL_SRC_DIR}/csrc/common
        ${ZBCCL_SRC_DIR}/include
        ${TORCH_DIR}/include
        ${TORCH_DIR}/include/torch/csrc/api/include
        ${TORCH_NPU_DIR}/include
        ${ASCEND_INCLUDE_DIR}/external
        ${ASCEND_INCLUDE_DIR}/experiment/platform
        ${ASCEND_INCLUDE_DIR}/experiment/runtime
        ${SHMEM_HOME_PATH}/shmem/include
        ${SHMEM_HOME_PATH}/memfabric_hybrid/include/smem/device
)

