# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
# ZBCCL is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

message(STATUS "build testcase is required")

include_directories(
        $ENV{SHMEM_HOME_PATH}/shmem/include
        ${ASCEND_HOME_PATH}/include
        ${TORCH_DIR}/include/torch/csrc/api/include
        ${TORCH_DIR}/include
        ${TORCH_DIR}/include/torch/csrc/distributed
        ${TORCH_DIR}/include/torch/csrc/utils
        ${TORCH_DIR}/include/c10/util
        ${TORCH_NPU_DIR}/include
        ${PYBIND11_DIR}/include
        ${CPYTHON_DIR}
        ${ZBCCL_BASE_DIR}
        ${ZBCCL_BASE_DIR}/third_party/ska
        ${ZBCCL_BASE_DIR}/third_party/mstx
        ${ZBCCL_SRC_DIR}/include
        ${ZBCCL_SRC_DIR}/csrc/adaptor/pytorch_npu
        ${ZBCCL_SRC_DIR}/csrc/
        ${ZBCCL_SRC_DIR}/csrc/ccl
        ${ZBCCL_SRC_DIR}/csrc/ccl/npu
        ${ZBCCL_SRC_DIR}/csrc/common
        ${ZBCCL_SRC_DIR}/csrc/dma
        ${ZBCCL_SRC_DIR}/csrc/sma
        ${ZBCCL_SRC_DIR}/csrc/under_api/cann
        ${ZBCCL_SRC_DIR}/csrc/under_api/memfabric
        ${ZBCCL_SRC_DIR}/csrc/bootstrap
        ${ZBCCL_SRC_DIR}/csrc/bootstrap/memory
        ${ZBCCL_SRC_DIR}/csrc/bootstrap/memory/memfabric
        ${ZBCCL_SRC_DIR}/csrc/bootstrap/memory/aclshmem
        ${ZBCCL_TEST_3RDPARTY_OUTPUT_DIR}/googletest/include
)

file(GLOB_RECURSE ZBCCL_TEST_CASE_SRCS
        ${CMAKE_CURRENT_SOURCE_DIR}/csrc/*
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
)

add_executable(test_zbccl ${ZBCCL_TEST_CASE_SRCS}
        $<TARGET_OBJECTS:common_objects>
        $<TARGET_OBJECTS:ccl_objects>
        $<TARGET_OBJECTS:under_api_objects>
        $<TARGET_OBJECTS:boostrap_objects>
        )

target_compile_options(test_zbccl PRIVATE -fPIC -Wno-float-equal)
target_link_options(test_zbccl PRIVATE -pie)
target_link_libraries(test_zbccl PUBLIC gtest ascendcl runtime zbccl_kernel)
#target_link_libraries(test_zbccl PUBLIC gtest zbccl_kernel acl_shared)

install(TARGETS test_zbccl
        RUNTIME DESTINATION ${ZBCCL_BASE_DIR}/output/ut
        PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE)