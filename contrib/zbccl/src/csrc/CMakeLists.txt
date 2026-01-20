# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
# MemFabric_Hybrid is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

message(STATUS "ACLSHMEM PATH = $ENV{SHMEM_HOME_PATH}")

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
)

file(GLOB_RECURSE ZBCCL_SRC
        ${ZBCCL_SRC_DIR}/csrc/adaptor/pytorch_npu/*.cpp
        ${ZBCCL_SRC_DIR}/csrc/*.cpp
        ${ZBCCL_SRC_DIR}/csrc/ccl/*.cpp
        ${ZBCCL_SRC_DIR}/csrc/ccl/npu/*.cpp
        ${ZBCCL_SRC_DIR}/csrc/common/*.cpp
        ${ZBCCL_SRC_DIR}/csrc/sma/*.cpp
        ${ZBCCL_SRC_DIR}/csrc/under_api/cann/*.cpp
        ${ZBCCL_SRC_DIR}/csrc/under_api/memfabric/*.cpp
        ${ZBCCL_SRC_DIR}/csrc/bootstrap/*.cpp
        ${ZBCCL_SRC_DIR}/csrc/bootstrap/memory/*.cpp
        ${ZBCCL_SRC_DIR}/csrc/bootstrap/memory/memfabric/*.cpp
)

## build the static and shared library
add_library(zbccl_objects OBJECT ${ZBCCL_SRC})

target_link_libraries(zbccl_objects PUBLIC zbccl)
target_compile_options(zbccl_objects PRIVATE -fPIC)

add_library(zbccl_static STATIC $<TARGET_OBJECTS:zbccl_objects>)
set_target_properties(zbccl_static PROPERTIES OUTPUT_NAME "zbccl")

add_library(zbccl_shared SHARED $<TARGET_OBJECTS:zbccl_objects>)
set_target_properties(zbccl_shared PROPERTIES OUTPUT_NAME "zbccl")
target_link_libraries(zbccl_shared PUBLIC pthread)

#install
file(GLOB HEADER_FILES ${ZBCCL_BASE_DIR}/include/*.h)

#install head files
install(FILES ${HEADER_FILES} DESTINATION ${TARGET_INSTALL_DIR}/zbccl/include)

# install library
install(TARGETS zbccl_shared
        LIBRARY DESTINATION ${TARGET_INSTALL_DIR}/zbccl/lib64
        PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
)