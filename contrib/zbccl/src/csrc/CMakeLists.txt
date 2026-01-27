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
        ${ASCEND_HOME_PATH}/include/experiment/runtime/runtime/
        ${ASCEND_HOME_PATH}/include/experiment/msprof/
        ${TORCH_DIR}/include/torch/csrc/api/include
        ${TORCH_DIR}/include
        ${TORCH_DIR}/include/torch/csrc/distributed
        ${TORCH_DIR}/include/torch/csrc/utils
        ${TORCH_DIR}/include/c10/util
        ${TORCH_NPU_DIR}/include
        ${PYBIND11_DIR}/include
        ${CPYTHON_DIR}/
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

add_subdirectory(ccl/npu/device/)

# we seperated those files for ut with different modules

if (NOT DEFINED DISABLE_ADAPTOR_COMPILE)
        # PyTorch adaptor
        file(GLOB_RECURSE PYTORCH_ADAPTOR_SRCS
                ${ZBCCL_SRC_DIR}/csrc/adaptor/pytorch_npu/*.cpp)
        add_library(pytorch_adaptor_objects OBJECT ${PYTORCH_ADAPTOR_SRCS})
        target_compile_options(pytorch_adaptor_objects PRIVATE -fPIC)
endif()

# boostrap objects
file(GLOB_RECURSE BOOSTRAP_SRCS
       ${ZBCCL_SRC_DIR}/csrc/bootstrap/*.cpp
       ${ZBCCL_SRC_DIR}/csrc/bootstrap/memory/*.cpp
       ${ZBCCL_SRC_DIR}/csrc/bootstrap/memory/memfabric/*.cpp)
add_library(boostrap_objects OBJECT ${BOOSTRAP_SRCS})
target_compile_options(boostrap_objects PRIVATE -fPIC)

# ccl objects
file(GLOB CCL_SRCS
        ${ZBCCL_SRC_DIR}/csrc/ccl/*.cpp
        ${ZBCCL_SRC_DIR}/csrc/ccl/npu/*.cpp)
add_library(ccl_objects OBJECT ${CCL_SRCS})
target_link_libraries(ccl_objects PRIVATE zbccl_kernel) # link not work on object, only for compile pass
target_compile_options(ccl_objects PRIVATE -fPIC)

# common objects
file(GLOB_RECURSE COMMON_SRCS
        ${ZBCCL_SRC_DIR}/csrc/common/*.cpp)
add_library(common_objects OBJECT ${COMMON_SRCS})
target_compile_options(common_objects PRIVATE -fPIC)

if(NOT DEFINED DISABLE_ALLOCATOR_COMPILE)
        # memory allocator objects
        file(GLOB_RECURSE ALLOCATOR_SRCS
                ${ZBCCL_SRC_DIR}/csrc/dma/*.cpp
                ${ZBCCL_SRC_DIR}/csrc/sma/*.cpp)
        add_library(allocator_objects OBJECT ${ALLOCATOR_SRCS})
        target_compile_options(allocator_objects PRIVATE -fPIC)
endif()

# under api objects
file(GLOB_RECURSE UNDER_API_SRCS
        ${ZBCCL_SRC_DIR}/csrc/under_api/cann/*.cpp
        ${ZBCCL_SRC_DIR}/csrc/under_api/memfabric/*.cpp)
add_library(under_api_objects OBJECT ${UNDER_API_SRCS})
target_compile_options(under_api_objects PRIVATE -fPIC)

# zbccl objects
file(GLOB ZBCCL_SRCS
        ${ZBCCL_SRC_DIR}/csrc/*.cpp)
list(FILTER ZBCCL_SRCS EXCLUDE REGEX "zbccl_pybind\\.cpp")
add_library(zbccl_objects OBJECT ${ZBCCL_SRCS})
target_compile_options(zbccl_objects PRIVATE -fPIC)

# static lib
add_library(zbccl_static STATIC $<TARGET_OBJECTS:boostrap_objects>
                                $<TARGET_OBJECTS:ccl_objects>
                                $<TARGET_OBJECTS:common_objects>
                                $<TARGET_OBJECTS:under_api_objects>
                                $<TARGET_OBJECTS:zbccl_objects>
                                )
target_link_libraries(zbccl_static PRIVATE zbccl_kernel)
if(NOT DEFINED DISABLE_ALLOCATOR_COMPILE)
        target_sources(zbccl_static PRIVATE $<TARGET_OBJECTS:allocator_objects>)
endif()
if(NOT DEFINED DISABLE_ADAPTOR_COMPILE)
        target_sources(zbccl_static PRIVATE $<TARGET_OBJECTS:pytorch_adaptor_objects>)
endif()
set_target_properties(zbccl_static PROPERTIES OUTPUT_NAME "zbccl_core")

# shared lib
add_library(zbccl_shared SHARED $<TARGET_OBJECTS:boostrap_objects>
                                $<TARGET_OBJECTS:ccl_objects>
                                $<TARGET_OBJECTS:common_objects>
                                $<TARGET_OBJECTS:under_api_objects>
                                $<TARGET_OBJECTS:zbccl_objects>
                                )
target_link_libraries(zbccl_shared PRIVATE zbccl_kernel)
if(NOT DEFINED DISABLE_ALLOCATOR_COMPILE)
        target_sources(zbccl_shared PRIVATE $<TARGET_OBJECTS:allocator_objects>)
endif()

if(NOT DEFINED DISABLE_ADAPTOR_COMPILE)
        target_sources(zbccl_shared PRIVATE $<TARGET_OBJECTS:pytorch_adaptor_objects>)
endif()
set_target_properties(zbccl_shared PROPERTIES OUTPUT_NAME "zbccl_core")
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