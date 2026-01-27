#
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
# ZBCCL is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
#
set(PROJECT_ZBCCL_MOCK ${ZBCCL_BASE_DIR}/output/ut)

include_directories(
        ${ZBCCL_SRC_DIR}/csrc/under_api/cann
        ${ZBCCL_SRC_DIR}/csrc/under_api/memfabric
)

set(ACL_STUB_FILE acl_stub.cpp)

add_library(acl_object OBJECT ${ACL_STUB_FILE})
target_compile_options(acl_object PRIVATE -fPIC)

add_library(acl_shared SHARED $<TARGET_OBJECTS:acl_object>)
set_target_properties(acl_shared PROPERTIES OUTPUT_NAME "ascendcl")

install(
        TARGETS acl_shared
        DESTINATION ${PROJECT_ZBCCL_MOCK}/lib64/cann/lib64
        PERMISSIONS OWNER_READ GROUP_READ
)

install(
        FILES version.info
        DESTINATION ${PROJECT_ZBCCL_MOCK}/lib64/cann/driver
        PERMISSIONS OWNER_READ GROUP_READ
)