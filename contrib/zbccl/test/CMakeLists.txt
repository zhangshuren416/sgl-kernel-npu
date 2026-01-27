# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
# ZBCCL is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

set(ZBCCL_TEST_BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(ZBCCL_TEST_3RDPARTY_DIR ${ZBCCL_TEST_BASE_DIR}/3rdparty)
set(ZBCCL_TEST_3RDPARTY_OUTPUT_DIR ${ZBCCL_TEST_BASE_DIR}/output/3rdparty)

link_directories(${ZBCCL_TEST_3RDPARTY_OUTPUT_DIR}/googletest/lib)
link_directories(${ASCEND_HOME_PATH}/runtime/lib64/)

if (BUILD_ZBCCL_MODULE_UT)
    message(STATUS "ZBCCL_TEST_3RDPARTY_OUTPUT_DIR = ${ZBCCL_TEST_3RDPARTY_OUTPUT_DIR}")
    add_subdirectory(3rdparty)
    add_subdirectory(ut)
endif ()