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
#include <gtest/gtest.h>

#include "zbccl_communicator.h"

using namespace zbccl;
using namespace zbccl::ccl;

class TestZBCCLCommunicator : public testing::Test
{
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {}

    void TearDown() override {}
};

TEST_F(TestZBCCLCommunicator, SizeOf)
{
    ZBCCL_LOG_DEBUG("size of ZBCommOptions" << sizeof(ZBCommOptions));
}

TEST_F(TestZBCCLCommunicator, CommunicatorCreate)
{
    ZBCCL_LOG_DEBUG("size of ZBCommOptions" << sizeof(ZBCommOptions));
}