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

#include "zbccl_comm_group_meta.h"

using namespace zbccl;
using namespace zbccl::ccl;

class TestZBCCLCommGroupMeta : public testing::Test
{
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override {}

    void TearDown() override {}
};

TEST_F(TestZBCCLCommGroupMeta, Initialization)
{
    ZBCCLInitStateExt stateExt;
    stateExt.cclMetaSpaceSize = 512;
    stateExt.cclGroupCap = 128;
    stateExt.myCCLMetaDeviceGva = reinterpret_cast<void *>(1024);
    stateExt.metaSizeOfDevice = 1024;

    auto &arranger = GroupMetaArranger::Instance();
    /* case1: meta space is not enough */
    auto result = arranger.Initialize(stateExt);
    EXPECT_TRUE(result != Z_OK);

    /* case2: meta space is just ok */
    stateExt.metaSizeOfDevice = 67108864; /* 512KB * 128 */
    result = arranger.Initialize(stateExt);
    EXPECT_TRUE(result == Z_OK);

    result = arranger.Initialize(stateExt);
    EXPECT_TRUE(result == Z_OK);

    arranger.UnInitialize();

    /* case3: meta space is larger a little bit */
    stateExt.metaSizeOfDevice = 67108864 + 1; /* 512KB * 128 */
    result = arranger.Initialize(stateExt);
    EXPECT_TRUE(result == Z_OK);
}

TEST_F(TestZBCCLCommGroupMeta, GetGroupIndex)
{
    ZBCCLInitStateExt stateExt;
    uintptr_t baseAddress = 1024;
    stateExt.cclMetaSpaceSize = 512;
    stateExt.cclGroupCap = 2;
    stateExt.myCCLMetaDeviceGva = reinterpret_cast<void *>(baseAddress);
    stateExt.metaSizeOfDevice = stateExt.cclMetaSpaceSize * 1024 * stateExt.cclGroupCap;

    auto &arranger = GroupMetaArranger::Instance();
    arranger.UnInitialize();
    auto result = arranger.Initialize(stateExt);
    EXPECT_TRUE(result == Z_OK);

    uint32_t index = 100;
    uintptr_t address = 0;
    uintptr_t addressParam = 0;
    uintptr_t addressExchange = 0;
    /* get one */
    result = arranger.CurrentGroup(index, address, addressParam, addressExchange);
    EXPECT_TRUE(result == Z_OK);
    EXPECT_TRUE(index == 0);
    EXPECT_TRUE(address == baseAddress);
    EXPECT_TRUE(addressParam == (baseAddress + sizeof(CommGroupInfo)));
    EXPECT_TRUE(addressExchange == (baseAddress + OPERATE_PARAM_SIZE));

    arranger.Move2NextGroup();

    /* get two */
    result = arranger.CurrentGroup(index, address, addressParam, addressExchange);
    EXPECT_TRUE(result == Z_OK);
    EXPECT_TRUE(index == 1);
    auto metaSpaceSizeInBytes = stateExt.cclMetaSpaceSize * 1024;
    EXPECT_TRUE(address == (baseAddress + metaSpaceSizeInBytes));
    EXPECT_TRUE(addressParam == (baseAddress + metaSpaceSizeInBytes + sizeof(CommGroupInfo)));
    EXPECT_TRUE(addressExchange == (baseAddress + metaSpaceSizeInBytes + OPERATE_PARAM_SIZE));

    arranger.Move2NextGroup();

    result = arranger.CurrentGroup(index, address, addressParam, addressExchange);
    EXPECT_TRUE(result != Z_OK);
}

TEST_F(TestZBCCLCommGroupMeta, GetSpaceSize)
{
    ZBCCLInitStateExt stateExt;
    uintptr_t baseAddress = 1024;
    stateExt.cclMetaSpaceSize = 512;
    stateExt.cclGroupCap = 2;
    stateExt.myCCLMetaDeviceGva = reinterpret_cast<void *>(baseAddress);
    stateExt.metaSizeOfDevice = stateExt.cclMetaSpaceSize * 1024 * stateExt.cclGroupCap;

    auto &arranger = GroupMetaArranger::Instance();
    arranger.UnInitialize();
    auto result = arranger.Initialize(stateExt);
    EXPECT_TRUE(result == Z_OK);

    EXPECT_TRUE(arranger.GetAddressExchangeSpaceSize() ==
                (stateExt.cclMetaSpaceSize * 1024 - OPERATE_PARAM_SIZE));
    EXPECT_TRUE(arranger.GetSingleMetaSpaceSize() == stateExt.cclMetaSpaceSize * 1024);
    EXPECT_TRUE(arranger.GetParamSpaceSize() == (OPERATE_PARAM_SIZE - sizeof(CommGroupInfo)));
    EXPECT_TRUE(arranger.GetCommGroupInfoSpaceSize() == sizeof(CommGroupInfo));
}
