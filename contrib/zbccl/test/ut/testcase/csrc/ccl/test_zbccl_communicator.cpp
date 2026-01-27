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
#include "zbccl_comm_group_meta.h"

using namespace zbccl;
using namespace zbccl::ccl;

class TestZBCCLCommunicator : public testing::Test
{
public:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetUp() override
    {
        /* for group meta */
        uintptr_t baseAddress = 1024;
        stateExt_.cclMetaSpaceSize = 512;
        stateExt_.cclGroupCap = 4;
        stateExt_.myCCLMetaDeviceGva = reinterpret_cast<void *>(baseAddress);
        stateExt_.metaSizeOfDevice = stateExt_.cclMetaSpaceSize * 1024 * stateExt_.cclGroupCap;
        /* for comm create */
        stateExt_.worldSize = 1;
        stateExt_.worldRankId = 0;
        stateExt_.gvaDevice = reinterpret_cast<void *>(1024);
        stateExt_.deviceId = 0;

        GroupMetaArranger::Instance().UnInitialize();
        GroupMetaArranger::Instance().Initialize(stateExt_);
    }

    void TearDown() override {}

    ZBCCLInitStateExt stateExt_;
};

TEST_F(TestZBCCLCommunicator, CommunicatorCreate)
{
    ZBCCL_LOG_DEBUG("size of CommGroupOptions: " << sizeof(CommGroupOptions));

    Communicator::DestroyAll();

    zbccl_comm_options_t commOptApi;

    /* case1: name is empty */
    static std::string name;
    commOptApi.name = const_cast<char *>(name.c_str());
    zbccl_comm_t communicator = nullptr;
    auto result = Communicator::Create(commOptApi, &communicator, stateExt_);
    EXPECT_TRUE(result != Z_OK);

    /* case2: name is nullptr */
    commOptApi.name = nullptr;
    communicator = nullptr;
    result = Communicator::Create(commOptApi, &communicator, stateExt_);
    EXPECT_TRUE(result != Z_OK);

    /* case3: create non-world group firstly */
    name = "moeep";
    commOptApi.name = const_cast<char *>(name.c_str());
    commOptApi.backendType = ZBCCL_BACK_BUTT;
    /* not the world one */
    commOptApi.isWorldGroup = false;
    commOptApi.groupSize = 1;
    commOptApi.groupRankId = 0;
    result = Communicator::Create(commOptApi, &communicator, stateExt_);
    EXPECT_TRUE(result != Z_OK);

    /* case3: create world group firstly */
    name = "moeep";
    commOptApi.name = const_cast<char *>(name.c_str());
    commOptApi.backendType = ZBCCL_BACK_BUTT;
    commOptApi.isWorldGroup = true;

    result = Communicator::Create(commOptApi, &communicator, stateExt_);
    EXPECT_TRUE(result == Z_OK);
    EXPECT_TRUE(Communicator::Count() == 1);

    zbccl_comm_t outComm = nullptr;
    result = Communicator::Lookup("moeep", &outComm);
    EXPECT_TRUE(result == Z_OK);
    EXPECT_TRUE(outComm != nullptr);

    zbccl_comm_t communicatorTp1 = nullptr;
    commOptApi.isWorldGroup = false;
    name = "tp1";
    commOptApi.name = const_cast<char *>(name.c_str());
    result = Communicator::Create(commOptApi, &communicatorTp1, stateExt_);
    EXPECT_TRUE(result == Z_OK);
    EXPECT_TRUE(communicatorTp1 != nullptr);
    EXPECT_TRUE(Communicator::Count() == 2);

    result = Communicator::Destroy(communicator, 0);
    EXPECT_TRUE(result != Z_OK);
    EXPECT_TRUE(Communicator::Count() == 2);

    result = Communicator::Destroy(communicatorTp1, 0);
    EXPECT_TRUE(result == Z_OK);
    EXPECT_TRUE(Communicator::Count() == 1);

    result = Communicator::Destroy(communicator, 0);
    EXPECT_TRUE(result == Z_OK);
    EXPECT_TRUE(Communicator::Count() == 0);

    EXPECT_TRUE(GroupMetaArranger::Instance().Initialized() == true);

    Communicator::DestroyAll();

    EXPECT_TRUE(GroupMetaArranger::Instance().Initialized() == false);
}