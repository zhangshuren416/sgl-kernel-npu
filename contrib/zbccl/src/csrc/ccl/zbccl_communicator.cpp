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
#include "zbccl_communicator.h"
#include "zbccl_comm_group_meta.h"
#include "zbccl_communicator_default.h"

namespace zbccl {
namespace ccl {
ZBCCLCommPtr ZBCCLComm::gWorldZBCCLComm{nullptr};
std::map<uintptr_t, ZBCCLCommPtr> ZBCCLComm::gZBCCLCommLookupMap_;
std::mutex ZBCCLComm::gMutex;

ZResult ZBCCLComm::Create(const zbccl_ccl_options_t &options, zbccl_comm_t *comm, const ZBCCLInitStateExt &extraState)
{
    /* translate api options to inner options */
    ZBCommOptions commOptions;
    commOptions.worldSize = extraState.worldSize;
    commOptions.groupSize = options.groupSize;
    commOptions.myWorldRank = extraState.worldRankId;
    commOptions.myGroupRank = options.groupRankId;
    commOptions.gva = extraState.gvaDevice;
    commOptions.deviceId = extraState.deviceId;

    std::lock_guard<std::mutex> guard(gMutex);
    /* init group meta arranger, already prevent initialize multiple time */
    auto &groupMetaArranger = GroupMetaArranger::Instance();
    auto result = groupMetaArranger.Initialize(extraState);
    if (result != Z_OK) {
        return result;
    }

    /* set size of spaces */
    commOptions.metaSizeOfDevice = groupMetaArranger.GetSingleMetaSpaceSize();
    commOptions.metaSizeForExchangeAddress = groupMetaArranger.GetAddressExchangeSpaceSize();
    commOptions.sizeForExchangeParam = GroupMetaArranger::OPERATE_PARAM_SIZE;

    /* get current index and myMetaGva */
    result = groupMetaArranger.CurrentGroup(commOptions.groupIndex, commOptions.myMetaDataGva);
    ZBCCL_VALIDATE_RETURN(result == Z_OK, "Get meta range for group failed, probably out of range", result);
    commOptions.myParamDataGva = commOptions.myMetaDataGva + commOptions.metaSizeForExchangeAddress;

    /* create comm object */
    auto commInner = CreateInner(options.backendType, commOptions, options.isWorldGroup);
    if (commInner == nullptr || commInner->Initialize() != Z_OK) {
        return Z_CREATE_COMM_FAILED;
    }

    *comm = commInner.Get();

    /* move to next group */
    groupMetaArranger.Move2NextGroup();

    return Z_OK;
}

ZResult ZBCCLComm::Destroy(zbccl_comm_t comm, uint32_t flags)
{
    std::lock_guard<std::mutex> guard(gMutex);
    ZBCCLCommPtr tmpComm = reinterpret_cast<ZBCCLComm *>(comm);

    return ZBCCLComm::DestroyInner(tmpComm);
}

void ZBCCLComm::DestroyAll()
{
    std::lock_guard<std::mutex> guard(gMutex);
    DestroyAllInner();
}

ZBCCLComm::ZBCCLComm(const ZBCommOptions &options, bool isWorldGroup, const ZBCCLCommPtr &worldGroup)
    : isWorldGroup_(isWorldGroup), worldGroup_(worldGroup)
{
    memcpy(&metaInfo_, &options, sizeof(ZBCommOptions));
}

ZBCCLCommPtr ZBCCLComm::CreateInner(zbccl_backend_t backendType, const ZBCommOptions &options, bool isWorldGroup)
{
    ZBCCL_LOG_INFO("ZBCommOptions dump: " << options);

    /* lock is acquired by caller already */

    if (backendType == ZBCCL_ASCEND_NPU) {
        auto comm = ZMakeRef<ZBCCLCommDefault>(options, isWorldGroup, gWorldZBCCLComm);
        if (comm == nullptr) {
            ZBCCL_LOG_AND_SET_LAST_ERROR("Create zbccl communicator failed, probably out of memory");
            return nullptr;
        }

        if (isWorldGroup && gWorldZBCCLComm == nullptr) {
            /*
             * if world group and not created, then
             * 1 created new one (created previously)
             * 2 set to global one
             * 3 increase reference and return
             */
            gWorldZBCCLComm = comm.Get();
            comm->IncreaseRef();
            return comm.Get();
        } else if (isWorldGroup && gWorldZBCCLComm != nullptr) {
            /*
             * if world group already created and return nullptr,
             * return nullptr directly as its already created
             */
            ZBCCL_LOG_AND_SET_LAST_ERROR("Create zbccl communicator failed as world group already created");
            return nullptr;
        } else if (!isWorldGroup && gWorldZBCCLComm == nullptr) {
            /*
             * if not world group and world group not created,
             * here we need to create world group firstly,
             * return nullptr
             */
            ZBCCL_LOG_AND_SET_LAST_ERROR("Create zbccl communicator failed as world group not created");
            return nullptr;
        } else {
            /*
             * if not world group and world group created
             */
            gZBCCLCommLookupMap_.emplace(reinterpret_cast<uintptr_t>(comm.Get()), comm.Get());
            return comm.Get();
        }
    }
    return nullptr;
}

ZResult ZBCCLComm::DestroyInner(zbccl::ccl::ZBCCLCommPtr &comm)
{
    ZBCCL_VALIDATE_RETURN(comm == nullptr, "invalid param, ZBCCLComm is null", Z_INVALID_PARAM);

    /* lock is acquired by caller already */

    /* if it is the world one */
    if (comm->isWorldGroup_) {
        if (gZBCCLCommLookupMap_.size() != 0) {
            ZBCCL_LOG_AND_SET_LAST_ERROR("Destroy other small ZBCCLComm firstly, then destroy the world one");
            return Z_ERROR;
        }

        if (gWorldZBCCLComm != nullptr) {
            ZBCCL_LOG_INFO("Destroying the world ZBCCLComm");
            gWorldZBCCLComm->DecreaseRef();
            gWorldZBCCLComm = nullptr;
        }
        return Z_OK;
    }

    /* erase from lookup map directly */
    gZBCCLCommLookupMap_.erase(reinterpret_cast<uintptr_t>(comm.Get()));

    return Z_OK;
}

void ZBCCLComm::DestroyAllInner()
{
    /* lock is acquired by caller already */

    /* clear all other world comm*/
    gZBCCLCommLookupMap_.clear();

    /* clear world one */
    if (gWorldZBCCLComm != nullptr) {
        gWorldZBCCLComm->DecreaseRef();
        gWorldZBCCLComm = nullptr;
    }
}
}  // namespace ccl
}  // namespace zbccl