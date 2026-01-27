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
#include "zbccl_npu_communicator_default.h"

namespace zbccl {
namespace ccl {
CommunicatorPtr Communicator::gWorldZBCCLComm{nullptr};
std::map<uintptr_t, CommunicatorPtr> Communicator::gCommLookupMap_;
std::map<std::string, CommunicatorPtr> Communicator::gCommLookupMapByName_;
std::mutex Communicator::gMutex;

ZResult Communicator::Create(const zbccl_comm_options_t &options, zbccl_comm_t *comm,
                             const ZBCCLInitStateExt &extraState)
{
    /* translate api options to inner options */
    CommGroupOptions commOptions;
    ZBCCL_ASSERT_RETURN(options.name != nullptr, Z_INVALID_PARAM);
    ZBCCL_ASSERT_RETURN(strlen(options.name) != 0, Z_INVALID_PARAM);
    commOptions.name = std::string(options.name);
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
    commOptions.metaSize = groupMetaArranger.GetSingleMetaSpaceSize();
    commOptions.sizeForCommGroupInfo = groupMetaArranger.GetCommGroupInfoSpaceSize();
    commOptions.sizeForParam = groupMetaArranger.GetParamSpaceSize();
    commOptions.sizeForExchangeAddress = groupMetaArranger.GetAddressExchangeSpaceSize();
    commOptions.localDeviceMemSize = ZBCCLInitState::Instance().ext_.localDeviceMemSize;

    /* get current index and myMetaGva */
    result = groupMetaArranger.CurrentGroup(commOptions.groupIndex, commOptions.myMetaGva, commOptions.myParamDataGva,
                                            commOptions.myAddressExchangeGva);
    ZBCCL_VALIDATE_RETURN(result == Z_OK, "Get meta range for group failed, probably out of range", result);

    /* create comm object */
    auto commInner = CreateInner(options.backendType, commOptions, options.isWorldGroup);
    if (commInner == nullptr || commInner->Initialize() != Z_OK) {
        return Z_CREATE_COMM_FAILED;
    }

    *comm = commInner.Get();

    ZBCCL_LOG_DEBUG("Created communicator successfully, name: " << commInner->Name() << ", ptr: " << commInner.Get());

    /* move to next group */
    groupMetaArranger.Move2NextGroup();

    return Z_OK;
}

ZResult Communicator::Destroy(zbccl_comm_t comm, uint32_t flags)
{
    std::lock_guard<std::mutex> guard(gMutex);
    CommunicatorPtr tmpComm = reinterpret_cast<Communicator *>(comm);

    ZBCCL_LOG_DEBUG("Try to destroy communicator, input ptr " << comm << ", converted ptr: " << tmpComm.Get());

    return Communicator::DestroyInner(tmpComm);
}

void Communicator::DestroyAll()
{
    std::lock_guard<std::mutex> guard(gMutex);
    DestroyAllInner();
}

ZResult Communicator::Lookup(const std::string &name, zbccl_comm_t *comm)
{
    std::lock_guard<std::mutex> guard(gMutex);

    CommunicatorPtr tmpComm;
    auto result = LookupInner(name, tmpComm);
    if (result != Z_OK) {
        return result;
    }

    *comm = tmpComm.Get();
    return Z_OK;
}

uint32_t Communicator::Count()
{
    std::lock_guard<std::mutex> guard(gMutex);
    return gCommLookupMapByName_.size();
}

CommunicatorPtr Communicator::CreateInner(zbccl_backend_t backendType, const CommGroupOptions &options,
                                          bool isWorldGroup)
{
    ZBCCL_LOG_DEBUG("CommGroupInfo dump: " << options);

    /* lock is acquired by caller already */

    if (gCommLookupMapByName_.find(options.name) != gCommLookupMapByName_.end()) {
        ZBCCL_LOG_AND_SET_LAST_ERROR("Create communicator failed as there is already one named " << options.name);
        return nullptr;
    }

    if (backendType == ZBCCL_ASCEND_NPU) {
        auto comm = ZMakeRef<NpuCommunicatorDefault>(options, isWorldGroup, gWorldZBCCLComm);
        if (comm == nullptr) {
            ZBCCL_LOG_AND_SET_LAST_ERROR("Create communicator failed, probably out of memory");
            return nullptr;
        }
        if (comm->Initialize()) {
            ZBCCL_LOG_AND_SET_LAST_ERROR("Init communicator failed.");
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
            gCommLookupMapByName_.emplace(options.name, comm.Get());
            return comm.Get();
        } else if (isWorldGroup && gWorldZBCCLComm != nullptr) {
            /*
             * if world group already created and return nullptr,
             * return nullptr directly as its already created
             */
            ZBCCL_LOG_AND_SET_LAST_ERROR("Create communicator failed as world group already created");
            return nullptr;
        } else if (!isWorldGroup && gWorldZBCCLComm == nullptr) {
            /*
             * if not world group and world group not created,
             * here we need to create world group firstly,
             * return nullptr
             */
            ZBCCL_LOG_AND_SET_LAST_ERROR("Create communicator failed as world group not created");
            return nullptr;
        } else {
            /*
             * if not world group and world group created
             */
            gCommLookupMap_.emplace(reinterpret_cast<uintptr_t>(comm.Get()), comm.Get());
            gCommLookupMapByName_.emplace(options.name, comm.Get());
            ZBCCL_LOG_DEBUG("Created communicator, name: " << options.name << ", isWorldGroup: " << isWorldGroup);
            return comm.Get();
        }
    }

    ZBCCL_LOG_DEBUG("Comm createInner exit with error");
    return nullptr;
}

ZResult Communicator::DestroyInner(CommunicatorPtr &comm)
{
    ZBCCL_VALIDATE_RETURN(comm != nullptr, "Invalid param, comm is null", Z_INVALID_PARAM);

    /* lock is acquired by caller already */

    /* if it is the world one */
    if (comm->isWorldGroup_) {
        if (gCommLookupMap_.size() != 0) {
            ZBCCL_LOG_AND_SET_LAST_ERROR("Destroy other non world communicator firstly, then destroy the world one");
            return Z_ERROR;
        }

        if (gWorldZBCCLComm != nullptr) {
            ZBCCL_LOG_INFO("Destroying the world communicator");
            gCommLookupMapByName_.erase(gWorldZBCCLComm->Name());
            gWorldZBCCLComm->DecreaseRef();
            gWorldZBCCLComm = nullptr;
        }
        return Z_OK;
    }

    /* erase from lookup map directly */
    auto iter = gCommLookupMap_.find(reinterpret_cast<uintptr_t>(comm.Get()));
    if (iter == gCommLookupMap_.end()) {
        ZBCCL_LOG_INFO("Destroy communicator failed as no such communicator existed");
        return Z_OK;
    }

    if (iter->second != nullptr) {
        gCommLookupMapByName_.erase(iter->second->Name());
        gCommLookupMap_.erase(iter);
    }

    return Z_OK;
}

void Communicator::DestroyAllInner()
{
    /* lock is acquired by caller already */

    /* clear all other world comm*/
    gCommLookupMap_.clear();
    gCommLookupMapByName_.clear();

    /* clear world one */
    if (gWorldZBCCLComm != nullptr) {
        gWorldZBCCLComm->DecreaseRef();
        gWorldZBCCLComm = nullptr;
    }

    /* reset  */
    GroupMetaArranger::Instance().UnInitialize();
}

ZResult Communicator::LookupInner(const std::string &name, CommunicatorPtr &comm)
{
    auto iter = gCommLookupMapByName_.find(name);
    if (iter == gCommLookupMapByName_.end()) {
        ZBCCL_LOG_INFO_AND_SET_LAST_ERROR("Communicator named " << name << " not existed");
        return Z_CCL_NOT_EXIST_BY_NAME;
    }

    ZBCCL_ASSERT_RETURN(iter->second != nullptr, Z_CCL_NOT_EXIST_BY_NAME);

    ZBCCL_LOG_DEBUG("Found communicator with name: " << name);
    comm = iter->second;
    return Z_OK;
}
}  // namespace ccl
}  // namespace zbccl