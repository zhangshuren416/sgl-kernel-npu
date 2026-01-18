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
#include "zbccl_communicator_default.h"

namespace zbccl {
namespace ccl {
ZBCCLCommPtr ZBCCLComm::gWorldZBCCLComm{nullptr};
std::mutex ZBCCLComm::gMutex;

ZBCCLComm::ZBCCLComm(const ZBCommOptions &options, bool isWorldGroup, const ZBCCLCommPtr &worldGroup)
    : isWorldGroup_(isWorldGroup), worldGroup_(worldGroup)
{
    metaInfo_.worldSize = options.worldSize;
    metaInfo_.groupSize = options.groupSize;
    metaInfo_.myWorldRank = options.myWorldRank;
    metaInfo_.myGroupRank = options.myGroupRank;
    metaInfo_.metaDataGva = options.metaDataGva;
    metaInfo_.myMetaDataGva = options.myMetaDataGva;
}

ZBCCLCommPtr ZBCCLComm::Create(zbccl_backend_t backendType, const ZBCommOptions &options, bool isWorldGroup)
{
    std::lock_guard<std::mutex> guard(gMutex);
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

ZResult ZBCCLComm::Destroy(zbccl::ccl::ZBCCLCommPtr &comm)
{
    ZBCCL_VALIDATE_RETURN(comm == nullptr, "invalid param, ZBCCLComm is null", Z_INVALID_PARAM);

    std::lock_guard<std::mutex> guard(gMutex);
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

void ZBCCLComm::DestroyAll()
{
    std::lock_guard<std::mutex> guard(gMutex);
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