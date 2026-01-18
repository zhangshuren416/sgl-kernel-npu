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
#ifndef ZBCCL_COMMUNICATOR_H
#define ZBCCL_COMMUNICATOR_H

#include "zbccl_common_includes.h"

namespace zbccl {
namespace ccl {
struct ZBCommOptions {
    uint16_t worldSize = 0;     /* the ranks in the world */
    uint16_t groupSize = 0;     /* the ranks in the group */
    uint16_t myWorldRank = 0;   /* rank id in the world */
    uint16_t myGroupRank = 0;   /* rank id in the group */
    uint64_t metaDataGva = 0;   /* gva of the world */
    uint64_t myMetaDataGva = 0; /* gva of mine */
};

struct ZBCommMetaInfo : ZBCommOptions {
    uint16_t peerGroupRank2WorldRank[ZBCCL_MAX_RANKS] = {}; /* rank id in group to world rank id relationship */
};

class ZBCCLComm;
using ZBCCLCommPtr = ZRef<ZBCCLComm>;

class ZBCCLComm : public ZReferable
{
public:
    static ZBCCLCommPtr Create(zbccl_backend_t backendType, const ZBCommOptions &options, bool isWorldGroup);
    static ZResult Destroy(ZBCCLCommPtr &comm);
    static void DestroyAll();

public:
    ZBCCLComm(const ZBCommOptions &options, bool isWorldGroup, const ZBCCLCommPtr &worldGroup);
    ~ZBCCLComm() override = default;

    /**
     * @brief Initialize communicator
     *
     * @return 0 if successful
     */
    virtual ZResult Initialize() noexcept = 0;

    /**
     * @brief Un-initialize communicator
     */
    virtual void UnInitialize() noexcept = 0;

    /**
     * @brief Do allReduce operation
     *
     * @return 0 if successful
     */
    virtual int32_t AllReduce(const void *send_buff, void *recv_buff, size_t count, zbccl_datatype_t data_type,
                              zbccl_reduce_op_t op) noexcept = 0;

    /**
     * @brief Do ReduceScatter operation
     *
     * @return 0 if successful
     */
    virtual int32_t ReduceScatter(const void *send_buff, void *recv_buff, size_t recv_count, zbccl_datatype_t data_type,
                                  zbccl_reduce_op_t op) noexcept = 0;

    /**
     * @brief Do allGather operation
     *
     * @return 0 if successful
     */
    virtual int32_t AllGather(const void *send_buff, void *recv_buff, size_t send_count,
                              zbccl_datatype_t data_type) noexcept = 0;

    bool IsWorldGroup() const;

    const ZBCommMetaInfo &GetMetaInfo() const;

protected:
    bool isWorldGroup_ = false; /* if it is world group */
    ZBCommMetaInfo metaInfo_{}; /* meta info */
    ZBCCLCommPtr worldGroup_;   /* world group */

private:
    static ZBCCLCommPtr gWorldZBCCLComm;                           /* the world comm, i.e. the first one */
    static std::mutex gMutex;                                      /* mutex for world comm */
    static std::map<uintptr_t, ZBCCLCommPtr> gZBCCLCommLookupMap_; /* all comm object except the world comm */
};

inline bool ZBCCLComm::IsWorldGroup() const
{
    return isWorldGroup_;
}

inline const ZBCommMetaInfo &ZBCCLComm::GetMetaInfo() const
{
    return metaInfo_;
}

}  // namespace ccl
}  // namespace zbccl

#endif  // ZBCCL_COMMUNICATOR_H
