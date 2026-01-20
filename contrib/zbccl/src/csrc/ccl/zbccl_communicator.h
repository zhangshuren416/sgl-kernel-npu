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
    uint16_t worldSize = 0;                  /* the ranks in the world */
    uint16_t groupSize = 0;                  /* the ranks in the group */
    uint16_t myWorldRank = 0;                /* rank id in the world */
    uint16_t myGroupRank = 0;                /* rank id in the group */
    void *gva = nullptr;                     /* gva of the world */
    uint64_t metaSizeOfDevice = 0;           /* size of meta */
    uintptr_t myMetaDataGva = 0;             /* gva of mine */
    uint64_t metaSizeForExchangeAddress = 0; /* max memory size for exchange operation data addresses */
    uintptr_t myParamDataGva = 0;            /* gva of for param exchange of operation */
    uint64_t sizeForExchangeParam = 0;       /* max memory size of passing param from host to device */
    uint16_t deviceId = 0;                   /* device Id */
    uint32_t groupIndex = 0;                 /* group index */

    friend std::ostream &operator<<(std::ostream &os, const ZBCommOptions &options)
    {
        os << "ZBCommOptions [worldSize: " << options.worldSize << ", groupSize: " << options.groupSize
           << ", myWorldRank: " << options.myWorldRank << ", myGroupRank: " << options.myGroupRank
           << ", gva: " << options.gva << ", metaSizeOfDevice: " << options.metaSizeOfDevice
           << ", myMetaDataGva: " << options.myMetaDataGva
           << ", metaSizeForExchangeAddress: " << options.metaSizeForExchangeAddress
           << ", myParamDataGva: " << options.myParamDataGva
           << ", sizeForExchangeParam: " << options.sizeForExchangeParam << ", deviceId: " << options.deviceId
           << ", groupIndex: " << options.groupIndex << "]";

        return os;
    }
};

struct ZBCommMetaInfo : ZBCommOptions {
    uint16_t peerGroupRank2WorldRank[ZBCCL_MAX_RANKS] = {}; /* rank id in group to world rank id relationship */
};

class ZBCCLComm;
using ZBCCLCommPtr = ZRef<ZBCCLComm>;

class ZBCCLComm : public ZReferable
{
public:
    static ZResult Create(const zbccl_ccl_options_t &options, zbccl_comm_t *comm, const ZBCCLInitStateExt &extraState);
    static ZResult Destroy(zbccl_comm_t comm, uint32_t flags);
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

    /**
     * @brief Do All2all operation
     *
     * @return 0 if successful
     */
    virtual int32_t All2All(const void *sendBuff, void *recvBuff, uint64_t data_count, zbccl_datatype_t dataType,
                            uint64_t stride_count, uint8_t repeat) noexcept = 0;

    /**
     * @brief Do dispatch normal notify operation
     *
     * @return 0 if successful
     */
    virtual int32_t DispatchNormalNotify(const zbccl_tensor_info_t *sendTokensPerExpert, int64_t sendCount,
                                         int64_t topKNum, const zbccl_tensor_info_t *recvBuff, int64_t *totalRecvTokens,
                                         const zbccl_tensor_info_t *recvTokensPerExpert,
                                         const zbccl_tensor_info_t *pushTargetOffset, int64_t flags) noexcept = 0;

    /**
     * @brief Dispatch normal layout
     *
     * @return 0 if successful
     */
    virtual int32_t DispatchNormalLayout(const zbccl_tensor_info_t *topkIndex, int64_t tokens, int64_t expertNum,
                                         int64_t topkNum, const zbccl_tensor_info_t *tokensPerRank,
                                         const zbccl_tensor_info_t *tokensPerExpert,
                                         const zbccl_tensor_info_t *isTokenInRank,
                                         const zbccl_tensor_info_t *tokenIndex, zbccl_comm_t comm, aclrtStream stream,
                                         int64_t flags) noexcept = 0;

    /**
     * @brief Dispatch operation
     *
     * @return 0 if successful
     */
    virtual int32_t DispatchNormal(const zbccl_tensor_info_t *srcTokens, const zbccl_tensor_info_t *topkIndex,
                                   const zbccl_tensor_info_t *sendTokensIndex,
                                   const zbccl_tensor_info_t *pushTargetOffset, int64_t expertNum,
                                   zbccl_quant_mode_t quantMode, const zbccl_tensor_info_t *destTokens,
                                   const zbccl_tensor_info_t *destScale, zbccl_comm_t comm, aclrtStream stream,
                                   int64_t flags) noexcept = 0;

    /**
     * @brief Combine operation
     *
     * @return 0 if successful
     */
    virtual int32_t CombineNormal(const zbccl_tensor_info_t *srcTokens, const zbccl_tensor_info_t *srcTokensPerEp,
                                  const zbccl_tensor_info_t *topKWeight, const zbccl_tensor_info_t *topkIndex,
                                  const zbccl_tensor_info_t *sendTokensIndex, uint16_t expertNum,
                                  const zbccl_tensor_info_t *destTokens, zbccl_comm_t comm, aclrtStream stream,
                                  int64_t flags) noexcept = 0;

    /**
     * @brief Check if it is world group
     *
     * @return true if world group
     */
    bool IsWorldGroup() const;

    const ZBCommMetaInfo &GetMetaInfo() const;

protected:
    bool isWorldGroup_ = false; /* if it is world group */
    ZBCommMetaInfo metaInfo_{}; /* meta info */
    ZBCCLCommPtr worldGroup_;   /* world group */

private:
    static ZBCCLCommPtr CreateInner(zbccl_backend_t backendType, const ZBCommOptions &options, bool isWorldGroup);
    static ZResult DestroyInner(ZBCCLCommPtr &comm);
    static void DestroyAllInner();

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
