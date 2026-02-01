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
#include "zbccl_comm_host_device_struct.h"

namespace zbccl {
namespace ccl {

class Communicator;
using CommunicatorPtr = ZRef<Communicator>;

struct CommGroupOptions {
    std::string name;                    /* name of group */
    uint16_t worldSize = 0;              /* the ranks in the world */
    uint16_t groupSize = 0;              /* the ranks in the group */
    uint16_t myWorldRank = 0;            /* rank id in the world */
    uint16_t myGroupRank = 0;            /* rank id in the group */
    void *gva = nullptr;                 /* gva of the world */
    uint64_t metaSize = 0;               /* size of meta */
    uintptr_t myMetaGva = 0;             /* gva of mine */
    uintptr_t myParamDataGva = 0;        /* gva of for param exchange of operation */
    uintptr_t myAddressExchangeGva = 0;  /* gva of for param exchange of operation */
    uint64_t sizeForCommGroupInfo = 0;   /* max memory size of passing param from host to device */
    uint64_t sizeForParam = 0;           /* max memory size of passing param from host to device */
    uint64_t sizeForExchangeAddress = 0; /* max memory size for exchange operation data addresses */
    uint16_t deviceId = 0;               /* device Id */
    uint32_t groupIndex = 0;             /* group index */
    uint64_t fftsConfig = 0;             /* ffts config for operator in inner option*/
    uint64_t localDeviceMemSize = 0;     /* local device memory size */

    friend std::ostream &operator<<(std::ostream &os, const CommGroupOptions &options)
    {
        os << "CommGroupOptions [name: " << options.name << ", worldSize: " << options.worldSize
           << ", groupSize: " << options.groupSize << ", myWorldRank: " << options.myWorldRank
           << ", myGroupRank: " << options.myGroupRank << ", gva: " << options.gva << ", metaSize: " << options.metaSize
           << ", myMetaGva: " << std::hex << options.myMetaGva << ", myParamDataGva: " << options.myParamDataGva
           << ", myAddressExchangeGva: " << options.myAddressExchangeGva  << std::dec
           << ", sizeForCommGroupInfo: " << options.sizeForCommGroupInfo << ", sizeForParam: " << options.sizeForParam
           << ", sizeForExchangeAddress: " << options.sizeForExchangeAddress << ", deviceId: " << options.deviceId
           << ", groupIndex: " << options.groupIndex << ", fftsConfig: " << options.fftsConfig
           << ", localDeviceMemSize: " << options.localDeviceMemSize << "]";

        return os;
    }
};

class Communicator : public ZReferable
{
public:
    /**
     * @brief Factory function to create communicator
     *
     * @param options      [in] options of communicator
     * @param comm         [in/out] communicator ptr created
     * @param extraState   [in] extra state after bootstrap
     *
     * @return 0 if successful
     */
    static ZResult Create(const zbccl_comm_options_t &options, zbccl_comm_t *comm, const ZBCCLInitStateExt &extraState);

    /**
     * @brief Destroy on communicator
     *
     * @param comm         [in] communicator to be destroyed
     * @param flags        [in] optional flags
     *
     * @return 0 if successful
     */
    static ZResult Destroy(zbccl_comm_t comm, uint32_t flags);

    /**
     * @brief Destroy all communicators
     */
    static void DestroyAll();

    /**
     * @brief Lookup communicator by name
     *
     * @param name         [in] name of the communicator
     * @param comm         [in/out] the communicator ptr found
     * @return 0 if successful
     */
    static ZResult Lookup(const std::string &name, zbccl_comm_t *comm);

    /**
     * @brief Get the global communicator
     *
     * @param comm         [in/out] the commnicator ptr
     * @return 0 if successful or else error code
     */
    static ZResult GetGlobalComm(uintptr_t &comm);

    /**
     * @brief Lookup communicator property
     *
     * @param comm         [in] communicator
     * @param property     [out] property of communicator
     * @return 0 if successful or else error code
     */
    static ZResult GetCommProperty(const zbccl_comm_t comm, zbccl_comm_property_t *property);

    /**
     * @brief Get the count of communicators
     *
     * @return Count of existing communicators
     */
    static uint32_t Count();

public:
    Communicator(const CommGroupOptions &options, bool isWorldGroup, const CommunicatorPtr &worldGroup)
        : isWorldGroup_(isWorldGroup), worldGroup_(worldGroup), options_(options)
    {}

    ~Communicator() override = default;

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
     * @brief construct comm from option
     */
    virtual void ConstructCommGroupInfo(const CommGroupOptions &opt) noexcept = 0;

    /**
     * @brief Do allReduce operation
     *
     * @return 0 if successful
     */
    virtual int32_t AllReduce(const void *send_buff, void *recv_buff, size_t count, zbccl_datatype_t data_type,
                              zbccl_reduce_op_t op, aclrtStream stream) noexcept = 0;

    /**
     * @brief Do ReduceScatter operation
     *
     * @return 0 if successful
     */
    virtual int32_t ReduceScatter(const void *send_buff, void *recv_buff, size_t recv_count, zbccl_datatype_t data_type,
                                  zbccl_reduce_op_t op, aclrtStream stream) noexcept = 0;

    /**
     * @brief Do allGather operation
     *
     * @return 0 if successful
     */
    virtual int32_t AllGather(const void *send_buff, void *recv_buff, size_t send_count, zbccl_datatype_t data_type,
                              aclrtStream stream) noexcept = 0;

    /**
     * @brief Do All2all operation
     *
     * @return 0 if successful
     */
    virtual int32_t All2All(const void *sendBuff, void *recvBuff, uint64_t data_count, zbccl_datatype_t dataType,
                            uint64_t stride_count, uint8_t repeat, aclrtStream stream) noexcept = 0;

    /**
     * @brief Do dispatch normal notify operation
     *
     * @return 0 if successful
     */
    virtual int32_t DispatchNormalNotify(const zbccl_tensor_info_t *sendTokensPerExpert, int64_t sendCount,
                                         int64_t topKNum, const zbccl_tensor_info_t *recvBuff, int64_t *totalRecvTokens,
                                         const zbccl_tensor_info_t *recvTokensPerExpert,
                                         const zbccl_tensor_info_t *pushTargetOffset,
                                         const zbccl_tensor_info_t *balanceMatrix, int64_t flags) noexcept = 0;

    /**
     * @brief Dispatch normal layout
     *
     * @return 0 if successful
     */
    virtual int32_t DispatchNormalLayout(const zbccl_tensor_info_t *topkIndex, int64_t tokens, int64_t expertNum,
                                         int64_t topkNum, const zbccl_tensor_info_t *tokensPerRank,
                                         const zbccl_tensor_info_t *tokensPerExpert,
                                         const zbccl_tensor_info_t *isTokenInRank,
                                         const zbccl_tensor_info_t *sendTokensIndex,
                                         const zbccl_tensor_info_t *notifySendData, aclrtStream stream,
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
                                  const zbccl_tensor_info_t *sendTokensIndex,
                                  const zbccl_tensor_info_t *balanceMatrix, uint16_t expertNum,
                                  const zbccl_tensor_info_t *destTokens, zbccl_comm_t comm, aclrtStream stream,
                                  int64_t flags) noexcept = 0;

    /**
     * @brief Check if it is world group
     *
     * @return true if world group
     */
    bool IsWorldGroup() const noexcept;

    /**
     * @brief Group info of communicator, this will be passed to device in the param area of group meta
     *
     * @return meta info
     */
    const CommGroupInfo &GetMetaInfo() const noexcept;

    /**
     * @brief Get the name of communicator
     *
     * @return name string of communicator
     */
    const std::string &Name() const noexcept;

protected:
    bool isWorldGroup_ = false;           /* if it is world group */
    CommGroupOptions options_{};          /* options */
    CommGroupInfo groupInfo_{};           /* meta info, which will be H2D to device, keep it simple */
    CommunicatorPtr worldGroup_{nullptr}; /* world group */

private:
    static CommunicatorPtr CreateInner(zbccl_backend_t backendType, const CommGroupOptions &options, bool isWorldGroup);
    static ZResult DestroyInner(CommunicatorPtr &comm);
    static void DestroyAllInner();
    static ZResult LookupInner(const std::string &name, CommunicatorPtr &comm);

    static CommunicatorPtr gWorldCommunicator;                           /* the world comm, i.e. the first one */
    static std::mutex gMutex;                                            /* mutex for world comm */
    static std::map<uintptr_t, CommunicatorPtr> gCommLookupMap_;         /* all comm object except the world comm */
    static std::map<std::string, CommunicatorPtr> gCommLookupMapByName_; /* all comm object */
};

inline bool Communicator::IsWorldGroup() const noexcept
{
    return isWorldGroup_;
}

inline const CommGroupInfo &Communicator::GetMetaInfo() const noexcept
{
    return groupInfo_;
}

inline const std::string &Communicator::Name() const noexcept
{
    return options_.name;
}

}  // namespace ccl
}  // namespace zbccl

#endif  // ZBCCL_COMMUNICATOR_H
