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
#ifndef SGL_KERNEL_NPU_BAO_ZBCCL_COMM_STRUCT_H
#define SGL_KERNEL_NPU_BAO_ZBCCL_COMM_STRUCT_H

namespace zbccl {
namespace ccl {
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
           << ", myMetaGva: " << options.myMetaGva << ", myParamDataGva: " << options.myParamDataGva
           << ", myAddressExchangeGva: " << options.myAddressExchangeGva
           << ", sizeForCommGroupInfo: " << options.sizeForCommGroupInfo << ", sizeForParam: " << options.sizeForParam
           << ", sizeForExchangeAddress: " << options.sizeForExchangeAddress << ", deviceId: " << options.deviceId
           << ", groupIndex: " << options.groupIndex << ", fftsConfig: " << options.fftsConfig
           << ", localDeviceMemSize: " << options.localDeviceMemSize << "]";

        return os;
    }
};

/**
 * @brief group info of this communicator, this struct will be copy to device, keep it simple
 */
struct CommGroupInfo {
    uint16_t groupSize = 0;                                 /* the ranks in the group */
    uint16_t myGroupRank = 0;                               /* rank id in the group */
    uintptr_t myMetaGva = 0;                                /* gva of mine */
    uintptr_t myParamDataGva = 0;                           /* gva of for param exchange of operation */
    uintptr_t myAddressExchangeGva = 0;                     /* gva of for param exchange of operation */
    uint64_t sizeForCommGroupInfo = 0;                      /* max memory size of passing param from host to device */
    uint64_t sizeForParam = 0;                              /* max memory size of passing param from host to device */
    uint64_t sizeForExchangeAddress = 0;                    /* max memory size for exchange operation data addresses */
    uint64_t fftsConfig;                                    /* copy from CommGroupOptions.fftsConfig */
    uint16_t peerGroupRank2WorldRank[ZBCCL_MAX_RANKS] = {}; /* rank id in group to world rank id relationship */
    uint64_t localDeviceMemSize;                            /* copy from CommGroupOptions.localDeviceMemSize */

    CommGroupInfo() = default;
    CommGroupInfo(const CommGroupOptions &options)
    {
        groupSize = options.groupSize;
        myGroupRank = options.myGroupRank;
        myMetaGva = options.myMetaGva;
        myParamDataGva = options.myParamDataGva;
        myAddressExchangeGva = options.myAddressExchangeGva;
        sizeForCommGroupInfo = options.sizeForCommGroupInfo;
        sizeForParam = options.sizeForParam;
        sizeForExchangeAddress = options.sizeForExchangeAddress;
        fftsConfig = options.fftsConfig;
        localDeviceMemSize = options.localDeviceMemSize;
    }

    friend std::ostream &operator<<(std::ostream &os, const CommGroupInfo &options)
    {
        os << "CommGroupInfo [groupSize: " << options.groupSize << ", myGroupRank: " << options.myGroupRank
           << ", myMetaGva: " << options.myMetaGva << ", myParamDataGva: " << options.myParamDataGva
           << ", myAddressExchangeGva: " << options.myAddressExchangeGva
           << ", sizeForCommGroupInfo: " << options.sizeForCommGroupInfo << ", sizeForParam: " << options.sizeForParam
           << ", sizeForExchangeAddress: " << options.sizeForExchangeAddress << ", fftsConfig: " << options.fftsConfig
           << ", localDeviceMemSize: " << options.localDeviceMemSize << ", peerGroupRank2WorldRank: [";

        for (auto i = 0; i < ZBCCL_MAX_RANKS; ++i) {
            if (options.peerGroupRank2WorldRank[i] != 0) {
                os << options.peerGroupRank2WorldRank[i] << ",";
            }
        }

        os << "]]";

        return os;
    }
};
}  // namespace ccl
}  // namespace zbccl

#endif  // SGL_KERNEL_NPU_BAO_ZBCCL_COMM_STRUCT_H
