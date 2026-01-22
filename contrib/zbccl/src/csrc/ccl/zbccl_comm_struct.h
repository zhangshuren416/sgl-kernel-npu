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
struct ZBCommOptions {
    std::string name;                        /* name */
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
        os << "ZBCommOptions [name: " << options.name << ", worldSize: " << options.worldSize
           << ", groupSize: " << options.groupSize << ", myWorldRank: " << options.myWorldRank
           << ", myGroupRank: " << options.myGroupRank << ", gva: " << options.gva
           << ", metaSizeOfDevice: " << options.metaSizeOfDevice << ", myMetaDataGva: " << options.myMetaDataGva
           << ", metaSizeForExchangeAddress: " << options.metaSizeForExchangeAddress
           << ", myParamDataGva: " << options.myParamDataGva
           << ", sizeForExchangeParam: " << options.sizeForExchangeParam << ", deviceId: " << options.deviceId
           << ", groupIndex: " << options.groupIndex << "]";

        return os;
    }
};

/**
 * @brief meta info of this communicator, this struct will be copy to device, keep it simple
 */
struct ZBCommMetaInfo {
    uint16_t worldSize = 0;                                 /* the ranks in the world */
    uint16_t groupSize = 0;                                 /* the ranks in the group */
    uint16_t myWorldRank = 0;                               /* rank id in the world */
    uint16_t myGroupRank = 0;                               /* rank id in the group */
    void *gva = nullptr;                                    /* gva of the world */
    uint64_t metaSizeOfDevice = 0;                          /* size of meta */
    uintptr_t myMetaDataGva = 0;                            /* gva of mine */
    uint64_t metaSizeForExchangeAddress = 0;                /* max memory size for exchange operation data addresses */
    uintptr_t myParamDataGva = 0;                           /* gva of for param exchange of operation */
    uint64_t sizeForExchangeParam = 0;                      /* max memory size of passing param from host to device */
    uint16_t peerGroupRank2WorldRank[ZBCCL_MAX_RANKS] = {}; /* rank id in group to world rank id relationship */

    ZBCommMetaInfo() = default;
    ZBCommMetaInfo(const ZBCommOptions &options)
    {
        worldSize = options.worldSize;
        groupSize = options.groupSize;
        myWorldRank = options.myWorldRank;
        myGroupRank = options.myGroupRank;
        gva = options.gva;
        metaSizeOfDevice = options.metaSizeOfDevice;
        myMetaDataGva = options.myMetaDataGva;
        metaSizeForExchangeAddress = options.metaSizeForExchangeAddress;
        myParamDataGva = options.myParamDataGva;
        sizeForExchangeParam = options.sizeForExchangeParam;
    }

    friend std::ostream &operator<<(std::ostream &os, const ZBCommMetaInfo &options)
    {
        os << "ZBCommMetaInfo [worldSize: " << options.worldSize << ", groupSize: " << options.groupSize
           << ", myWorldRank: " << options.myWorldRank << ", myGroupRank: " << options.myGroupRank
           << ", gva: " << options.gva << ", metaSizeOfDevice: " << options.metaSizeOfDevice
           << ", myMetaDataGva: " << options.myMetaDataGva
           << ", metaSizeForExchangeAddress: " << options.metaSizeForExchangeAddress
           << ", myParamDataGva: " << options.myParamDataGva
           << ", sizeForExchangeParam: " << options.sizeForExchangeParam << ", peerGroupRank2WorldRank: [";

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
