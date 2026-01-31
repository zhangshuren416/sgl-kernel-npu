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
#ifndef ZBCCL_STRUCT_DUMP_HELPER_H
#define ZBCCL_STRUCT_DUMP_HELPER_H

#include "zbccl_common_includes.h"

namespace zbccl {
constexpr uint16_t CCL_META_SPACE_SIZE_MIN = 1024;     /* 1MB */
constexpr uint16_t CCL_META_SPACE_SIZE_DEFAULT = 1024; /* 1MB */
constexpr uint16_t CCL_META_SPACE_SIZE_MAX = 4096;     /* 4MB */
constexpr uint16_t CCL_GROUP_COUNT_CAP_MIN = 1;
constexpr uint16_t CCL_GROUP_COUNT_CAP_DEFAULT = 128;
constexpr uint16_t CCL_GROUP_COUNT_CAP_MAX = 256;

static inline std::ostream &operator<<(std::ostream &os, const zbccl_bootstrap_options_t &options)
{
    os << "zbccl_bootstrap_options_t [flags: " << options.flags << ", bootstrap_type: " << options.btType
       << ", ipPort: " << ((options.ipPort == nullptr) ? "" : options.ipPort) << ", worldSize: " << options.worldSize
       << ", rankId: " << options.rankId << ", deviceId: " << options.deviceId
       << ", startConfigServer: " << options.startConfigServer << ", deviceMemorySize: " << options.deviceMemorySize
       << ", dataOperationType: " << options.dataOperationType << ", cclMetaSpaceSize: " << options.cclMetaSpaceSize
       << ", cclGroupCap: " << options.cclGroupCap << "]";

    return os;
}

static inline std::ostream &operator<<(std::ostream &os, const zbccl_bootstrap_output_t &output)
{
    os << "zbccl_bootstrap_output_t [deviceGva: " << output.deviceGva
       << ", allocatedDeviceMemorySize: " << output.allocatedDeviceMemorySize << ", myDeviceGva: " << output.myDeviceGva
       << ", myCCLMetaDeviceGva: " << output.myCCLMetaDeviceGva << ", metaSizeOfDevice: " << output.metaSizeOfDevice
       << ", mySMAGva: " << output.mySMAGva << ", myDeviceGva: " << output.myDeviceGva
       << ", smaSizeOfDevice: " << output.smaSizeOfDevice << "]";

    return os;
}

static inline std::ostream &operator<<(std::ostream &os, const zbccl_allocator_options_t &options)
{
    os << "zbccl_allocator_options_t [gva: " << options.gva << ", myGva: " << options.myGva
       << ", size: " << options.size << "]";

    return os;
}

static inline std::ostream &operator<<(std::ostream &os, const zbccl_comm_options_t &options)
{
    os << "zbccl_comm_options_t [name: " << (options.name != nullptr ? options.name : "")
       << "backendType: " << options.backendType << ", flags: " << options.flags
       << ", isWorldGroup: " << options.isWorldGroup << ", groupSize: " << options.groupSize
       << ", groupRankId: " << options.groupRankId << "]";

    return os;
}

static inline std::ostream &operator<<(std::ostream &os, const zbccl_comm_property_t &property)
{
    os << "zbccl_comm_property_t [backendType: " << property.backendType << ", flags: " << property.flags
       << ", isWorldGroup: " << property.isWorldGroup << ", groupSize: " << property.groupSize
       << ", groupRankId: " << property.groupRankId
       << ", myGVA: " << property.myGVA << ", myMetaGVA: " << property.myMetaGVA
       << ", sizeOfMetaArea: " << property.sizeOfMetaArea
       << ", sizeOfMetaForAddressExchange: " << property.sizeOfMetaForAddressExchange
       << ", myMetaGVAForOpParam: " << property.myMetaGVAForOpParam
       << ", sizeOfMetaForOpParam: " << property.sizeOfMetaForOpParam << ", groupIndex: " << property.groupIndex << "]";

    return os;
}

static inline std::ostream &operator<<(std::ostream &os, const zbccl_tensor_info_t &info)
{
    os << "zbccl_tensor_info_t [data: " << info.data << ", dataType: " << info.dataType << ", dim: " << info.dim << "]";

    return os;
}
}  // namespace zbccl

#endif  // ZBCCL_STRUCT_DUMP_HELPER_H
