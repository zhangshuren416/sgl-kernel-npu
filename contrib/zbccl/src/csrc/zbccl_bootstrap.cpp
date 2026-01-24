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
#include "zbccl_common_includes.h"
#include "zbccl_bootstrap_default.h"
#include "zbccl_init_state.h"

using namespace zbccl;
using namespace zbccl::bootstrap;

#ifdef __cplusplus
extern "C" {
#endif

ZBCCL_API int32_t zbccl_bootstrap_options_init(zbccl_bootstrap_options_t *options)
{
    ZBCCL_VALIDATE_RETURN(options != nullptr, "invalid param, bootstrap options should not be null", Z_INVALID_PARAM);

    bzero(options, sizeof(zbccl_bootstrap_options_t));
    options->btType = BOOT_BY_MEMFABRIC;
    options->startConfigServer = 0;
    options->cclMetaSpaceSize = CCL_META_SPACE_SIZE_DEFAULT;
    options->cclGroupCap = CCL_GROUP_COUNT_CAP_DEFAULT;

    return Z_OK;
}

ZBCCL_API int32_t zbccl_bootstrap(zbccl_bootstrap_options_t *options, zbccl_bootstrap_output_t *output)
{
    ZBCCL_VALIDATE_RETURN(options != nullptr, "invalid param, bootstrap options should not be null", Z_INVALID_PARAM);
    ZBCCL_VALIDATE_RETURN(output != nullptr, "invalid param, bootstrap output should not be null", Z_INVALID_PARAM);

    ZBCCL_LOG_INFO("options dump, " << (*options));

    auto bootstrap = Bootstrap::Create(*options);
    if (bootstrap == nullptr) {
        return Z_ERROR;
    }

    /* get and set output */
    auto &out = bootstrap->GetOutput();
    memcpy(output, &out, sizeof(zbccl_bootstrap_output_t));

    /* set init state */
    auto &state = ZBCCLInitState::Instance();
    state.Bootstrapped(true);
    state.ext_.btType = options->btType;
    state.ext_.worldSize = options->worldSize;
    state.ext_.worldRankId = options->rankId;
    state.ext_.deviceId = options->deviceId;
    state.ext_.cclMetaSpaceSize = options->cclMetaSpaceSize;
    state.ext_.cclGroupCap = options->cclGroupCap;
    state.ext_.gvaDevice = output->deviceGva;
    state.ext_.myCCLMetaDeviceGva = output->myCCLMetaDeviceGva;
    state.ext_.metaSizeOfDevice = output->metaSizeOfDevice;
    state.ext_.mySMAGva = output->mySMAGva;
    state.ext_.smaSizeOfDevice = output->smaSizeOfDevice;
    state.ext_.localDeviceMemSize = output->allocatedDeviceMemorySize;

    return Z_OK;
}

ZBCCL_API int32_t zbccl_unbootstrap(uint32_t flags)
{
    /* check if sma and ccl still there */
    if (ZBCCLInitState::Instance().HasCommunicator()) {
        ZBCCL_LOG_AND_SET_LAST_ERROR(
            "Cannot un-bootstrap as there are still communicator existed, need to destroy all communicators firstly");
        return Z_CANNOT_UNBOOTSTRAP;
    } else if (ZBCCLInitState::Instance().SmaInitialized()) {
        ZBCCL_LOG_AND_SET_LAST_ERROR(
            "Cannot un-bootstrap as sma is still not un-initialized, need to un-initialize sma firstly");
        return Z_CANNOT_UNBOOTSTRAP;
    }

    Bootstrap::Destroy();
    ZBCCLInitState::Instance().Reset();
    ZBCCL_LOG_INFO("ZBCCL un-bootstrap successfully");
    return Z_OK;
}

#ifdef __cplusplus
}
#endif