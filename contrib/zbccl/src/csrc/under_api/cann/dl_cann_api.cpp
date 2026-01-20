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
#include <dlfcn.h>

#include "dl_cann_api.h"

namespace zbccl {
namespace underapi {
bool DlCannApi::gLoaded = false;
std::mutex DlCannApi::gMutex;
void *DlCannApi::rtHandle;
const char *DlCannApi::gAscendAclLibName = "libascendcl.so";

aclrtGetSocNameFunc DlCannApi::pAclrtGetSocName = nullptr;
rtGetDeviceInfoFunc DlCannApi::pRtGetDeviceInfo = nullptr;
aclrtGetDeviceFunc DlCannApi::pAclrtGetDevice = nullptr;
aclrtSetDeviceFunc DlCannApi::pAclrtSetDevice = nullptr;
aclrtSynchronizeStreamFunc DlCannApi::pAclrtSynchronizeStream = nullptr;
aclrtMallocFunc DlCannApi::pAclrtMalloc = nullptr;
aclrtFreeFunc DlCannApi::pAclrtFree = nullptr;
aclrtMallocHostFunc DlCannApi::pAclrtMallocHost = nullptr;
aclrtFreeHostFunc DlCannApi::pAclrtFreeHost = nullptr;
aclrtMemcpyFunc DlCannApi::pAclrtMemcpy = nullptr;
aclrtMemcpyAsyncFunc DlCannApi::pAclrtMemcpyAsync = nullptr;
aclrtMemsetFunc DlCannApi::pAclrtMemset = nullptr;
rtGetLogicDevIdByUserDevIdFunc DlCannApi::pRtGetLogicDevIdByUserDevId = nullptr;

ZResult DlCannApi::LoadLibrary(const std::string &libDirPath)
{
    std::lock_guard<std::mutex> guard(gMutex);
    if (gLoaded) {
        return Z_OK;
    }

    std::string realPath;
    if (!Func::LibraryRealPath(libDirPath, std::string(gAscendAclLibName), realPath)) {
        ZBCCL_LOG_ERROR(libDirPath << "get real library [" << gAscendAclLibName << "] failed");
        return Z_FILE_NOT_FOUND;
    }

    /* dlopen library */
    rtHandle = dlopen(realPath.c_str(), RTLD_NOW | RTLD_NODELETE);
    if (rtHandle == nullptr) {
        ZBCCL_LOG_ERROR("Failed to open library [" << realPath << "], error: " << dlerror());
        return Z_DL_OPEN_LIB_FAILED;
    }

    /* load sym */
    DL_LOAD_SYM(pAclrtGetSocName, aclrtGetSocNameFunc, rtHandle, "aclrtGetSocName");
    DL_LOAD_SYM(pRtGetDeviceInfo, rtGetDeviceInfoFunc, rtHandle, "rtGetDeviceInfo");
    DL_LOAD_SYM(pAclrtGetDevice, aclrtGetDeviceFunc, rtHandle, "aclrtGetDevice");
    DL_LOAD_SYM(pAclrtSetDevice, aclrtSetDeviceFunc, rtHandle, "aclrtSetDevice");
    DL_LOAD_SYM(pAclrtSynchronizeStream, aclrtSynchronizeStreamFunc, rtHandle, "aclrtSynchronizeStream");
    DL_LOAD_SYM(pAclrtMalloc, aclrtMallocFunc, rtHandle, "aclrtMalloc");
    DL_LOAD_SYM(pAclrtFree, aclrtFreeFunc, rtHandle, "aclrtFree");
    DL_LOAD_SYM(pAclrtMallocHost, aclrtMallocHostFunc, rtHandle, "aclrtMallocHost");
    DL_LOAD_SYM(pAclrtFreeHost, aclrtFreeHostFunc, rtHandle, "aclrtFreeHost");
    DL_LOAD_SYM(pAclrtMemcpy, aclrtMemcpyFunc, rtHandle, "aclrtMemcpy");
    DL_LOAD_SYM(pAclrtMemcpyAsync, aclrtMemcpyAsyncFunc, rtHandle, "aclrtMemcpyAsync");
    DL_LOAD_SYM(pAclrtMemset, aclrtMemsetFunc, rtHandle, "aclrtMemset");
    DL_LOAD_SYM(pRtGetLogicDevIdByUserDevId, rtGetLogicDevIdByUserDevIdFunc, rtHandle, "rtGetLogicDevIdByUserDevId");

    gLoaded = true;

    return Z_OK;
}

void DlCannApi::CleanupLibrary()
{
    std::lock_guard<std::mutex> guard(gMutex);
    if (!gLoaded) {
        return;
    }

    pAclrtGetSocName = nullptr;
    pRtGetDeviceInfo = nullptr;
    pAclrtGetDevice = nullptr;
    pAclrtSetDevice = nullptr;
    pAclrtSynchronizeStream = nullptr;
    pAclrtMalloc = nullptr;
    pAclrtFree = nullptr;
    pAclrtMallocHost = nullptr;
    pAclrtFreeHost = nullptr;
    pAclrtMemcpy = nullptr;
    pAclrtMemcpyAsync = nullptr;
    pAclrtMemset = nullptr;
    pRtGetLogicDevIdByUserDevId = nullptr;

    if (rtHandle != nullptr) {
        dlclose(rtHandle);
        rtHandle = nullptr;
    }
    gLoaded = false;
}
}  // namespace underapi
}  // namespace zbccl