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
#ifndef ZBCCL_ADAPTOR_PY_UTIL_H
#define ZBCCL_ADAPTOR_PY_UTIL_H

#include "zbccl_def.h"
#include <torch/csrc/distributed/c10d/Types.hpp>
#include "torch_npu/csrc/core/npu/sys_ctrl/npu_sys_ctrl.h"
#include "torch_npu/csrc/framework/FormatHelper.h"
#include "torch_npu/csrc/core/NPUBridge.h"
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "torch_npu/csrc/core/npu/DeviceUtils.h"
#include "torch_npu/csrc/core/npu/NPUFormat.h"
#include "torch_npu/csrc/core/npu/NPUEvent.h"
#include "torch_npu/csrc/framework/OpCommand.h"

namespace zbccl {
namespace adaptor {
namespace pytorch_npu {

std::vector<at::Device> GetDeviceList(const std::vector<at::Tensor> &tensors);

std::string GetKeyFromDevices(const std::vector<at::Device> &devices);

void SyncStreams(const std::vector<at::Device> &devices,
    std::vector<c10_npu::NPUEvent> &zbcclEvents,
    std::vector<c10_npu::NPUStream> &zbcclStreams);

void CheckTensors(const std::vector<at::Tensor> &tensors);

std::vector<at::Tensor> CastOriginFormat(const std::vector<at::Tensor>& inputTensors);

int32_t CheckNpuTensorsDifferentDevices(const std::vector<at::Tensor> &tensors);

uint64_t GetNumelForZBCCL(const at::Tensor &t);

zbccl_datatype_t GetZBcclDataType(at::ScalarType type);

zbccl_reduce_op_t GetZBcclReduceOp(const c10d::ReduceOp op);

}
}
}

#endif