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
#include "zbccl_pytorch_util.h"

namespace zbccl {
namespace adaptor {
namespace pytorch_npu {

const std::map<at::ScalarType, zbccl_datatype_t> kScalarTypeToZBcclDataType = {
    {at::kByte, ZBCCL_DATA_TYPE_UINT8},
    {at::kChar, ZBCCL_DATA_TYPE_INT8},
    {at::kShort, ZBCCL_DATA_TYPE_INT16},
    {at::kInt, ZBCCL_DATA_TYPE_INT32},
    {at::kLong, ZBCCL_DATA_TYPE_INT64},
    {at::kHalf, ZBCCL_DATA_TYPE_FP16},
    {at::kFloat, ZBCCL_DATA_TYPE_FP32},
    {at::kDouble, ZBCCL_DATA_TYPE_FP64},
    {at::kBool, ZBCCL_DATA_TYPE_UINT8},
    {at::kBFloat16, ZBCCL_DATA_TYPE_BFP16},
};

std::vector<at::Device> GetDeviceList(const std::vector<at::Tensor> &tensors)
{
    std::vector<at::Device> devices;
    devices.reserve(tensors.size());
    for (auto &tensor : tensors) {
        devices.push_back(tensor.device());
    }
    return devices;
}

std::string GetKeyFromDevices(const std::vector<at::Device> &devices)
{
    std::string deviceList;
    for (auto &device : devices) {
        if (deviceList.empty()) {
            deviceList = std::to_string(device.index());
        } else {
            deviceList += "," + std::to_string(device.index());
        }
    }
    return deviceList;
}

void SyncStreams(const std::vector<at::Device> &devices, std::vector<c10_npu::NPUEvent> &events,
    std::vector<c10_npu::NPUStream> &streams)
{
    for (size_t i = 0; i < devices.size(); ++i) {
        c10_npu::NPUStream &zbcclSteam = streams[i];
        c10_npu::NPUEvent &event = events[i];
        event.record(c10_npu::getCurrentNPUStream(devices[i].index()));
        event.block(zbcclSteam);
    }
}

void CheckTensors(const std::vector<at::Tensor> &tensors)
{
    if (tensors.empty()) {
        TORCH_CHECK(false, "Tensor list must be nonempty", DIST_ERROR(ErrCode::PARAM));
    }

    // ZBCCL support one NPU per process only
    if (tensors.size() != 1) {
        TORCH_CHECK(false, "Tensor list mustn't be larger than the number of available NPUs", DIST_ERROR(ErrCode::VALUE));
    }

    const auto &first = tensors.front();
    if (!torch_npu::utils::is_npu(first) || first.is_sparse()) {
        TORCH_CHECK(false, "Tensors must be NPU and dense", DIST_ERROR(ErrCode::TYPE));
    }

    if (!first.is_contiguous(first.suggest_memory_format())) {
        TORCH_CHECK(false, "Tensors must be contiguous", DIST_ERROR(ErrCode::TYPE));
    }
}

std::vector<at::Tensor> CastOriginFormat(const std::vector<at::Tensor> &inputTensors)
{
    std::vector<at::Tensor> inputTensors_;
    inputTensors_.resize(inputTensors.size());
    size_t index = 0;
    for (auto &tensor : inputTensors) {
        if (at_npu::native::FormatHelper::IsBaseFormatType(tensor)) {
            inputTensors_[index] = tensor;
        } else {
            auto origin_format = torch_npu::NPUBridge::GetNpuStorageImpl(tensor)->npu_desc_.origin_format_;
            inputTensors_[index] = at_npu::native::npu_format_cast(tensor, origin_format);
        }
        index++;
    }
    return inputTensors_;
}

void CheckNpuTensorsDifferentDevices(const std::vector<at::Tensor> &tensors)
{
    if (tensors.size() != 1) {
        TORCH_CHECK(false, "Tensor list mustn't be larger than the number of available NPUs", DIST_ERROR(ErrCode::VALUE));
    }

    const auto &first = tensors.front();
    std::unordered_set<decltype(first.get_device())> usedDevices;
    usedDevices.reserve(tensors.size());

    for (auto &t : tensors) {
        if (!torch_npu::utils::is_npu(t) || t.is_sparse()) {
            TORCH_CHECK(false, "Tensors must be NPU and dense", DIST_ERROR(ErrCode::TYPE));
        }
        if (t.scalar_type() != first.scalar_type()) {
            TORCH_CHECK(false, "Tensors must have identical type", DIST_ERROR(ErrCode::TYPE));
        }
        if (t.sizes() != first.sizes()) {
            TORCH_CHECK(false, "Tensors must have identical size", DIST_ERROR(ErrCode::TYPE));
        }
        if (t.strides() != first.strides()) {
            TORCH_CHECK(false, "Tensors must have identical strides", DIST_ERROR(ErrCode::TYPE));
        }
        if (!t.is_contiguous(t.suggest_memory_format())) {
            TORCH_CHECK(false, "Tensors must be contiguous", DIST_ERROR(ErrCode::TYPE));
        }
        // if (!at_npu::native::FormatHelper::IsBaseFormatType(t) && (t.storage().data_ptr().get() != t.data_ptr())) {
            // TORCH_CHECK(false, "For a tensor of internal format, it's storage_offset must be 0", DIST_ERROR(ErrCode::NOT_SUPPORT));
        // }
        const auto inserted = usedDevices.insert(t.get_device()).second;
        if (!inserted) {
            TORCH_CHECK(false, "Tensors must be on distinct NPU devices", DIST_ERROR(ErrCode::TYPE));
        }
    }
}

uint64_t GetNumelForZBCCL(const at::Tensor &t)
{
    if (!at_npu::native::FormatHelper::IsBaseFormatType(t)) {
        if (t.storage().data_ptr().get() != t.data_ptr()) {
            TORCH_CHECK(false, "For a tensor of internal format, it's storage_offset must be 0", DIST_ERROR(ErrCode::NOT_SUPPORT));
        }
        auto sizes = torch_npu::NPUBridge::GetNpuStorageImpl(t)->npu_desc_.storage_sizes_;
        uint64_t n = 1;
        for (auto s : sizes) {
            n *= s;
        }
        return n;
    }
    return t.numel();
}

zbccl_datatype_t GetZBcclDataType(at::ScalarType type)
{
    try {
        return kScalarTypeToZBcclDataType.at(type);
    } catch (std::out_of_range &e) {
        throw std::runtime_error("Unsupported data type for ZBCCL process group" + DIST_ERROR(ErrCode::NOT_SUPPORT));
    }
}


}
}
}