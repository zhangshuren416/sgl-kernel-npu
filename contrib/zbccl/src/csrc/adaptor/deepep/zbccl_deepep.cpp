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

#include <cmath>
#include <pybind11/functional.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/core/npu/NPUEvent.h>
#include <torch_npu/csrc/npu/Event.h>
#include "zbccl_common_includes.h"
#include "zbccl_operations.h"

#include "zbccl_deepep.h"

namespace zbccl {
namespace adaptor {
namespace deep_ep {
constexpr int PADDING_SIZE = 1;
constexpr size_t COMM_NAME_LEN = 128;
constexpr int A3_MAX_HCCS_PEERS = 384;
constexpr int A2_MAX_HCCS_PEERS = 8;

const zbccl_tensor_info_t* transfer_tensor_info(const torch::Tensor &ori_tensor)
{
    thread_local zbccl_tensor_info_t result = {};

    result.data = const_cast<void*>(ori_tensor.data_ptr());
    auto at_type = ori_tensor.scalar_type();
    switch (at_type) {
        case at::ScalarType::Char:
            result.dataType = ZBCCL_DATA_TYPE_INT8;
            break;
        case at::ScalarType::Short:
            result.dataType = ZBCCL_DATA_TYPE_INT16;
            break;
        case at::ScalarType::Int:
            result.dataType = ZBCCL_DATA_TYPE_INT32;
            break;
        case at::ScalarType::Half:
            result.dataType = ZBCCL_DATA_TYPE_FP16;
            break;
        case at::ScalarType::Long:
            result.dataType = ZBCCL_DATA_TYPE_INT64;
            break;
        case at::ScalarType::Bool:
            result.dataType = ZBCCL_DATA_TYPE_UINT8;
            break;
        case at::ScalarType::Float:
            result.dataType = ZBCCL_DATA_TYPE_FP32;
            break;
        case at::ScalarType::Double:
            result.dataType = ZBCCL_DATA_TYPE_FP64;
            break;
        case at::ScalarType::BFloat16:
            result.dataType = ZBCCL_DATA_TYPE_BFP16;
            break;
        default:
            throw std::runtime_error("Unsupported torch scalar type.");
    }

    const auto& sizes = ori_tensor.sizes();
    result.dim = static_cast<uint16_t>(sizes.size());
    if (result.dim > 25) {
        throw std::runtime_error("Tensor dimension exceeds maximum supported (25).");
    }
    for (size_t i = 0; i < sizes.size(); ++i) {
        result.shape[i] = static_cast<uint16_t>(sizes[i]);
    }

    return &result;
}

Buffer::Buffer(int rank, int num_ranks, int64_t num_nvl_bytes, int64_t num_rdma_bytes, bool low_latency_mode, std::string moe_group_name)
    : rank(rank), num_ranks(num_ranks),
    num_nvl_bytes(num_nvl_bytes), num_rdma_bytes(num_rdma_bytes),
    low_latency_mode(low_latency_mode), moe_group_name(moe_group_name)
{
    ZBCCL_ASSERT_S(0 <= rank and rank < num_ranks, "rank check failed:", Z_INVALID_VALUE);
    ZBCCL_CHECK_S(aclrtGetDevice(&device_id) == ACL_SUCCESS, "get device_id failed");
    ZBCCL_CHECK_S(!moe_group_name.empty() and moe_group_name.size() < COMM_NAME_LEN, "moe_group_name check failed:", Z_INVALID_VALUE);

    comm_ = zbccl_comm_get_by_name(moe_group_name.c_str());

    soc_version = op::GetCurrentPlatformInfo().GetSocVersion();
    num_rdma_ranks = 1;
    num_nvl_ranks = num_ranks;
    rdma_rank = rank;
    nvl_rank = rank;
    if (soc_version == op::SocVersion::ASCEND910B) {
        ZBCCL_ASSERT_S(num_ranks < A2_MAX_HCCS_PEERS || num_ranks % A2_MAX_HCCS_PEERS == 0, "num_ranks check failed:", Z_INVALID_VALUE);
        num_rdma_ranks = std::max(1, num_ranks / A2_MAX_HCCS_PEERS);
        num_nvl_ranks = std::min(num_ranks, A2_MAX_HCCS_PEERS);
        rdma_rank = rank / A2_MAX_HCCS_PEERS;
        nvl_rank = rank % A2_MAX_HCCS_PEERS;
        num_max_hccs_peers = A2_MAX_HCCS_PEERS;
    } else {
        num_max_hccs_peers = A3_MAX_HCCS_PEERS;
    }
}

Buffer::~Buffer() noexcept(false) {}

bool Buffer::is_available() const
{
    return available;
}

bool Buffer::is_internode_available() const {
    // Current version does not support internode
    return is_available() and num_ranks > num_max_hccs_peers;
}

std::tuple<torch::Tensor, std::optional<torch::Tensor>, torch::Tensor, torch::Tensor, std::optional<EventHandle>>
Buffer::get_dispatch_layout(const torch::Tensor &topk_idx, int num_experts, std::optional<EventHandle> &previous_event,
                            bool async, bool allocate_on_comm_stream)
{
    ZBCCL_ASSERT_S(topk_idx.dim() == 2, "Layout: check topk_idx dim failed:", Z_INVALID_VALUE);
    ZBCCL_ASSERT_S(topk_idx.is_contiguous(), "Layout: check topk_idx contiguous failed:", Z_INVALID_VALUE);
    ZBCCL_ASSERT_S(num_experts > 0, "Layout: check num_experts failed:", Z_INVALID_VALUE);

    const int num_tokens = static_cast<int>(topk_idx.size(0));
    const int num_topk = static_cast<int>(topk_idx.size(1));

    auto device = topk_idx.device();
    auto num_tokens_per_expert = at::zeros({num_experts}, at::dtype(at::kInt).device(device));
    auto num_tokens_per_rank = at::zeros({num_ranks}, at::dtype(at::kInt).device(device));
    auto is_token_in_rank = at::zeros({num_tokens, num_ranks}, at::dtype(at::kInt).device(device));
    auto send_token_idx = at::zeros({num_tokens, num_topk}, at::dtype(at::kInt).device(device));
    auto num_tokens_per_rdma_rank = std::optional<torch::Tensor>();
    if (is_internode_available()) {
        num_tokens_per_rdma_rank = torch::empty({num_rdma_ranks}, dtype(at::kInt).device(device));
    }

    auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
    int64_t flags = 0;

    auto ret = zbccl_dispatch_normal_layout(transfer_tensor_info(topk_idx),
                                num_tokens, num_experts, num_topk, num_ranks, 
                                transfer_tensor_info(num_tokens_per_rank),
                                transfer_tensor_info(num_tokens_per_expert),
                                transfer_tensor_info(is_token_in_rank),
                                transfer_tensor_info(send_token_idx),
                                acl_stream, flags);

    this->send_token_idx = send_token_idx;
    std::optional<EventHandle> event;

    return {num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert, is_token_in_rank, event};
}

}  // namespace deep_ep
}  // namespace adaptor
}  // namespace zbccl