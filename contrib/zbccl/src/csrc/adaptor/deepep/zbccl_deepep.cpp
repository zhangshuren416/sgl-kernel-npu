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
    ZBCCL_CHECK_S(!moe_group_name.empty() and moe_group_name.size() < COMM_NAME_LEN, "moe_group_name check failed:");

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
        ZBCCL_ASSERT_S(num_ranks < A3_MAX_HCCS_PEERS || num_ranks % A3_MAX_HCCS_PEERS == 0, "num_ranks check failed:", Z_INVALID_VALUE);
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

int Buffer::get_num_rdma_ranks() const
{
    return num_rdma_ranks;
}

int Buffer::get_rdma_rank() const
{
    return rdma_rank;
}

std::tuple<torch::Tensor, std::optional<torch::Tensor>, torch::Tensor, torch::Tensor, std::optional<EventHandle>>
Buffer::get_dispatch_layout(const torch::Tensor &topk_idx, int num_experts, std::optional<EventHandle> &previous_event,
                            bool async, bool allocate_on_comm_stream)
{
    ZBCCL_CHECK_S(topk_idx.dim() == 2, "Layout: check topk_idx dim failed:");
    ZBCCL_CHECK_S(topk_idx.is_contiguous(), "Layout: check topk_idx contiguous failed:");
    ZBCCL_CHECK_S(num_experts > 0, "Layout: check num_experts failed:");

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

std::tuple<at::Tensor, std::optional<at::Tensor>, std::optional<at::Tensor>, std::optional<at::Tensor>,
           std::vector<int>, at::Tensor, std::optional<EventHandle>>
Buffer::intranode_dispatch(const at::Tensor &x, const std::optional<at::Tensor> &x_scales,
                       const std::optional<at::Tensor> &topk_idx, const std::optional<at::Tensor> &topk_weights,
                       const std::optional<at::Tensor> &num_tokens_per_rank, const at::Tensor &is_token_in_rank,
                       const std::optional<at::Tensor> &num_tokens_per_expert,
                       int num_worst_tokens, const Config &config, std::optional<EventHandle> &previous_event,
                       bool async, bool allocate_on_comm_stream, bool use_quant)
{
    int num_channels = config.num_sms / 2;
    auto device = x.device();

    zbccl_quant_mode_t quant_mode = use_quant ? QUANT_BF16_2_INT8 : NO_QUANT;
    auto recv_topk_idx = std::optional<at::Tensor>();
    auto recv_topk_weights = std::optional<at::Tensor>();
    // Wait streams
    std::optional<EventHandle> event;

    ZBCCL_CHECK_S(num_tokens_per_rank.has_value(), "num_tokens_per_rank is empty");
    ZBCCL_CHECK_S(num_tokens_per_expert.has_value(), "num_tokens_per_expert is empty");

    // Type checks
    ZBCCL_CHECK_S(num_tokens_per_expert->scalar_type() == at::kInt, "num_tokens_per_expert scalar type is not kInt");
    ZBCCL_CHECK_S(num_tokens_per_rank->scalar_type() == at::kInt, "num_tokens_per_rank scalar type is not kInt");

    // Shape and contiguous checks
    ZBCCL_CHECK_S(x.dim() == 2 and x.is_contiguous(), "x dim not 2 or not comtiguous");
    ZBCCL_CHECK_S(num_tokens_per_expert->dim() == 1 and num_tokens_per_expert->is_contiguous(), "num_tokens_per_expert check failed");
    ZBCCL_CHECK_S(num_tokens_per_expert->size(0) % num_ranks == 0, "num_tokens_per_expert check failed");
    ZBCCL_CHECK_S(num_tokens_per_rank->dim() == 1 and num_tokens_per_rank->is_contiguous(), "num_tokens_per_rank check failed");
    ZBCCL_CHECK_S(num_tokens_per_rank->size(0) == num_ranks, "num_tokens_per_rank check failed");

    auto num_tokens = static_cast<int>(x.size(0)), hidden = static_cast<int>(x.size(1));
    auto num_experts = static_cast<int>(num_tokens_per_expert->size(0));
    ZBCCL_CHECK_S(num_experts > 0, "num_experts check failed");
    auto num_local_experts = static_cast<int>(num_experts / num_ranks);
    auto new_num_tokens_per_expert = num_tokens_per_expert.value();
    auto send_token_idx = this->send_token_idx;

    // Top-k checks
    int num_topk = 0;
    ZBCCL_CHECK_S(topk_idx.has_value(), "topk_idx is empty");
    if (topk_idx.has_value()) {
        num_topk = static_cast<int>(topk_idx->size(1));
        ZBCCL_CHECK_S(num_tokens == topk_idx->size(0), "topk_idx check failed");
        ZBCCL_CHECK_S(topk_idx->dim() == 2 and topk_idx->is_contiguous(), "topk_idx check failed");
    }
    auto expert_ids = topk_idx.value().to(at::kInt);
    int topk_num = static_cast<int>(expert_ids.size(1));

    // FP8 scales checks
    float *x_scales_ptr = nullptr;
    int num_scales = 0, scale_token_stride = 0, scale_hidden_stride = 0;
    if (x_scales.has_value()) {
        ZBCCL_CHECK_S(x.element_size() == 1, "x element size is not 1");
        ZBCCL_CHECK_S(x_scales->scalar_type() == at::kFloat or x_scales->scalar_type() == at::kInt, "x_scales scalar type check failed");
        ZBCCL_CHECK_S(x_scales->dim() == 2, "x_scales shape check failed");
        ZBCCL_CHECK_S(x_scales->size(0) == num_tokens, "x_scales shape check failed");
        num_scales = x_scales->dim() == 1 ? 1 : static_cast<int>(x_scales->size(1));
        x_scales_ptr = static_cast<float *>(x_scales->data_ptr());
        scale_token_stride = static_cast<int>(x_scales->stride(0));
        scale_hidden_stride = static_cast<int>(x_scales->stride(1));
    }

    std::vector<int> num_recv_tokens_per_expert_list;
    // indicates the value type of the output num_recv_tokens_per_expert_list, with a range of [0, 1]
    // 0 means the prefix sum of the number of tokens received by each expert;
    // 1 means the number of tokens received by each expert (default)
    int expert_token_nums_type = get_value_from_env("MOE_EXPERT_TOKEN_NUMS_TYPE", 1);
    ZBCCL_CHECK_S(expert_token_nums_type == 1 or expert_token_nums_type == 0, "expert_token_nums_type is invalid");

    int send_per_group = 1;  // (send_to_expert_num,)
    int send_count = send_per_group * num_experts;
    auto recv_data = torch::empty({num_ranks, num_experts}, at::dtype(at::kInt).device(device));
    auto put_offset = torch::empty({num_experts, num_ranks}, at::dtype(at::kInt).device(device));
    auto recv_tokens_per_expert = torch::empty({num_local_experts}, at::dtype(at::kLong).device(device));
    int64_t total_recv_token = 0;

    auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
    int64_t flags = 0;

    // call notify
    auto notify_ret = zbccl_dispatch_normal_notify(transfer_tensor_info(new_num_tokens_per_expert), 
        send_count, topk_num, 
        transfer_tensor_info(recv_data), 
        &total_recv_token,
        transfer_tensor_info(recv_tokens_per_expert), 
        transfer_tensor_info(put_offset), 
        comm_, acl_stream, flags);

    int num_recv_tokens = (total_recv_token == 0) ? 1 : total_recv_token;  // max recv_tokens in all rank
    auto expandx_out = use_quant ? torch::empty({num_recv_tokens, hidden}, at::dtype(at::kChar).device(device))
                                 : torch::empty({num_recv_tokens, hidden}, x.options());
    auto dynamic_scales_out = use_quant ? torch::empty({num_recv_tokens}, at::dtype(at::kFloat).device(device))
                                        : torch::empty({1}, at::dtype(at::kFloat).device(device));
    // call dispatch
    auto dispatch_ret = zbccl_dispatch_normal(transfer_tensor_info(x),
        transfer_tensor_info(expert_ids),
        transfer_tensor_info(send_token_idx),
        transfer_tensor_info(put_offset),
        num_experts, quant_mode,
        transfer_tensor_info(expandx_out),
        transfer_tensor_info(dynamic_scales_out),
        comm_, acl_stream, flags);

    auto recv_token_per_exp_cpu = recv_tokens_per_expert.to(at::kCPU);
    auto recv_token_per_exp_ptr = recv_token_per_exp_cpu.data_ptr<int64_t>();

    int token_cnt = 0;
    for (int local_e = 0; local_e < num_local_experts; ++local_e) {
        int current_tokens = static_cast<int>(recv_token_per_exp_ptr[local_e]);
        token_cnt = (expert_token_nums_type == 0) ? token_cnt + current_tokens : current_tokens;
        num_recv_tokens_per_expert_list.emplace_back(token_cnt);
    }
    // Return values
    return {expandx_out,
            dynamic_scales_out,
            recv_topk_idx,
            recv_topk_weights,
            num_recv_tokens_per_expert_list,
            put_offset,
            event};
}

std::tuple<torch::Tensor, std::optional<torch::Tensor>, std::optional<EventHandle>> 
Buffer::intranode_combine(const torch::Tensor &x, const torch::Tensor &topk_idx, 
    const std::optional<torch::Tensor> &topk_weights, const torch::Tensor &put_offset)
{
    ZBCCL_CHECK_S(x.dim() == 2 and x.is_contiguous(), "x dim not 2 or not comtiguous");
    ZBCCL_CHECK_S(topk_idx.dim() == 2 and topk_idx.is_contiguous(), "topk_idx dim not 2 or not comtiguous");
    auto recv_x = x;
    auto expert_ids = topk_idx.to(at::kInt);
    auto device = x.device();

    const int num_tokens = static_cast<int>(expert_ids.size(0));
    const int num_topk = static_cast<int>(expert_ids.size(1));
    const int hidden = static_cast<int>(recv_x.size(1));

    ZBCCL_CHECK_S(topk_weights.has_value(), "topk_weights is empty");
    if (topk_weights.has_value()) {
        ZBCCL_CHECK_S(topk_weights->dim() == 2 and topk_weights->is_contiguous(), "topk_weights dim not 2 or not comtiguous");
        ZBCCL_CHECK_S(num_topk == topk_weights->size(1), "topk_weights shape check failed");
        ZBCCL_CHECK_S(topk_weights->scalar_type() == at::kFloat, "topk_weights scalar type check failed");
    }
    auto expert_scales = topk_weights.value();
    auto ep_send_counts = put_offset;
    uint16_t moe_expert_number = static_cast<uint16_t>(put_offset.size(0));
    auto send_token_idx = this->send_token_idx;

    auto combined_x = torch::empty({expert_scales.size(0), hidden}, x.options());
    std::optional<torch::Tensor> recv_topk_weights;
    std::optional<EventHandle> event;

    auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
    int64_t flags = 0;

    // call combine
    int ret = zbccl_combine_normal(transfer_tensor_info(recv_x),
        transfer_tensor_info(ep_send_counts),
        transfer_tensor_info(expert_scales),
        transfer_tensor_info(expert_ids),
        transfer_tensor_info(send_token_idx),
        moe_expert_number,
        transfer_tensor_info(combined_x),
        comm_, acl_stream, flags);

    return {combined_x, recv_topk_weights, event};
}

}  // namespace deep_ep
}  // namespace adaptor
}  // namespace zbccl