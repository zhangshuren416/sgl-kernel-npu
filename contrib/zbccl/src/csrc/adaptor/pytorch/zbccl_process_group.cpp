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
#include "zbccl_process_group.h"

namespace zbccl {
namespace backend {

ProcessGroupZBCCL::WorkZBCCL::WorkZBCCL(const std::vector<at::Device> &devices, c10d::OpType opType)
    : Work(-1, opType), devices_(devices), workStartTime_(std::chrono::steady_clock::now())
{}

ProcessGroupZBCCL::WorkZBCCL::~WorkZBCCL() {}

bool ProcessGroupZBCCL::WorkZBCCL::isCompleted()
{
    return true;
}

bool ProcessGroupZBCCL::WorkZBCCL::isSuccess() const
{
    return true;
}

void ProcessGroupZBCCL::WorkZBCCL::synchronizeInternal(std::chrono::milliseconds timeout) {}

bool ProcessGroupZBCCL::WorkZBCCL::wait(std::chrono::milliseconds timeout)
{
    return true;
}

void ProcessGroupZBCCL::WorkZBCCL::synchronize() {}

bool ProcessGroupZBCCL::WorkZBCCL::finishedNPUExecution()
{
    return true;
}

std::vector<at::Tensor> ProcessGroupZBCCL::WorkZBCCL::result()
{
    return *outputs_;
}

void ProcessGroupZBCCL::WorkZBCCL::checkAndThrowException() const {}

void ProcessGroupZBCCL::WorkZBCCL::checkAndSetException() const {}

bool ProcessGroupZBCCL::WorkZBCCL::finishedNPUExecutionInternal() const
{
    return true;
}

c10::intrusive_ptr<c10::ivalue::Future> ProcessGroupZBCCL::WorkZBCCL::getFuture()
{
    return future_;
}

const int64_t ProcessGroupZBCCL::kProcessGroupZBCCLOpTimeoutMillis = 10 * 1000;

ProcessGroupZBCCL::ProcessGroupZBCCL(int rank, int size) : c10d::Backend(rank, size), teamId_(0), store_(nullptr) {}

ProcessGroupZBCCL::ProcessGroupZBCCL(const c10::intrusive_ptr<c10d::Store> &store, int rank, int size, uint32_t teamId)
    : c10d::Backend(rank, size), teamId_(teamId), store_(store)
{}

void ProcessGroupZBCCL::abc()
{
    return;
}

c10::intrusive_ptr<c10d::Work> ProcessGroupZBCCL::allreduce(std::vector<at::Tensor> &tensors,
                                                            const c10d::AllreduceOptions &opts)
{
    return nullptr;
}

c10::intrusive_ptr<c10d::Work> ProcessGroupZBCCL::allgather(std::vector<std::vector<at::Tensor>> &outputTensors,
                                                            std::vector<at::Tensor> &inputTensors,
                                                            const c10d::AllgatherOptions &opts)
{
    return nullptr;
}

c10::intrusive_ptr<c10d::Work> ProcessGroupZBCCL::broadcast(std::vector<at::Tensor> &tensors,
                                                            const c10d::BroadcastOptions &opts)
{
    return nullptr;
}

c10::intrusive_ptr<c10d::Work> ProcessGroupZBCCL::reduce_scatter(std::vector<at::Tensor> &outputTensors,
                                                                 std::vector<std::vector<at::Tensor>> &inputTensors,
                                                                 const c10d::ReduceScatterOptions &opts)
{
    return nullptr;
}

c10::intrusive_ptr<c10d::Backend> ProcessGroupZBCCL::createBackend(const c10::intrusive_ptr<::c10d::Store> &store,
                                                                   int rank, int size,
                                                                   const std::chrono::duration<float> &timeout)
{
    auto backend = c10::make_intrusive<ProcessGroupZBCCL>(rank, size);
    backend->store_ = store;
    return backend;
}

ProcessGroupZBCCL::~ProcessGroupZBCCL() {}

PYBIND11_MODULE(process_group, m)
{
    m.def("createProcessGroupZBCCL", &ProcessGroupZBCCL::createBackend);
}

}  // namespace backend
}  // namespace zbccl