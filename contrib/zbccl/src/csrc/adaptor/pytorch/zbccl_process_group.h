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
#ifndef ZBCCL_PROCESS_GROUP_H
#define ZBCCL_PROCESS_GROUP_H

#include <torch/custom_class.h>
#include <torch/csrc/python_headers.h>
#include <c10/util/intrusive_ptr.h>
#include <c10/util/irange.h>
#include <c10d/Backend.hpp>
#include <c10d/Store.hpp>
#include <c10d/Utils.hpp>
#include <c10d/comm.hpp>
#include <c10d/Work.hpp>
#include <pybind11/chrono.h>

#include <torch/csrc/THP.h>
#include <torch/python.h>

#include <torch/csrc/Exceptions.h>
#include <ATen/core/functional.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/utils/object_ptr.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/tensor_flatten.h>
#include <torch/csrc/distributed/c10d/python_comm_hook.h>

#include "torch_npu/csrc/npu/Event.h"

namespace zbccl {
namespace backend {

const std::string ZBCCL_BACKEND_NAME = "zbccl";

class ProcessGroupZBCCL : public c10d::Backend
{
public:
    class WorkZBCCL : public c10d::Work, public std::enable_shared_from_this<WorkZBCCL>
    {
    public:
        // Constructor takes a list of NPU devices to adapt framework, But LCCL support one device only!!!
        explicit WorkZBCCL(const std::vector<at::Device> &devices, c10d::OpType opType);

        ~WorkZBCCL() override;
        // Checks if request has completed. In this specific case of LCCL, it checks
        // if the LCCL operation has completed on the NPU in its own LCCL stream.
        // Non-blocking operation.
        bool isCompleted() override;

        bool isSuccess() const override;

        bool wait(std::chrono::milliseconds timeout) override;

        // Get a Future object that will be marked as completed internally.
        c10::intrusive_ptr<c10::ivalue::Future> getFuture() override;

        // Let current stream wait on the completing of the LCCL work
        // Throws on exceptions. Blocking operation, which will wait for work completion.
        void synchronize() override;

        // Helper function that checks if the LCCL have finished execution on the NPUs
        bool finishedNPUExecution();

        std::vector<at::Tensor> result() override;

    protected:
        // The cached list of NPU devices to operate on. LCCL support one device per rank only
        std::vector<at::Device> devices_;

        // The LCCL communicators used for this work item.
        // std::vector<at_npu::lccl::LcclComm> lcclComms_;

        // multiple runtime devices. These start npu events are needed by desync debugging if enabled.
        std::shared_ptr<std::vector<c10_npu::NPUEvent>> lcclStartEvents_;

        // The end npu events of LCCL operator tracking this work item on multiple npu devices.
        std::shared_ptr<std::vector<c10_npu::NPUEvent>> lcclEndEvents_;

        // Clone of blockingWait_ from ProcessGroupZBCCL.
        bool blockingWait_ = false;

        // Clone of opTimeout_ from ProcessGroupZBCCL.
        std::chrono::milliseconds opTimeout_;

        // Time point representing when the work started.
        std::chrono::time_point<std::chrono::steady_clock> workStartTime_;

    private:
        // Helper function for synchronize
        void synchronizeInternal(std::chrono::milliseconds timeout);

        // Checks for LCCL errors and sets an appropriate exception_ptr.
        void checkAndSetException() const;

        // Checks for LCCL errors and throws an appropriate exception.
        void checkAndThrowException() const;

        // Just checks whether NPU execution has completed, without modifying
        // exception_ptr.
        bool finishedNPUExecutionInternal() const;

        // Store a reference to LCCL collective's outputs, used by result and to
        // give a more descriptive message when representing the Work as a string.
        std::shared_ptr<std::vector<at::Tensor>> outputs_;

        // Reference to the store so that we can write aborted communicators to the store.
        c10::intrusive_ptr<c10d::Store> store_;

        // The future returned by getFuture.
        c10::intrusive_ptr<at::ivalue::Future> future_;

        std::vector<at::Tensor> lazy_destroy_tensors_;

        friend class ProcessGroupZBCCL;
    };

    ProcessGroupZBCCL(int rank, int size);

    ProcessGroupZBCCL(const c10::intrusive_ptr<c10d::Store> &store, int rank, int size, uint32_t teamId);

    ~ProcessGroupZBCCL() override;

    const std::string getBackendName() const override
    {
        return ZBCCL_BACKEND_NAME;
    }

    void abc();

    c10::intrusive_ptr<c10d::Work> allreduce(std::vector<at::Tensor> &tensors,
                                             const c10d::AllreduceOptions &opts = c10d::AllreduceOptions()) override;

    c10::intrusive_ptr<c10d::Work> allgather(std::vector<std::vector<at::Tensor>> &outputTensors,
                                             std::vector<at::Tensor> &inputTensors,
                                             const c10d::AllgatherOptions &opts = c10d::AllgatherOptions()) override;

    c10::intrusive_ptr<c10d::Work> broadcast(std::vector<at::Tensor> &tensors,
                                             const c10d::BroadcastOptions &opts = c10d::BroadcastOptions()) override;

    c10::intrusive_ptr<c10d::Work>
    reduce_scatter(std::vector<at::Tensor> &outputTensors, std::vector<std::vector<at::Tensor>> &inputTensors,
                   const c10d::ReduceScatterOptions &opts = c10d::ReduceScatterOptions()) override;

    static const int64_t kProcessGroupZBCCLOpTimeoutMillis;

    static c10::intrusive_ptr<c10d::Backend> createBackend(const c10::intrusive_ptr<::c10d::Store> &store, int rank,
                                                           int size, const std::chrono::duration<float> &timeout);

    static void ProcessGroupZBCCLConstructor() __attribute__((constructor))
    {
        py::object module = py::module::import("torch.distributed");
        py::object register_backend = module.attr("Backend").attr("register_backend");
        register_backend("zbccl", py::cpp_function(ProcessGroupZBCCL::createBackend));
    }

protected:
    c10::intrusive_ptr<c10d::Store> store_;

private:
    uint32_t teamId_;
};

}  // namespace backend
}  // namespace zbccl

#endif  // ZBCCL_PROCESS_GROUP_H