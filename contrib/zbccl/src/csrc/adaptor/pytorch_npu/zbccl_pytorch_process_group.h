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
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <torch/csrc/THP.h>
#include <torch/python.h>

#include <torch/csrc/Exceptions.h>
#include <ATen/core/functional.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/utils/object_ptr.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/tensor_flatten.h>
#include <torch/csrc/distributed/c10d/python_comm_hook.h>

#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/core/npu/NPUEvent.h>
#include <torch_npu/csrc/npu/Event.h>

#include "zbccl_common_includes.h"

namespace zbccl {
namespace adaptor {
namespace pytorch_npu {

const std::string ZBCCL_BACKEND_NAME = "zbccl";
constexpr std::chrono::milliseconds WORKER_MAX_TIMEOUT{600000};
using RSOptions = c10d::ReduceScatterOptions;
using MilliSeconds = std::chrono::milliseconds;

class ProcessGroupZBCCL : public c10d::Backend
{
public:
    class WorkZBCCL : public c10d::Work, public std::enable_shared_from_this<WorkZBCCL>
    {
    public:
        // Constructor takes a list of NPU devices to adapt framework, But zbccl support one device only!!!
        explicit WorkZBCCL(const std::vector<at::Device> &devices, int rank, c10d::OpType opType);

        ~WorkZBCCL() override;
        // Checks if request has completed. In this specific case of zbccl, it checks
        // if the zbccl operation has completed on the NPU in its own zbccl stream.
        // Non-blocking operation.
        bool isCompleted() override;

        bool isSuccess() const override;

        bool wait(std::chrono::milliseconds timeout) override;

        // Get a Future object that will be marked as completed internally.
        c10::intrusive_ptr<c10::ivalue::Future> getFuture() override;

        // Let current stream wait on the completing of the zbccl work
        // Throws on exceptions. Blocking operation, which will wait for work completion.
        void synchronize() override;

        // Helper function that checks if the zbccl have finished execution on the NPUs
        bool finishedNPUExecution();

        std::vector<at::Tensor> result() override;

    protected:
        // The cached list of NPU devices to operate on. zbccl support one device per rank only
        std::vector<at::Device> devices_;

        // The zbccl communicators used for this work item.
        std::vector<zbccl_comm_t> zbcclComms_;

        // multiple runtime devices. These start npu events are needed by desync debugging if enabled.
        // std::shared_ptr<std::vector<c10_npu::NPUEvent>> zbcclStartEvents_;

        // The end npu events of zbccl operator tracking this work item on multiple npu devices.
        std::shared_ptr<std::vector<c10_npu::NPUEvent>> zbcclEndEvents_;

        // Clone of blockingWait_ from ProcessGroupZBCCL.
        bool blockingWait_ = false;

        // Clone of opTimeout_ from ProcessGroupZBCCL.
        std::chrono::milliseconds opTimeout_;

        // Time point representing when the work started.
        std::chrono::time_point<std::chrono::steady_clock> workStartTime_;

    private:
        // Helper function for synchronize
        void synchronizeInternal(std::chrono::milliseconds timeout);

        // Checks for zbccl errors and sets an appropriate exception_ptr.
        void checkAndSetException() const;

        // Checks for zbccl errors and throws an appropriate exception.
        void checkAndThrowException() const;

        // Just checks whether NPU execution has completed, without modifying
        // exception_ptr.
        bool finishedNPUExecutionInternal() const;

        // Store a reference to zbccl collective's outputs, used by result and to
        // give a more descriptive message when representing the Work as a string.
        std::shared_ptr<std::vector<at::Tensor>> outputs_;

        // Reference to the store so that we can write aborted communicators to the store.
        c10::intrusive_ptr<c10d::Store> store_;

        // The future returned by getFuture.
        c10::intrusive_ptr<at::ivalue::Future> future_;

        std::vector<at::Tensor> lazy_destroy_tensors_;

        friend class ProcessGroupZBCCL;
    };

    struct Options : c10d::Backend::Options {
        explicit Options(bool isHighPriorityStream = false);

        static c10::intrusive_ptr<Options> create(bool isHigh = false, MilliSeconds tm = WORKER_MAX_TIMEOUT)
        {
            return c10::make_intrusive<Options>(isHigh);
        }

        MilliSeconds opTimeout;

        bool isHighPriorityStream;

        std::vector<uint32_t> globalRanksInGroup;

        std::string groupId;
    };

    ProcessGroupZBCCL(
        const c10::intrusive_ptr<c10d::Store>& store,
        int rank,
        int size,
        c10::intrusive_ptr<Options> options = Options::create());

    ~ProcessGroupZBCCL() override;

    const std::string getBackendName() const override
    {
        return ZBCCL_BACKEND_NAME;
    }

    c10::intrusive_ptr<c10d::Work> allreduce(std::vector<at::Tensor> &tensors,
                                             const c10d::AllreduceOptions &opts = c10d::AllreduceOptions()) override;

    c10::intrusive_ptr<c10d::Work> _allgather_base(at::Tensor &output, at::Tensor &input,
                                                   const c10d::AllgatherOptions &opt = c10d::AllgatherOptions());

    c10::intrusive_ptr<c10d::Work> allgather(std::vector<std::vector<at::Tensor>> &outputTensors,
                                             std::vector<at::Tensor> &inputTensors,
                                             const c10d::AllgatherOptions &opts = c10d::AllgatherOptions()) override;

    c10::intrusive_ptr<c10d::Work> broadcast(std::vector<at::Tensor> &tensors,
                                             const c10d::BroadcastOptions &opts = c10d::BroadcastOptions()) override;

    c10::intrusive_ptr<c10d::Work> reduce_scatter(std::vector<at::Tensor> &outputTensors,
                                                  std::vector<std::vector<at::Tensor>> &inputTensors,
                                                  const RSOptions &opts = RSOptions()) override;

    c10::intrusive_ptr<c10d::Work> _reduce_scatter_base(at::Tensor &output, at::Tensor &input,
                                                        const RSOptions &opts = RSOptions()) override;

    std::string getZBCCLCommName() noexcept;

protected:
    bool blockingWait_ = false;
    std::chrono::milliseconds opTimeout_;
    c10::intrusive_ptr<c10d::Store> store_;
    std::unordered_map<std::string, std::vector<c10_npu::NPUStream>> zbcclStreams_;
    std::unordered_map<std::string, std::vector<c10_npu::NPUEvent>> zbcclEvents_;
    std::mutex mutext_;
    c10::intrusive_ptr<Options> options_;


private:
    zbccl_comm_t groupComm_{nullptr};
    std::string groupName_;
    int myWorldRank_;
    static std::atomic<uint64_t> groupCounter_;

private:
    template <typename Fn, typename PreProcess, typename PostProcess>
    c10::intrusive_ptr<c10d::Work> collective(
        std::vector<at::Tensor>& input,
        std::vector<at::Tensor>& output,
        Fn fn,
        PreProcess pre,
        PostProcess post,
        c10d::OpType opType);

    uint64_t GetNextGroupCounter() noexcept;

    int32_t PrepareCommunicator(int rank, int size) noexcept;

    std::string ConstructCommName() noexcept;

    int32_t PrepareResources(const std::vector<at::Device> &devs) noexcept;
};

}  // namespace pytorch_npu
}  // namespace adaptor
}  // namespace zbccl

#endif  // ZBCCL_PROCESS_GROUP_H