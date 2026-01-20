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
#include "zbccl_pytorch_process_group.h"
#include "zbccl_pytorch_util.h"
#include "zbccl_operations.h"
// #include "torch_npu/csrc/core/npu/sys_ctrl/npu_sys_ctrl.h"
// #include "torch_npu/csrc/framework/FormatHelper.h"
// #include "torch_npu/csrc/core/NPUBridge.h"
// #include "torch_npu/csrc/core/npu/NPUGuard.h"
// #include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
// #include "torch_npu/csrc/core/npu/DeviceUtils.h"
// #include "torch_npu/csrc/core/npu/NPUFormat.h"
// #include "torch_npu/csrc/framework/OpCommand.h"

namespace zbccl {
namespace adaptor {
namespace pytorch_npu {

constexpr int64_t kSynchronizeBusyWaitMillis = 10;

ProcessGroupZBCCL::WorkZBCCL::WorkZBCCL(const std::vector<at::Device> &devices, int rank, c10d::OpType opType)
    : Work(rank, opType), devices_(devices), workStartTime_(std::chrono::steady_clock::now())
{
    // zbcclEndEvents_ = std::make_shared<std::vector<c10_npu::NPUEvent>>(devices.size());
}

ProcessGroupZBCCL::WorkZBCCL::~WorkZBCCL() {}

bool ProcessGroupZBCCL::WorkZBCCL::isCompleted()
{
    checkAndSetException();
    return exception() || finishedNPUExecutionInternal();
}

bool ProcessGroupZBCCL::WorkZBCCL::isSuccess() const
{
    if (exception()) {
        return false;
    }
    return finishedNPUExecutionInternal();
}

void ProcessGroupZBCCL::WorkZBCCL::synchronizeInternal(std::chrono::milliseconds timeout)
{
    // for (const auto i: c10::irange(devices_.size())) {
    //     auto currentStream = c10_npu::getCurrentNPUStream(devices_[i].index());
    //     // Block the current stream on the LCCL stream
    //     (*zbcclEndEvents_)[i].block(currentStream);
    //     ASCEND_LOGI("Event: block lccl work is successfully executed, event=%p", (*zbcclEndEvents_)[i].event());
    // }

    // // In case of blocking, wait for the operation to complete.
    // if (blockingWait_) {
    //     // Wait for the operation to complete.
    //     while (!isCompleted()) {
    //         auto currentTimepoint = std::chrono::steady_clock::now();
    //         if (std::chrono::duration_cast<std::chrono::milliseconds>(currentTimepoint - workStartTime_) > opTimeout_) {
    //             throw std::runtime_error("Operation has exceeded timeout limit!");
    //         }
    //         checkAndThrowException();
    //         std::this_thread::sleep_for(std::chrono::milliseconds(kSynchronizeBusyWaitMillis));
    //     }
    //     checkAndThrowException();
    // }
}

bool ProcessGroupZBCCL::WorkZBCCL::wait(std::chrono::milliseconds timeout)
{
    synchronizeInternal(timeout);
    return true;
}

void ProcessGroupZBCCL::WorkZBCCL::synchronize()
{
    synchronizeInternal(kNoTimeout);
}

bool ProcessGroupZBCCL::WorkZBCCL::finishedNPUExecution()
{
    checkAndSetException();
    return finishedNPUExecutionInternal();
}

std::vector<at::Tensor> ProcessGroupZBCCL::WorkZBCCL::result()
{
    return *outputs_;
}

void ProcessGroupZBCCL::WorkZBCCL::checkAndThrowException() const
{
    // Set the appropriate exception if found.
    checkAndSetException();

    // Throw an exception, only if we have a valid exception.
    if (exception()) {
        std::rethrow_exception(exception());
    }
}

void ProcessGroupZBCCL::WorkZBCCL::checkAndSetException() const
{
    if (exception()) {
        // We already have an exception.
        return;
    }
}

bool ProcessGroupZBCCL::WorkZBCCL::finishedNPUExecutionInternal() const
{
    // // If in the Finalize, should not query event
    // if (!c10_npu::NpuSysCtrl::GetInstance().GetInitFlag()) {
    //     return false;
    // }
    // try {
    //     for (const auto i: c10::irange(devices_.size())) {
    //         // Checking the work's corresponding ASCEND events' status
    //         if (!(*zbcclEndEvents_)[i].query()) {
    //             return false;
    //         }
    //     }
    // } catch (const std::exception &e) {
    //     if (std::string(e.what()).find("driver shutting down") == std::string::npos) {
    //         throw std::runtime_error(DIST_ERROR(ErrCode::INTERNAL));
    //     }
    //     LOG(INFO) << "[Rank " << rank_ << "] Event query failed with exception: " << e.what();
    // }

    // return true;
}

c10::intrusive_ptr<c10::ivalue::Future> ProcessGroupZBCCL::WorkZBCCL::getFuture()
{
    return future_;
}

const int64_t ProcessGroupZBCCL::kProcessGroupZBcclOpTimeoutMillis = 10 * 1000;

ProcessGroupZBCCL::ProcessGroupZBCCL(int rank, int size) : c10d::Backend(rank, size), teamId_(0), store_(nullptr) {}

ProcessGroupZBCCL::ProcessGroupZBCCL(const c10::intrusive_ptr<c10d::Store> &store, int rank, int size, uint32_t teamId)
    : c10d::Backend(rank, size), teamId_(teamId), store_(store)
{}


template<typename Fn, typename PreProcess, typename PostProcess>
c10::intrusive_ptr<c10d::Work> ProcessGroupZBCCL::collective(std::vector<at::Tensor> &inputs,
                                                             std::vector<at::Tensor> &outputs, Fn fn, PreProcess pre,
                                                             PostProcess post, c10d::OpType opType)
{
    // const auto devices = GetDeviceList(inputs);
    // auto key = GetKeyFromDevices(devices);

    // // std::vector<at_npu::lccl::LcclComm> lcclComms;
    // // lcclComms = getLCCLComm(key, devices);

    // // Used many times below, so we stash the unordered_map lookup
    // auto &zbcclSteams = zbcclStreams_[key];
    // // First let LCCL streams wait for input tensors allocation streams
    // SyncStreams(devices, zbcclEvents_[key], zbcclSteams);

    // // Work itself will create the events on all NPUs of tensors
    // auto work = c10::make_intrusive<ProcessGroupZBCCL::WorkZBCCL>(devices, rank_, opType);
    // // Store references to outputs to be used by WorkLCCL::result and operator<<.
    // work->outputs_ = std::make_shared<std::vector<at::Tensor>>(outputs);

    // // c10_npu::OptionalNPUGuard npuGuard;
    // pre(zbcclSteams, work);

    // for (const auto i: c10::irange(inputs.size())) {
    //     // npuGuard.set_index(devices[i].index());
    //     c10_npu::NPUStream &zbcclStream = zbcclSteams[i];

    //     // Both `inputs' and `outputs' are created on a worker stream and used in
    //     // different zbcclSteams.  Hence, both must record the zbcclStream to
    //     // prevent being freed before the collective finishes.
    //     //
    //     // We only record `inputs' here, and leave recording `outputs' to `fn' for
    //     // operations where `inputs' and `outputs' are not the same.
    //     //
    //     // See [Sync Streams].
    //     // c10_npu::NPUCachingAllocator::recordStream(inputs[i].storage().data_ptr(), zbcclStream);        // TODO
    // }
    // {
    //     for (const auto i: c10::irange(inputs.size())) {
    //         // npuGuard.set_index(devices[i].index());
    //         // to avoid to much task pushed to the stream, leading to stream overflow
    //         // insert sync point fluxLimit(key, i)

    //         c10_npu::NPUStream &zbcclStream = zbcclSteams[i];
    //         auto ret = fn(inputs[i], outputs[i], zbcclStream);
    //         TORCH_CHECK(ret == 0, "ZBCCL function error:", opTypeToString(opType).c_str(), ", error code is", ret, "\n");
    //     }
    // }
    // post(zbcclSteams, work);
    // {
    //     c10_npu::NPUMultiStreamGuard guard(zbcclSteams);
    //     work->future_ = c10::make_intrusive<at::ivalue::Future>(c10::ListType::create(c10::TensorType::get()), devices);
    //     work->future_->markCompleted(at::IValue(*work->outputs_));
    // }

    // for (size_t i = 0; i < inputs.size(); ++i) {
    //     c10_npu::NPUStream &zbcclStream = zbcclStreams_[key][i];
    //     (*(work->zbcclEndEvents_))[i].record(zbcclStream);
    //     ASCEND_LOGI("Event: record lccl work is successfully executed, event=%p", (*(work->zbcclEndEvents_))[i].event());
    //     // work->lcclComms_[i] = lcclComms[i];
    // }
    // work->blockingWait_ = blockingWait_;
    // work->opTimeout_ = opTimeout_;
    // return work;
    return nullptr;
}

template<typename Fn>
c10::intrusive_ptr<c10d::Work> ProcessGroupZBCCL::collective(std::vector<at::Tensor> &inputs,
                                                             std::vector<at::Tensor> &outputs, Fn fn,
                                                             c10d::OpType opType)
{
    // return collective(
    //     inputs, outputs, fn,
    //     [](std::vector<c10_npu::NPUStream> &, c10::intrusive_ptr<ProcessGroupZBCCL::WorkZBCCL> &) {
    //     },
    //     [](std::vector<c10_npu::NPUStream> &, c10::intrusive_ptr<ProcessGroupZBCCL::WorkZBCCL> &) {
    //     },
    //     opType);
    return nullptr;
}

int32_t ProcessGroupZBCCL::CreateZBCCLComm(int rank, int size)
{
    return 0;
}

c10::intrusive_ptr<c10d::Work> ProcessGroupZBCCL::allreduce(std::vector<at::Tensor> &tensors,
                                                            const c10d::AllreduceOptions &opts)
{
    return nullptr;
}


c10::intrusive_ptr<c10d::Work> ProcessGroupZBCCL::_allgather_base(at::Tensor &outputTensor, at::Tensor &inputTensor,
    const c10d::AllgatherOptions &opts)
{
    return nullptr;
    // if (inputTensor.dtype() != outputTensor.dtype()) {
    //     TORCH_CHECK(false, "output tensor must have the same type as input tensor", DIST_ERROR(ErrCode::PARAM));
    // }

    // if (inputTensor.numel() * size_ != outputTensor.numel()) {
    //     TORCH_CHECK(false, "output tensor size must be equal to world_size times input tensor size", DIST_ERROR(ErrCode::PARAM));
    // }

    // std::vector<at::Tensor> inputTensors = {inputTensor};
    // std::vector<at::Tensor> outputTensors = {outputTensor};
    // // CheckNpuTensorsDifferentDevices(inputTensors);
    // // CheckNpuTensorsDifferentDevices(outputTensors);

    // // auto inputTensors_ = CastOriginFormat(inputTensors);

    // return collective(
    //     inputTensors, outputTensors,
    //     [&](at::Tensor &input, at::Tensor &output, c10_npu::NPUStream &stream) {
    //         RECORD_FUNCTION("ZBcclAllgatherBase", std::vector<c10::IValue>({input}));
    //         // if (c10_npu::option::OptionsManager::GetMultiStreamMemoryReuse() != c10_npu::option::AVOID_RECORD_STREAM) {
    //         //     c10_npu::NPUCachingAllocator::recordStream(output.storage().data_ptr(), stream);
    //         // }

    //         auto inputDataPtr = input.data_ptr();
    //         auto outputDataPtr = output.data_ptr();
    //         auto numel = GetNumelForZBCCL(input);
    //         auto zbcclType = GetZBcclDataType(input.scalar_type());
    //         zbccl_comm_t comm = nullptr;
    //         // auto zbccl_call = [inputDataPtr, outputDataPtr, numel, zbcclType, comm, stream]() -> int {
    //         // auto ret = zbccl_all_gather(inputDataPtr, outputDataPtr, numel, zbcclType, comm, stream.stream(false));
    //         // return ret;
    //         // };
    //         // at_npu::native::OpCommand::RunOpApiV2("ZBcclAllgather", zbccl_call);
    //         return 0;
    //     },
    //     [&](std::vector<c10_npu::NPUStream> &, c10::intrusive_ptr<ProcessGroupZBCCL::WorkZBCCL> &) {},
    //     [&](std::vector<c10_npu::NPUStream> &, c10::intrusive_ptr<ProcessGroupZBCCL::WorkZBCCL> &) {},
    //     c10d::OpType::ALLGATHER
    // );
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
    std::cout << "in c++ create pg" << std::endl;
    backend->store_ = store;
    return backend;
}

// c10::intrusive_ptr<c10d::Backend> ProcessGroupZBCCL::createBackend(::c10d::DistributedBackendOptions &options,
//                                                        ProcessGroupZBCCL::Options &zbcclOpt)
// {
//     auto backend = c10::make_intrusive<ProcessGroupZBCCL>(0, 0);
//     // backend->store_ = store;
//     std::cout << "in c++ " << std::endl;
//     return backend;
// }

ProcessGroupZBCCL::~ProcessGroupZBCCL() {}

}  // namespace pytorch_npu
}  // namespace adaptor
}  // namespace zbccl

void pybind11_adaptor(py::module &m)
{
    m.def("createProcessGroupZBCCL", &zbccl::adaptor::pytorch_npu::ProcessGroupZBCCL::createBackend);
}