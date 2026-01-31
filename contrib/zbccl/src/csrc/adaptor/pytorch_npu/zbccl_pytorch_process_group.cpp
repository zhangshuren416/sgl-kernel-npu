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
#include "zbccl_common_includes.h"

namespace zbccl {
namespace adaptor {
namespace pytorch_npu {

constexpr int64_t kSynchronizeBusyWaitMillis = 10;

ProcessGroupZBCCL::WorkZBCCL::WorkZBCCL(const std::vector<at::Device> &devices, int rank, c10d::OpType opType)
    : Work(rank, opType), devices_(devices), workStartTime_(std::chrono::steady_clock::now())
{
    zbcclEndEvents_ = std::make_shared<std::vector<c10_npu::NPUEvent>>(devices.size());
    zbcclComms_.resize(devices.size());
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
    for (const auto i: c10::irange(devices_.size())) {
        auto currentStream = c10_npu::getCurrentNPUStream(devices_[i].index());
        // Block the current stream on the zbccl stream
        (*zbcclEndEvents_)[i].block(currentStream);
        ZBCCL_LOG_INFO("Event: block zbccl work is successfully executed, event=" << (*zbcclEndEvents_)[i].event());
    }

    // In case of blocking, wait for the operation to complete.
    if (blockingWait_) {
        // Wait for the operation to complete.
        while (!isCompleted()) {
            auto currentTimepoint = std::chrono::steady_clock::now();
            if (std::chrono::duration_cast<std::chrono::milliseconds>(currentTimepoint - workStartTime_) > opTimeout_) {
                throw std::runtime_error("Operation has exceeded timeout limit!");
            }
            checkAndThrowException();
            std::this_thread::sleep_for(std::chrono::milliseconds(kSynchronizeBusyWaitMillis));
        }
        checkAndThrowException();
    }
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
    if (!c10_npu::NpuSysCtrl::GetInstance().GetInitFlag()) {
        return false;
    }
    try {
        for (const auto i: c10::irange(devices_.size())) {
            if (!(*zbcclEndEvents_)[i].query()) {
                return false;
            }
        }
    } catch (const std::exception &e) {
        if (std::string(e.what()).find("driver shutting down") == std::string::npos) {
            throw std::runtime_error("finish execution internal failed.");
        }
        ZBCCL_LOG_INFO("[Rank " << rank_ << "] Event query failed with exeption: " << e.what());
    }

    return true;
}

c10::intrusive_ptr<c10::ivalue::Future> ProcessGroupZBCCL::WorkZBCCL::getFuture()
{
    return future_;
}

const int64_t ProcessGroupZBCCL::kProcessGroupZBcclOpTimeoutMillis = 10 * 1000;

ProcessGroupZBCCL::ProcessGroupZBCCL(int rank, int size) : c10d::Backend(rank, size), store_(nullptr) {}

ProcessGroupZBCCL::ProcessGroupZBCCL(const c10::intrusive_ptr<c10d::Store> &store,
    int rank, int size, std::chrono::milliseconds timeout) : c10d::Backend(rank, size), store_(store)
{
    auto timeoutMill = timeout * 1000;
    if (timeoutMill > WORKER_MAX_TIMEOUT) {
        timeoutMill = WORKER_MAX_TIMEOUT;
        auto inputTm = static_cast<int>(timeout.count());
        ZBCCL_LOG_WARN("timeout " << inputTm << " exceed, set to default value");
    }
    opTimeout_ = timeoutMill;
}

int32_t ProcessGroupZBCCL::GetZBCCLComm(const std::string &key,
                                        const std::vector<at::Device> &devices,
                                        std::vector<zbccl_comm_t> &zbcclComms)
{
    if (devices.empty()) {
        ZBCCL_LOG_ERROR("input devices is empty.");
        return Z_INVALID_PARAM;
    }

    {
        std::lock_guard<std::mutex> lock(mutext_);
        if (devZBCCLCommMap_.find(key) != devZBCCLCommMap_.end()) {
            zbcclComms = devZBCCLCommMap_[key];
            return Z_OK;
        }
    }
    zbcclComms.resize(devices.size());

    c10_npu::OptionalNPUGuard npuGuard;
    std::vector<c10_npu::NPUStream> streamVal;
    streamVal.reserve(devices.size());

    for (size_t i = 0; i < devices.size(); ++i) {
        npuGuard.set_index(devices[i].index());
        std::string curCommKey = ZBCCL_BACKEND_NAME + "_" + key + "_dev:" + std::to_string(i);

        zbccl_comm_options_t opt;
        opt.backendType = ZBCCL_ASCEND_NPU;
        opt.isWorldGroup = 1;
        opt.groupSize = size_;
        opt.groupRankId = rank_;
        opt.name = const_cast<char *>(curCommKey.c_str());
        auto ret = zbccl_comm_create(&opt, &zbcclComms[i]);
        if (ret != Z_OK || zbcclComms[i] == nullptr) {
            ZBCCL_LOG_ERROR("create comm failed, ret=" << ret << ", rank=" << rank_ << ", size="
                << size_ << ", key=" << curCommKey);
            return Z_CREATE_COMM_FAILED;
        }

        ZBCCL_LOG_DEBUG("create comm success, rank=" << rank_ << ", size=" << size_ << ", key=" << curCommKey);
        streamVal.push_back(c10_npu::getNPUStreamFromPool(devices[i].index()));
    }

    std::lock_guard<std::mutex> lock(mutext_);
    zbcclStreams_.emplace(key, std::move(streamVal));
    zbcclEvents_.emplace(std::piecewise_construct, std::make_tuple(key), std::make_tuple(devices.size()));
    devZBCCLCommMap_.emplace(key, zbcclComms);
    return Z_OK;
}

template<typename Fn, typename PreProcess, typename PostProcess>
c10::intrusive_ptr<c10d::Work> ProcessGroupZBCCL::collective(std::vector<at::Tensor> &inputs,
                                                             std::vector<at::Tensor> &outputs, Fn fn, PreProcess pre,
                                                             PostProcess post, c10d::OpType opType)
{
    const auto devices = GetDeviceList(inputs);
    auto key = GetKeyFromDevices(devices);

    std::vector<zbccl_comm_t> zbcclComms;
    ZBCCL_CHECK_S(GetZBCCLComm(key, devices, zbcclComms) == Z_OK, "get zbccl comm failed.");

    auto &zbcclStreams = zbcclStreams_[key];
    SyncStreams(devices, zbcclEvents_[key], zbcclStreams);

    auto work = c10::make_intrusive<ProcessGroupZBCCL::WorkZBCCL>(devices, rank_, opType);
    work->outputs_ = std::make_shared<std::vector<at::Tensor>>(outputs);

    c10_npu::OptionalNPUGuard npuGuard;
    pre(zbcclStreams, work);

    for (const auto i: c10::irange(inputs.size())) {
        npuGuard.set_index(devices[i].index());
        c10_npu::NPUStream &zbcclStream = zbcclStreams[i];

        // Both `inputs' and `outputs' are created on a worker stream and used in
        // different zbcclStreams.  Hence, both must record the zbcclStream to
        // prevent being freed before the collective finishes.
        //
        // We only record `inputs' here, and leave recording `outputs' to `fn' for
        // operations where `inputs' and `outputs' are not the same.
        //
        // See [Sync Streams].
        c10_npu::NPUCachingAllocator::recordStream(inputs[i].storage().data_ptr(), zbcclStream);        // TODO
    }

    {
        for (const auto i: c10::irange(inputs.size())) {
            npuGuard.set_index(devices[i].index());
            // to avoid to much task pushed to the stream, leading to stream overflow
            // insert sync point fluxLimit(key, i)

            int32_t ret = fn(inputs[i], outputs[i], zbcclStreams[i], zbcclComms[i]);
            ZBCCL_CHECK_S(ret == 0, "zbccl process group fn exec failed");
        }
    }

    post(zbcclStreams, work);
    {
        c10_npu::NPUMultiStreamGuard guard(zbcclStreams);
        work->future_ = c10::make_intrusive<at::ivalue::Future>(c10::ListType::create(c10::TensorType::get()), devices);
        work->future_->markCompleted(at::IValue(*work->outputs_));
    }

    for (size_t i = 0; i < inputs.size(); ++i) {
        c10_npu::NPUStream &zbcclStream = zbcclStreams[i];
        (*(work->zbcclEndEvents_))[i].record(zbcclStream);
        ZBCCL_LOG_DEBUG("Event: record zbccl work is successfully executed, event=" <<
            (*(work->zbcclEndEvents_))[i].event());
        work->zbcclComms_[i] = zbcclComms[i];
    }
    work->blockingWait_ = blockingWait_;
    work->opTimeout_ = opTimeout_;
    return work;
}

c10::intrusive_ptr<c10d::Work> ProcessGroupZBCCL::allreduce(std::vector<at::Tensor> &tensors,
                                                            const c10d::AllreduceOptions &opts)
{
    return nullptr;
}


c10::intrusive_ptr<c10d::Work> ProcessGroupZBCCL::_allgather_base(at::Tensor &outputTensor, at::Tensor &inputTensor,
    const c10d::AllgatherOptions &opts)
{
    if (inputTensor.dtype() != outputTensor.dtype()) {
        ZBCCL_CHECK_S(false, "output tensor must have the same dtype as input tensor");
    }

    if (inputTensor.numel() * size_ != outputTensor.numel()) {
        ZBCCL_CHECK_S(false, "output tensor size must be equal to world_size times input tensor size");
    }

    std::vector<at::Tensor> inputTensors = {inputTensor};
    std::vector<at::Tensor> outputTensors = {outputTensor};
    ZBCCL_CHECK_S(CheckNpuTensorsDifferentDevices(inputTensors) == 0, "check input tensor failed.");
    ZBCCL_CHECK_S(CheckNpuTensorsDifferentDevices(outputTensors) == 0, "check output tenso failed.");

    // // auto inputTensors_ = CastOriginFormat(inputTensors);  // TODO

    return collective(
        inputTensors, outputTensors,
        [&](at::Tensor &input, at::Tensor &output, c10_npu::NPUStream &stream, zbccl_comm_t comm) {
            RECORD_FUNCTION("ZBCCLAllGatherBase", std::vector<c10::IValue>({input}));
            c10_npu::NPUCachingAllocator::recordStream(output.storage().data_ptr(), stream);    // TODO

            void *inputDataPtr = input.data_ptr();
            void *outputDataPtr = output.data_ptr();
            auto numel = GetNumelForZBCCL(input);
            auto zbcclType = GetZBcclDataType(input.scalar_type());

            auto ret = zbccl_all_gather(inputDataPtr, outputDataPtr, numel, zbcclType, comm, stream.stream(false));
            return ret;
        },
        [&](std::vector<c10_npu::NPUStream> &, c10::intrusive_ptr<ProcessGroupZBCCL::WorkZBCCL> &) {},
        [&](std::vector<c10_npu::NPUStream> &, c10::intrusive_ptr<ProcessGroupZBCCL::WorkZBCCL> &) {},
        c10d::OpType::ALLGATHER
    );
}


c10::intrusive_ptr<c10d::Work> ProcessGroupZBCCL::_reduce_scatter_base(at::Tensor &outputTensor, at::Tensor &inputTensor,
    const c10d::ReduceScatterOptions &opts)
{
    if (inputTensor.dtype() != outputTensor.dtype()) {
        ZBCCL_CHECK_S(false, "output tensor must have the same dtype as input tensor");
    }

    if (inputTensor.numel() != outputTensor.numel() * size_) {
        ZBCCL_CHECK_S(false, "input tensor size must be equal to world_size times output tensor size");
    }

    std::vector<at::Tensor> inputTensors = {inputTensor};
    std::vector<at::Tensor> outputTensors = {outputTensor};
    ZBCCL_CHECK_S(CheckNpuTensorsDifferentDevices(inputTensors) == 0, "check input tensor failed.");
    ZBCCL_CHECK_S(CheckNpuTensorsDifferentDevices(outputTensors) == 0, "check output tenso failed.");

    // // auto inputTensors_ = CastOriginFormat(inputTensors);

    return collective(
        inputTensors, outputTensors,
        [&](at::Tensor &input, at::Tensor &output, c10_npu::NPUStream &stream, zbccl_comm_t comm) {
            RECORD_FUNCTION("ZBcclReduceScatterBase", std::vector<c10::IValue>({}));
            c10_npu::NPUCachingAllocator::recordStream(output.storage().data_ptr(), stream);    // TODO

            void *inputDataPtr = input.data_ptr();
            void *outputDataPtr = output.data_ptr();
            auto numel = GetNumelForZBCCL(output);
            auto zbcclType = GetZBcclDataType(input.scalar_type());
            auto zbcclReduceOp = GetZBcclReduceOp(opts.reduceOp);

            auto ret = zbccl_reduce_scatter(inputDataPtr, outputDataPtr, numel, zbcclType, zbcclReduceOp, comm, stream.stream(false));
            return ret;
        },
        [&](std::vector<c10_npu::NPUStream> &, c10::intrusive_ptr<ProcessGroupZBCCL::WorkZBCCL> &) {},
        [&](std::vector<c10_npu::NPUStream> &, c10::intrusive_ptr<ProcessGroupZBCCL::WorkZBCCL> &) {},
        c10d::OpType::REDUCE_SCATTER
    );
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
    auto tm = std::chrono::duration_cast<std::chrono::milliseconds>(timeout);
    auto backend = c10::make_intrusive<ProcessGroupZBCCL>(store, rank, size, tm);
    return backend;
}

ProcessGroupZBCCL::~ProcessGroupZBCCL() {}

}  // namespace pytorch_npu
}  // namespace adaptor
}  // namespace zbccl

void pybind11_adaptor(py::module &m)
{
    m.def("createProcessGroupZBCCL", &zbccl::adaptor::pytorch_npu::ProcessGroupZBCCL::createBackend);
}