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

#include <pybind11/pybind11.h>

#include "c10_npu_dma.h"
#include "zbccl_pytorch_process_group.h"
#include "zbccl_def.h"
#include "zbccl_bootstrap.h"
#include "zbccl.h"

namespace py = pybind11;

static zbccl_bootstrap_output_t output;

int32_t zbccl_bootstrap_wrapper(zbccl_bootstrap_options_t &opt)
{
    return zbccl_bootstrap(&opt, &output);
}

void pybind11_definitions(py::module_ &m)
{
    py::enum_<zbccl_bootstrap_type_t>(m, "ZBCCLBootstrapType")
        .value("BOOT_BY_MEMFABRIC", zbccl_bootstrap_type_t::BOOT_BY_MEMFABRIC);

    py::enum_<zbccl_backend_t>(m, "ZBCCLBackendType")
        .value("ZBCCL_ASCEND_NPU", zbccl_backend_t::ZBCCL_ASCEND_NPU);

    py::class_<zbccl_bootstrap_options_t>(m, "ZBCCLBootstrapOption")
        .def(py::init<>())
        .def_readwrite("flags", &zbccl_bootstrap_options_t::flags)
        .def_readwrite("btType", &zbccl_bootstrap_options_t::btType)
        .def_readwrite("worldSize", &zbccl_bootstrap_options_t::worldSize)
        .def_readwrite("rankId", &zbccl_bootstrap_options_t::rankId)
        .def_readwrite("deviceId", &zbccl_bootstrap_options_t::deviceId)
        .def_readwrite("startConfigServer", &zbccl_bootstrap_options_t::startConfigServer)
        .def_readwrite("deviceMemorySize", &zbccl_bootstrap_options_t::deviceMemorySize)
        .def_readwrite("dataOperationType", &zbccl_bootstrap_options_t::dataOperationType)
        .def_readwrite("cclMetaSpaceSize", &zbccl_bootstrap_options_t::cclMetaSpaceSize)
        .def_readwrite("cclGroupCap", &zbccl_bootstrap_options_t::cclGroupCap)
        .def_property("ipPort", [](const zbccl_bootstrap_options_t &opt) {
            return std::string(opt.ipPort, strlen(opt.ipPort));
        }, [](zbccl_bootstrap_options_t &opt, const std::string &ipPort) {
            if (ipPort.size() >= ZBCCL_MAX_IPPORT_LEN) {
                throw std::runtime_error("ipPort is too long");
            }
            std::copy(ipPort.begin(), ipPort.end(), opt.ipPort);
            opt.ipPort[ipPort.size()] = '\0';
        });

    py::class_<zbccl_comm_property_t>(m, "ZBCCLCommProperty")
        .def(py::init<>())
        .def_readwrite("backendType", &zbccl_comm_property_t::backendType)
        .def_readwrite("isWorldGroup", &zbccl_comm_property_t::isWorldGroup)
        .def_readwrite("groupSize", &zbccl_comm_property_t::groupSize)
        .def_readwrite("groupRankId", &zbccl_comm_property_t::groupRankId)
        .def_readwrite("symmetricMetaGva", &zbccl_comm_property_t::symmetricMetaGva)
        .def_property("myGVA", [](const zbccl_comm_property_t &prop) {
            return reinterpret_cast<uintptr_t>(prop.myGVA);
        }, [](zbccl_comm_property_t &prop, uintptr_t myGVA) {
            prop.myGVA = reinterpret_cast<void*>(myGVA);
        })
        .def_property("myMetaGVA", [](const zbccl_comm_property_t &prop) {
            return reinterpret_cast<uintptr_t>(prop.myMetaGVA);
        }, [](zbccl_comm_property_t &prop, uintptr_t myMetaGVA) {
            prop.myMetaGVA = reinterpret_cast<void*>(myMetaGVA);
        })
        .def_property("myMetaGVAForOpParam", [](const zbccl_comm_property_t &prop) {
            return reinterpret_cast<uintptr_t>(prop.myMetaGVAForOpParam);
        }, [](zbccl_comm_property_t &prop, uintptr_t myMetaGVAForOpParam) {
            prop.myMetaGVAForOpParam = reinterpret_cast<void*>(myMetaGVAForOpParam);
        })
        .def_property("myMetaGVAForOpExchange", [](const zbccl_comm_property_t &prop) {
            return reinterpret_cast<uintptr_t>(prop.myMetaGVAForOpExchange);
        }, [](zbccl_comm_property_t &prop, uintptr_t myMetaGVAForOpExchange) {
            prop.myMetaGVAForOpExchange = reinterpret_cast<void*>(myMetaGVAForOpExchange);
        })
        .def_readwrite("sizeOfMetaArea", &zbccl_comm_property_t::sizeOfMetaArea)
        .def_readwrite("sizeOfMetaForOpParam", &zbccl_comm_property_t::sizeOfMetaForOpParam)
        .def_readwrite("sizeOfMetaForAddressExchange", &zbccl_comm_property_t::sizeOfMetaForAddressExchange)
        .def_readwrite("localDeviceMemSize", &zbccl_comm_property_t::localDeviceMemSize)
        .def_property("name", [](const zbccl_comm_property_t &prop) {
            return std::string(prop.name);
        }, [](zbccl_comm_property_t &prop, const std::string &name) {
            if (name.size() >= ZBCCL_COMM_NAME_MAX) {
                throw std::runtime_error("name is too long");
            }
            std::copy(name.begin(), name.end(), prop.name);
            prop.name[name.size()] = '\0';
        });
}

void pybind11_functions(py::module_ &m)
{
    m.def("zbccl_bootstrap", &zbccl_bootstrap_wrapper);
    m.def("zbccl_unbootstrap", &zbccl_unbootstrap);
    m.def("zbccl_set_logger_level", &zbccl_set_logger_level);
    m.def("zbccl_version", &zbccl_version);

    // communicator
    m.def("zbccl_comm_get_global", []() -> uintptr_t {
        return reinterpret_cast<uintptr_t>(zbccl_comm_get_global());
    });
    m.def("zbccl_comm_get_by_name", [](const char* name) ->uintptr_t {
        return reinterpret_cast<uintptr_t>(zbccl_comm_get_by_name(name));
    });
    m.def("zbccl_comm_get_property", [](uintptr_t comm) {
        zbccl_comm_property_t prop;
        zbccl_comm_get_property(reinterpret_cast<zbccl_comm_t>(comm), &prop);
        return prop;
    });
}

void pybind11_bootstrap(py::module_ &m)
{
    pybind11_definitions(m);
    pybind11_functions(m);
}

PYBIND11_MODULE(zbccl, m) {
    m.doc() = "zbccl package";

    auto allocator = m.def_submodule("allocator", "zbccl allocator");
    auto npu_adaptor = m.def_submodule("npu_adaptor", "zbccl npu pytorch adaptor");

    pybind11_allocator(allocator);
    pybind11_adaptor(npu_adaptor);
    pybind11_bootstrap(m);
}