# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
# MemFabric_Hybrid is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
import logging
import os
import glob
from pathlib import Path
import sysconfig
import subprocess
import shutil

import setuptools
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

import torch
import torch_npu

logger = logging.getLogger(__name__)


def _find_ascend_home():
    """
    Find the ASCEND toolkit home directory.
    It prioritizes the ASCEND_TOOLKIT_HOME environment variable.
    If not set, it falls back to the common default installation path:
    /usr/local/Ascend/ascend-toolkit/latest
    """
    home = os.environ.get("ASCEND_TOOLKIT_HOME")
    if home:
        return home
    default_home = "/usr/local/Ascend/ascend-toolkit/latest"
    if os.path.isdir(default_home):
        return default_home
    maybe = "/usr/local/Ascend/ascend-toolkit"
    latest = os.path.join(maybe, "latest")
    return latest if os.path.isdir(latest) else default_home


def _find_sheme_home():
    home = os.environ.get("SHMEM_HOME_PATH")
    if home:
        return home
    default_home = "/usr/local/Ascend/shmem/latest"
    return default_home


def _find_python_include():
    return sysconfig.get_path('include')

def _get_version(version_dir):
    with open(f"{version_dir}/VERSION", "r", encoding="utf-8") as f:
        version = f.read().strip()
        return version

ascend_home = Path(_find_ascend_home()).resolve()
shmem_home = Path(_find_sheme_home()).resolve()
python_include_dir = Path(_find_python_include()).resolve()
torch_dir = Path(os.path.dirname(torch.__file__)).resolve()
torch_npu_dir = Path(os.path.dirname(torch_npu.__file__)).resolve()
repo_root = Path(__file__).parent.parent.parent.parent.parent  # sgl-kernel-npu/
zbccl_root = repo_root / "contrib/zbccl/"
versoin = _get_version(version_dir=zbccl_root)

# allocator compile inputs
include_dirs = [
    f"{python_include_dir}",
    f"{ascend_home}/include",
    f"{ascend_home}/include/experiment/runtime/runtime/",
    f"{torch_npu_dir}/include",
    f"{torch_dir}/",
    f"{torch_dir}/include",
    f"{torch_dir}/include/torch/csrc/api/include",
    f"{torch_dir}/include/torch/csrc/distributed/",
    f"{torch_dir}/include/torch/csrc/utils/",
    f"{torch_dir}/include/c10/util/",
    f"{zbccl_root}/",
    f"{zbccl_root}/third_party/ska",
    f"{zbccl_root}/third_party/mstx",
    f"{zbccl_root}/src/include",
    f"{zbccl_root}/src/csrc/",
    f"{zbccl_root}/src/csrc/ccl",
    f"{zbccl_root}/src/csrc/ccl/npu",
    f"{zbccl_root}/src/csrc/common",
    f"{zbccl_root}/src/csrc/dma",
    f"{zbccl_root}/src/csrc/sma",
    f"{zbccl_root}/src/csrc/under_api/cann",
    f"{zbccl_root}/src/csrc/under_api/memfabric",
    f"{zbccl_root}/src/csrc/bootstrap",
    f"{zbccl_root}/src/csrc/bootstrap/memory",
    f"{zbccl_root}/src/csrc/bootstrap/memory/memfabric",
    f"{zbccl_root}/src/csrc/bootstrap/memory/aclshmem",
    f"{zbccl_root}/src/csrc/adaptor/pytorch_npu/",
    f"{zbccl_root}/src/csrc/adaptor/deepep/",
]

library_dirs = [
    f"{torch_dir}/lib",
    f"{torch_npu_dir}/lib",
    f"{ascend_home}/lib64",
    sysconfig.get_config_var("LIBDIR"),
    f"{repo_root}/output/",
]

csrc_dir = repo_root / "contrib" / "zbccl" / "src" / "csrc"
sources = ([f"{csrc_dir}/zbccl_pybind.cpp"] + \
           glob.glob(str(csrc_dir / "dma" / "*.cpp")) + \
           glob.glob(str(csrc_dir / "sma" / "*.cpp")) + \
           glob.glob(str(csrc_dir / "adaptor" / "pytorch_npu" / "*.cpp")) + \
           glob.glob(str(csrc_dir / "adaptor" / "deepep" / "*.cpp")))

libraries = ["torch", "torch_npu", "c10", "torch_python", "tiling_api", "platform", "opapi", "zbccl_core", "zbccl_kernel"]

logger.warning(f"Using ASCEND_TOOLKIT_HOME at: {ascend_home}")
logger.warning(f"{include_dirs=}")
logger.warning(f"{sources=}")
logger.warning(f"{library_dirs=}")
logger.warning(f"{libraries=}")

extra_compile_args = ["-std=c++17", "-hno-unused-parameter", "-lno-unused-function", "-Wno-unused-function",
                      "-Wunused-value", "-Wcast-align",
                      "-Wcast-qual", "-Winvalid-pch", "-Wwrite-strings", "-Wsign-compare", "-Wextra",
                      "-O3", "-fvisibility-inlines-hidden", "-fstack-protector-strong",
                      "-Wl,-z,noexecstack", "-Wl,-z,relro", "-Wl,-z,now", "-fPIE", "-fPIC",
                      "-ftrapv"]  # "-fvisibility=hidden"
common_macros = []

class CustomBuildExtension(BuildExtension):
    def build_base_zbccl(self):
        # make dir
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = Path(cur_dir).parent.parent.parent.parent
        build_dir = os.path.join(f"{root_dir}", "build")
        output_dir = os.path.join(f"{root_dir}", "output")
        shutil.rmtree(build_dir, ignore_errors=True)
        shutil.rmtree(output_dir, ignore_errors=True)
        os.makedirs(build_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        print(f"make build dir:{build_dir}, output dir:{output_dir}")

        # cmake
        cmake_cmd = [
            "cmake",
            "..",
            "-DSOC_VERSION=Ascend910_9382",
            "-DBUILD_ZBCCL_MODULE_UT=OFF",
            "-DCMAKE_BUILD_TYPE=Debug",
            "-DDISABLE_ADAPTOR_COMPILE=ON",
            "-DDISABLE_ALLOCATOR_COMPILE=ON"
        ]
        result = subprocess.run(cmake_cmd, cwd=build_dir)
        if result.returncode != 0:
            print(f"python cmake exec failed ret code {result.returncode}, msg {result.stderr}")
            raise RuntimeError("cmake exec failed")
        else:
            print("python cmake exec success")

        # make
        make_cmd = [
            "make",
            "-j9"
        ]
        result = subprocess.run(make_cmd, cwd=build_dir)
        if result.returncode != 0:
            print(f"python make exec failed ret code {result.returncode}, msg {result.stderr}")
            raise RuntimeError("make exec failed")
        else:
            print("python make exec success")

        # copy
        static_output = glob.glob(f"{repo_root}/**/*.a", recursive=True)
        for x in static_output:
            static_name = os.path.basename(x)
            dst = f"{output_dir}/{static_name}"
            shutil.copy2(x, dst)
            print(f"copy {x} to {dst}")

    def run(self):
        self.build_base_zbccl()
        super().run()


setup(
    name="zbccl",
    version=versoin,
    ext_modules=[
        CppExtension(
            name="zbccl.zbccl",  # TORCH_EXTENSION_NAME
            sources=sources,
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            libraries=libraries,
            define_macros=[
                *common_macros,
            ],
            extra_compile_args=extra_compile_args,
            cxx_std=17
        )
    ],
    python_requires=">=3.10",
    packages=setuptools.find_packages(
        include=["zbccl", "zbccl.*"]
    ),
    cmdclass={'build_ext': CustomBuildExtension},
)
