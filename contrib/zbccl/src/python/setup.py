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
import shutil
import glob
from pathlib import Path
import sysconfig

import setuptools
from setuptools import setup

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


ascend_home = Path(_find_ascend_home()).resolve()
shmem_home = Path(_find_sheme_home()).resolve()
python_include_dir = Path(_find_python_include()).resolve()
torch_dir = Path(os.path.dirname(torch.__file__)).resolve()
torch_npu_dir = Path(os.path.dirname(torch_npu.__file__)).resolve()
repo_root = Path(__file__).resolve().parents[4]  # sgl-kernel-npu/


include_dirs = [
    str(python_include_dir),
    str((ascend_home / "include").resolve()),
    str((torch_npu_dir / "include").resolve()),
    str((torch_dir / "include").resolve()),
    str((torch_dir / "include/torch/csrc/api/include").resolve()),
    str((shmem_home / "shmem/include").resolve()),
    str((repo_root / "contrib/zbccl").resolve()),
    str((repo_root / "contrib/zbccl/src/include").resolve()),
    str((repo_root / "contrib/zbccl/src/csrc/ccl").resolve()),
    str((repo_root / "contrib/zbccl/src/csrc/common").resolve()),
    str((repo_root / "contrib/zbccl/src/csrc/dma").resolve()),
    str((repo_root / "contrib/zbccl/src/csrc/sma").resolve())
]

library_dirs = [
    str((torch_dir / "lib").resolve()),
    str((torch_npu_dir / "lib").resolve()),
    str((shmem_home / "shmem/lib").resolve())
]

csrc_dir = repo_root / "contrib" / "zbccl" / "src" / "csrc"
source_dirs = glob.glob(str(csrc_dir / "ccl" / "*.cpp")) + \
    glob.glob(str(csrc_dir / "common" / "*.cpp")) + \
    glob.glob(str(csrc_dir / "dma" / "*.cpp")) + \
    glob.glob(str(csrc_dir / "sma" / "*.cpp")) + \
    glob.glob(str(csrc_dir / "*.cpp"))

logger.warning(f"Using ASCEND_TOOLKIT_HOME at: {ascend_home}")
logger.warning(f"Using SHMEM_HOME_PATH at: {shmem_home}")
logger.warning(f"Include dirs: {include_dirs}")
logger.warning(f"Library dirs: {library_dirs}")


extra_compile_args = ["-std=c++17", "-hno-unused-parameter", "-lno-unused-function", "-Wunused-value", "-Wcast-align",
                      "-Wcast-qual", "-Winvalid-pch", "-Wwrite-strings", "-Wsign-compare", "-Wextra",
                      "-O3", "-fvisibility-inlines-hidden", "-fstack-protector-strong",
                      "-Wl,-z,noexecstack", "-Wl,-z,relro", "-Wl,-z,now", "-fPIE", "-fPIC", "-ftrapv"]  # "-fvisibility=hidden"
common_macros = []

setup(
    name="zbccl",
    version="0.0.1",
    ext_modules=[
        setuptools.Extension(
            "zbccl.lib.libzbccl",
            sources=source_dirs,
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            # CUDA -> ACL
            libraries=["torch", "torch_npu", "shmem"],
            define_macros=[
                *common_macros,
            ],
            extra_compile_args=extra_compile_args,
            py_limited_api=True,
            language="c++"
        )
    ],
    python_requires=">=3.10",
    packages=setuptools.find_packages(
        include=["zbccl", "zbccl.*"]
    ),
)
