#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import subprocess
import sys

from setuptools import Extension, find_packages, setup
from torch.utils import cpp_extension

if sys.version_info < (3, 6):
    sys.exit("Sorry, Python >= 3.6 is required for fairseq.")


def write_version_py():
    with open(os.path.join("fairseq", "version.txt")) as f:
        version = f.read().strip()

    # write version info to fairseq/version.py
    with open(os.path.join("fairseq", "version.py"), "w") as f:
        f.write('__version__ = "{}"\n'.format(version))
    return version


version = write_version_py()


with open("README.md") as f:
    readme = f.read()


if sys.platform == "darwin":
    extra_compile_args = ["-stdlib=libc++", "-O3"]
else:
    extra_compile_args = ["-std=c++11", "-O3"]


class NumpyExtension(Extension):
    """Source: https://stackoverflow.com/a/54128391"""

    def __init__(self, *args, **kwargs):
        self.__include_dirs = []
        super().__init__(*args, **kwargs)

    @property
    def include_dirs(self):
        import numpy

        return self.__include_dirs + [numpy.get_include()]

    @include_dirs.setter
    def include_dirs(self, dirs):
        self.__include_dirs = dirs


extensions = [
    NumpyExtension(
        "fairseq.data.data_utils_fast",
        sources=["fairseq/data/data_utils_fast.pyx"],
        language="c++",
        extra_compile_args=extra_compile_args,
    )
]

def do_setup(package_data):
    setup(
        name="foreigner_speech",
        version=version,
        description="Foreigner Speech Recognition Library",
        url="https://github.com/soohyunme/foreigner_speech",
        classifiers=[
            "Intended Audience :: Developers",
            "Programming Language :: Python :: 3",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
        long_description=readme,
        long_description_content_type="text/markdown",
        install_requires=[
            "cffi",
            "cython",
            "hydra-core>=1.0.7,<1.1",
            "omegaconf<2.1",
            "numpy>=1.21.3",
            "regex",
            "sacrebleu>=1.4.12",
            "torch>=1.13",
            "tqdm",
            "bitarray",
            "torchaudio>=0.8.0",
            "scikit-learn",
            "packaging",
        ],
        dependency_links=[],
        packages=find_packages(
            exclude=[
                "examples",
                "examples.*",
            ]
        ),
        package_data=package_data,
        ext_modules=extensions,
        entry_points={
            "console_scripts": [
                "fairseq-eval-lm = fairseq_cli.eval_lm:cli_main",
                "fairseq-generate = fairseq_cli.generate:cli_main",
                "fairseq-hydra-train = fairseq_cli.hydra_train:cli_main",
                "fairseq-interactive = fairseq_cli.interactive:cli_main",
                "fairseq-preprocess = fairseq_cli.preprocess:cli_main",
                "fairseq-score = fairseq_cli.score:cli_main",
                "fairseq-train = fairseq_cli.train:cli_main",
                "fairseq-validate = fairseq_cli.validate:cli_main",
            ],
        },
        zip_safe=False,
    )


def get_files(path, relative_to="fairseq"):
    all_files = []
    for root, _dirs, files in os.walk(path, followlinks=True):
        root = os.path.relpath(root, relative_to)
        for file in files:
            if file.endswith(".pyc"):
                continue
            all_files.append(os.path.join(root, file))
    return all_files


if __name__ == "__main__":
    try:
        # symlink examples into fairseq package so package_data accepts them
        fairseq_examples = os.path.join("fairseq", "examples")
        if "build_ext" not in sys.argv[1:] and not os.path.exists(fairseq_examples):
            os.symlink(os.path.join("..", "examples"), fairseq_examples)

        package_data = {
            "fairseq": (
                get_files(fairseq_examples)
                + get_files(os.path.join("fairseq", "config"))
            )
        }
        do_setup(package_data)
    finally:
        if "build_ext" not in sys.argv[1:] and os.path.islink(fairseq_examples):
            os.unlink(fairseq_examples)