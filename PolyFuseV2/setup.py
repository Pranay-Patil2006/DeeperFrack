import os
import sys
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

class get_pybind_include(object):
    """Helper class to determine the pybind11 include path
    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked. """
    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)

ext_modules = [
    Extension(
        'timeloop_pybind',
        ['polyfuse/cpp_parsers/timeloop_pybind.cpp'],
        include_dirs=[
            get_pybind_include(),
            get_pybind_include(user=True),
            '/opt/timeloop/include'
        ],
        libraries=['timeloop-model', 'yaml-cpp'],
        library_dirs=['/opt/timeloop/lib'],
        extra_compile_args=['-O3', '-Wall', '-std=c++17', '-fPIC'],
        extra_link_args=['-Wl,-rpath,/opt/timeloop/lib'],
        language='c++'
    ),
]

setup(
    name='timeloop_pybind',
    version='0.1.0',
    author='PolyFuse',
    description='A native pybind11 wrapper for Timeloop workload parsing',
    ext_modules=ext_modules,
    setup_requires=['pybind11>=2.5.0'],
    install_requires=['pybind11>=2.5.0'],
    zip_safe=False,
)
