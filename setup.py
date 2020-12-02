from __future__ import division, print_function, absolute_import

import copy
import os
import platform
import sys


# note that this package depends on NumPy's distutils. It also
# piggy-backs on NumPy configuration in order to link against the
# appropriate BLAS/LAPACK implementation.
from numpy.distutils.command.build_ext import build_ext
from numpy.distutils.core import setup, Extension
from numpy.distutils import system_info as np_sys_info
from numpy.distutils import misc_util as np_misc_util
from setup_helpers import add_flag_checking

import versioneer


BASE_PATH = 'gulinalg'
C_SRC_PATH = os.path.join(BASE_PATH, 'src')
LAPACK_LITE_PATH = os.path.join(C_SRC_PATH, 'lapack_lite')

# Use information about the LAPACK library used in NumPy.
# if not present, fallback to using the included lapack-lite

MODULE_SOURCES = [os.path.join(C_SRC_PATH, 'gulinalg.c.src'),
                  os.path.join(C_SRC_PATH, 'conditional_omp.h')]
MODULE_DEPENDENCIES = copy.copy(MODULE_SOURCES)

lapack_info = np_sys_info.get_info('lapack_opt', 0)
lapack_lite_files = [os.path.join(LAPACK_LITE_PATH, f)
                     for f in ['python_xerbla.c', 'zlapack_lite.c',
                               'dlapack_lite.c', 'blas_lite.c',
                               'dlamch.c', 'f2c_lite.c', 'f2c.h']]

if not lapack_info:
    # No LAPACK in NumPy
    print('### Warning: Using unoptimized blas/lapack @@@')
    MODULE_SOURCES.extend(lapack_lite_files[:-1]) # all but f2c.h
    MODULE_DEPENDENCIES.extend(lapack_lite_files)
else:
    if sys.platform == 'win32':
        print('### Warning: python.xerbla.c is disabled ###')
    else:
        MODULE_SOURCES.extend(lapack_lite_files[:1]) # python_xerbla.c
        MODULE_DEPENDENCIES.extend(lapack_lite_files[:1])

npymath_info = np_misc_util.get_info('npymath')
extra_opts = copy.deepcopy(lapack_info)

for key, val in npymath_info.items():
    if extra_opts.get(key):
        extra_opts[key].extend(val)
    else:
        extra_opts[key] = copy.deepcopy(val)

# make sure the compiler can find conditional_omp.h
extra_opts['include_dirs'] += [os.path.join(C_SRC_PATH)]

cmdclass = versioneer.get_cmdclass()

if "GULINALG_DISABLE_OPENMP" not in os.environ:
    # OpenMP will be disabled unless omp_test_c below compiles successfully
    omp_test_c = """#include <omp.h>
int main(int argc, char** argv) { return(0); }"""


    msc_flag_defines = [[["/openmp"], [], omp_test_c, "HAVE_VC_OPENMP"]]

    # flag_defines list order is:
    #     [compile_flags, link_flags, test_code, defines]
    openmp_libs = []
    if "msc" in platform.python_compiler().lower():
        # OpenMP flags for MSVC
        flag_defines = [[["/openmp"], [], omp_test_c, "HAVE_VC_OPENMP"]]
    elif 'GULINALG_USING_ICC' in os.environ:
        # OpenMP flags for Intel's icc
        flag_defines = [
            [["-qopenmp"], ["-qopenmp"], omp_test_c, "HAVE_OPENMP"]
        ]
    else:
        GULINALG_INTEL_OPENMP = 'GULINALG_INTEL_OPENMP' in os.environ
        if GULINALG_INTEL_OPENMP:
            # Link to Intel OpenMP library (instead of libgomp default for GCC)
            # Reference at Intel's site: https://tinyurl.com/yxmf7bwy
            openmp_link_args = []
            openmp_libs = ['iomp5', 'pthread']
        else:
            openmp_link_args = ['-fopenmp']

        # OpenMP flags for gcc
        flag_defines = [
            [["-fopenmp"], openmp_link_args, omp_test_c, "HAVE_OPENMP"]
        ]

    extbuilder = add_flag_checking(build_ext, flag_defines, "gulinalg",
                                   openmp_libs)
    cmdclass['build_ext'] = extbuilder
else:
    cmdclass['build_ext'] = build_ext

if 'extra_compile_args' not in extra_opts:
    extra_opts['extra_compile_args'] = []

if "msc" not in platform.python_compiler().lower():
    # Need C99 for _Pragma support
    # On windows, the MSVC-specific __pragma is used instead
    extra_opts['extra_compile_args'] += ['-std=c99']

print(f"extra_opts={extra_opts}")
gufunc_module = Extension('gulinalg._impl',
                          sources = MODULE_SOURCES,
                          depends = MODULE_DEPENDENCIES,
                          **extra_opts)

packages = [
    'gulinalg',
    'gulinalg.tests',
]

ext_modules = [
    gufunc_module,
]

setup(name='gulinalg',
      version=versioneer.get_version(),
      description='gufuncs for linear algebra',
      author='Continuum Analytics, Inc.; Quansight',
      ext_modules=ext_modules,
      packages=packages,
      license='BSD',
      long_description=open('README.md').read(),
      cmdclass=cmdclass,
)
