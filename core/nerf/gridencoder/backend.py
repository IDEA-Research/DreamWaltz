import os
from torch.utils.cpp_extension import load

_src_path = os.path.dirname(os.path.abspath(__file__))

nvcc_flags = [
    '-O3', '-std=c++14',
    '-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__', '-U__CUDA_NO_HALF2_OPERATORS__',
]

if os.name == "posix":
    c_flags = ['-O3', '-std=c++14']

build_directory = os.path.join(_src_path, 'build')
os.makedirs(build_directory, exist_ok=True)

_backend = load(name='_grid_encoder',
                extra_cflags=c_flags,
                extra_cuda_cflags=nvcc_flags,
                build_directory=build_directory,
                sources=[os.path.join(_src_path, 'src', f) for f in [
                    'gridencoder.cu',
                    'bindings.cpp',
                ]],
                )

__all__ = ['_backend']