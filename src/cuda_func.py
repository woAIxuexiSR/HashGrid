from torch.utils.cpp_extension import load as load_cuda
import os
from glob import glob
import torch
import numpy as np


def load_cuda_imp(Debug=False, Verbose=False):
    src_dir = os.path.dirname(os.path.abspath(__file__)) + "/cuda"
    os.environ["TORCH_EXTENSIONS_DIR"] = os.path.join(src_dir, "build")
    cflags = ["--extended-lambda", "--expt-relaxed-constexpr"]
    if Debug:
        cflags += ["-G", "-g", "-O0"]
        cflags += ["-DDEBUG"]
    else:
        cflags += ["-O3"]
        cflags += ["-DNDEBUG"]

    cuda_files = glob(src_dir + "/*.cu")
    include_paths = [src_dir + "/include"]
    return load_cuda(
        name="CUDA_MODULE",
        sources=cuda_files,
        extra_include_paths=include_paths,
        extra_cuda_cflags=cflags,
        verbose=Verbose,
    )


cuda_imp = load_cuda_imp(Debug=False, Verbose=False)


def check_tensor(tensor, dtype):
    assert tensor.dtype == dtype
    assert tensor.is_cuda
    assert tensor.is_contiguous()


def knn(xi, xf, k):
    """
    xi: torch.Tensor, shape [batch_size, input_dim], long
    xf: torch.Tensor, shape [batch_size, input_dim], float
    k: int

    return:
        neigs: torch.Tensor, shape [batch_size, k, input_dim], long
        ws: torch.Tensor, shape [batch_size, k], float
    """

    check_tensor(xi, torch.int64)
    check_tensor(xf, torch.float32)

    return cuda_imp.knn(xi, xf, k)
