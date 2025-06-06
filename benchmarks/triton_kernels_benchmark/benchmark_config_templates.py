from triton_kernels_benchmark.benchmark_testing import BenchmarkCategory, BenchmarkConfig

from triton_kernels_benchmark import (
    fused_softmax,
    gemm_benchmark,
    flash_attention_benchmark,
    gemm_tensor_desc_benchmark,
    gemm_tensor_of_ptr_benchmark,
)

CONFIGS = [
    BenchmarkConfig(
        key="softmax",
        get_benchmark=fused_softmax.get_benchmark,
        run_opts={},
        categories={BenchmarkCategory.CORE, BenchmarkCategory.SOFTMAX},
        description="Triton Softmax kernel benchmark",
    ),
    BenchmarkConfig(
        key="gemm",
        get_benchmark=gemm_benchmark.get_benchmark,
        run_opts={},
        categories={BenchmarkCategory.CORE, BenchmarkCategory.GEMM},
        description="Triton GEMM kernel benchmark",
    ),
    BenchmarkConfig(
        key="gemm-tensor-of-ptr",
        get_benchmark=gemm_tensor_of_ptr_benchmark.get_benchmark,
        run_opts={},
        categories={BenchmarkCategory.EXPERIMENTAL, BenchmarkCategory.GEMM},
        description="Triton GEMM kernel benchmark - with tensor of pointer",
    ),
    BenchmarkConfig(
        key="gemm-tensor-desc",
        get_benchmark=gemm_tensor_desc_benchmark.get_benchmark,
        run_opts={},
        categories={BenchmarkCategory.EXPERIMENTAL, BenchmarkCategory.GEMM},
        description="Triton GEMM kernel benchmark - with tensor descriptor",
    ),
    BenchmarkConfig(
        key="gemm_bt",
        get_benchmark=gemm_benchmark.get_benchmark,
        run_opts={"transpose_b": True},
        categories={BenchmarkCategory.CORE, BenchmarkCategory.GEMM},
        description="Triton GEMM (A@B^t) kernel benchmark",
    ),
    BenchmarkConfig(
        key="gemm_at",
        get_benchmark=gemm_benchmark.get_benchmark,
        run_opts={"transpose_a": True},
        categories={BenchmarkCategory.EXPERIMENTAL, BenchmarkCategory.GEMM},
        description="Triton GEMM (A^t@B) kernel benchmark",
    ),
    BenchmarkConfig(
        key="flash_attention",
        get_benchmark=flash_attention_benchmark.get_benchmark,
        run_opts={"fa_kernel_mode": "fwd"},
        categories={BenchmarkCategory.CORE, BenchmarkCategory.FLASH_ATTENTION},
    ),
]
