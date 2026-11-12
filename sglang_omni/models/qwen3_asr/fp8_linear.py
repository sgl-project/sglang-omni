"""FP8 e4m3 decoder linears for Qwen3-ASR on SM120: a weight-only decode kernel and the W8A8 prefill dispatch."""

from typing import NamedTuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
import torch.nn as nn
from cutlass.base_dsl.jit_executor import JitCompiledFunction
from cutlass.cute.runtime import from_dlpack, make_fake_stream

from sglang_omni.models.qwen3_asr import w8a8_linear
from sglang_omni.models.qwen3_asr.w8a8_linear import (
    copy_rows_tile,
    make_gmem_tiled_copy,
    make_smem_layout_ab,
    store_tile,
)

MMA_SHAPE = (16, 8, 16)
FP8_MAX = 448.0
# note (emre570): decode reads more weight bytes than it does math, so up to this many rows the weight-only kernel wins.
WEIGHT_ONLY_MAX_ROWS = 64
DECODER_LINEAR_NAMES = ("qkv_proj", "o_proj", "gate_up_proj", "down_proj")


@cute.kernel
def fp8_linear_kernel(
    mX: cute.Tensor,
    mW: cute.Tensor,
    mScale: cute.Tensor,
    mY: cute.Tensor,
    sX_layout: cute.Layout,
    sW_layout: cute.Layout,
    sX_swizzle: cute.Swizzle,
    sW_swizzle: cute.Swizzle,
    tiled_copy_X: cute.TiledCopy,
    tiled_copy_W: cute.TiledCopy,
    tiled_mma: cute.TiledMma,
    cta_tiler: cutlass.Constexpr,
    num_stages: cutlass.Constexpr,
):
    thread_index, _, _ = cute.arch.thread_idx()
    block_index_m, block_index_n, _ = cute.arch.block_idx()
    tile_m, tile_n, tile_k = cta_tiler
    num_rows = mX.shape[0]
    num_columns = mW.shape[0]
    num_threads = cute.size(tiled_mma)

    tile_coord = (block_index_m, block_index_n, None)
    gX = cute.local_tile(mX, cta_tiler, tile_coord, proj=(1, None, 1))
    gW = cute.local_tile(mW, cta_tiler, tile_coord, proj=(None, 1, 1))
    gY = cute.local_tile(mY, cta_tiler, tile_coord, proj=(1, 1, None))
    # note (emre570): mScale is an (M, N) view with stride (0, 1), so every token sees the same column scale.
    gScale = cute.local_tile(mScale, cta_tiler, tile_coord, proj=(1, 1, None))

    @cute.struct
    class SharedStorage:
        x: cute.struct.Align[
            cute.struct.MemRange[mX.element_type, cute.cosize(sX_layout)], 16
        ]
        w: cute.struct.Align[
            cute.struct.MemRange[mW.element_type, cute.cosize(sW_layout)], 16
        ]

    smem_allocator = cutlass.utils.SmemAllocator()
    storage = smem_allocator.allocate(SharedStorage.size_in_bytes(), byte_alignment=16)
    sX = SharedStorage(storage).x.get_tensor(sX_layout, swizzle=sX_swizzle)
    sW = SharedStorage(storage).w.get_tensor(sW_layout, swizzle=sW_swizzle)

    thr_copy_X = tiled_copy_X.get_slice(thread_index)
    thr_copy_W = tiled_copy_W.get_slice(thread_index)

    tXgX = thr_copy_X.partition_S(gX)
    tWgW = thr_copy_W.partition_S(gW)

    tXsX = thr_copy_X.partition_D(sX)
    tWsW = thr_copy_W.partition_D(sW)

    k_tile_count = cute.size(tXgX, mode=[3])

    thr_mma = tiled_mma.get_slice(thread_index)

    tCsX = thr_mma.partition_A(sX)
    tCsW = thr_mma.partition_B(sW)
    tCgY = thr_mma.partition_C(gY)
    tCgScale = thr_mma.partition_C(gScale)
    tCcY = thr_mma.partition_C(cute.make_identity_tensor((tile_m, tile_n)))

    tCrX = tiled_mma.make_fragment_A(tCsX[None, None, None, 0])
    tCrC = tiled_mma.make_fragment_C(tCgY)

    # note (emre570): W sits in registers twice, fp8 from SMEM and bf16 for the MMA, both in the B fragment layout.
    tCrW = cute.make_fragment_like(
        tiled_mma.make_fragment_B(tCsW[None, None, None, 0]),
        cutlass.BFloat16,
    )
    tCrW_fp8 = cute.make_fragment_like(tCrW, mW.element_type)

    tCrC.fill(0.0)

    # note (emre570): 8-bit W does not match the 16-bit ldmatrix order, so only X uses ldmatrix.
    atom_copy_s2r_X = cute.make_copy_atom(
        cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 4),
        mX.element_type,
    )
    tiled_copy_s2r_X = cute.make_tiled_copy_A(atom_copy_s2r_X, tiled_mma)
    thr_copy_ldmatrix_X = tiled_copy_s2r_X.get_slice(thread_index)

    tCsX_copy_view = thr_copy_ldmatrix_X.partition_S(sX)
    tCrX_copy_view = thr_copy_ldmatrix_X.retile(tCrX)

    k_block_count = cute.size(tCrX, mode=[2])

    # Row of X (and of W) that this thread copies in each copy pass
    x_rows_per_copy_pass = tile_m // cute.size(tXsX, mode=[1])
    x_thread_copy_row = thread_index // (num_threads // x_rows_per_copy_pass)
    w_rows_per_copy_pass = tile_n // cute.size(tWsW, mode=[1])
    w_thread_copy_row = thread_index // (num_threads // w_rows_per_copy_pass)

    block_first_row = block_index_m * tile_m
    block_first_column = block_index_n * tile_n
    is_full_block_m = block_first_row + tile_m <= num_rows
    # note (emre570): N is static, so shapes that fill every N tile compile without the N mask (a dead branch still costs about 1 us at decode).
    if cutlass.const_expr(num_columns % tile_n == 0):
        is_full_block_n = True
    else:
        is_full_block_n = block_first_column + tile_n <= num_columns

    for stage in cutlass.range_constexpr(num_stages):
        if stage < k_tile_count:
            copy_rows_tile(
                tiled_copy_X,
                tXgX[None, None, None, stage],
                tXsX[None, None, None, stage],
                is_full_block_m,
                block_first_row,
                x_thread_copy_row,
                x_rows_per_copy_pass,
                num_rows,
            )
            copy_rows_tile(
                tiled_copy_W,
                tWgW[None, None, None, stage],
                tWsW[None, None, None, stage],
                is_full_block_n,
                block_first_column,
                w_thread_copy_row,
                w_rows_per_copy_pass,
                num_columns,
            )
        else:
            pass
        cute.arch.cp_async_commit_group()

    cute.arch.cp_async_wait_group(num_stages - 1)
    cute.arch.sync_threads()

    for k_tile in range(k_tile_count):
        slot = k_tile % num_stages
        tCsX_p = tCsX_copy_view[None, None, None, slot]
        tCsW_p = tCsW[None, None, None, slot]

        cute.autovec_copy(tCsW_p, tCrW_fp8)
        tCrW.store(tCrW_fp8.load().to(cutlass.BFloat16))

        for k_block in cutlass.range(k_block_count, unroll_full=True):
            cute.copy(
                tiled_copy_s2r_X,
                tCsX_p[None, None, k_block],
                tCrX_copy_view[None, None, k_block],
            )
            cute.gemm(
                tiled_mma,
                tCrC,
                tCrX[None, None, k_block],
                tCrW[None, None, k_block],
                tCrC,
            )

        # note (emre570): every thread must finish reading this slot before cp.async writes the next tile into it.
        cute.arch.sync_threads()

        if k_tile + num_stages < k_tile_count:
            copy_rows_tile(
                tiled_copy_X,
                tXgX[None, None, None, k_tile + num_stages],
                tXsX[None, None, None, slot],
                is_full_block_m,
                block_first_row,
                x_thread_copy_row,
                x_rows_per_copy_pass,
                num_rows,
            )
            copy_rows_tile(
                tiled_copy_W,
                tWgW[None, None, None, k_tile + num_stages],
                tWsW[None, None, None, slot],
                is_full_block_n,
                block_first_column,
                w_thread_copy_row,
                w_rows_per_copy_pass,
                num_columns,
            )
        else:
            pass
        cute.arch.cp_async_commit_group()

        cute.arch.cp_async_wait_group(num_stages - 1)
        cute.arch.sync_threads()

    tCrScale = cute.make_fragment_like(tCrC)
    if is_full_block_n:
        cute.autovec_copy(tCgScale, tCrScale)
    else:
        for index in cutlass.range(cute.size(tCrScale), unroll_full=True):
            column = block_first_column + tCcY[index][1]
            if column < num_columns:
                tCrScale[index] = tCgScale[index]
            else:
                tCrScale[index] = cutlass.Float32(0.0)

    tCrY = cute.make_fragment_like(tCrC, mY.element_type)
    tCrY.store((tCrC.load() * tCrScale.load()).to(mY.element_type))

    store_tile(
        tCrY,
        tCgY,
        tCcY,
        is_full_block_m and is_full_block_n,
        block_first_row,
        block_first_column,
        num_rows,
        num_columns,
    )


@cute.jit
def fp8_linear_host(
    mX: cute.Tensor,
    mW: cute.Tensor,
    mScale: cute.Tensor,
    mY: cute.Tensor,
    stream: cuda.CUstream,
    cta_tiler: cutlass.Constexpr,
    num_stages: cutlass.Constexpr,
):
    tile_m, tile_n, tile_k = cta_tiler
    # note (emre570): M is one MMA atom row, and each warp owns 16 columns of N.
    num_warps = tile_n // 16
    atom_layout_mnk = (1, num_warps, 1)
    num_threads = num_warps * 32
    M = cute.size(mY, mode=[0])
    N = cute.size(mY, mode=[1])

    mScale = cute.make_tensor(mScale.iterator, cute.make_layout((M, N), stride=(0, 1)))

    mma_operation = cute.nvgpu.warp.MmaF16BF16Op(
        mX.element_type, cutlass.Float32, MMA_SHAPE
    )
    permutation_mnk = (
        atom_layout_mnk[0] * MMA_SHAPE[0],
        atom_layout_mnk[1] * MMA_SHAPE[1] * 2,
        atom_layout_mnk[2] * MMA_SHAPE[2],
    )
    tiled_mma = cute.make_tiled_mma(
        mma_operation,
        cute.make_layout(atom_layout_mnk),
        permutation_mnk=permutation_mnk,
    )

    copy_bits = 128
    x_copy_elements = copy_bits // mX.element_type.width
    assert (
        num_threads * x_copy_elements <= tile_m * tile_k
    ), "X copy pass larger than the X tile"
    w_copy_elements = copy_bits // mW.element_type.width
    assert (
        num_threads * w_copy_elements <= tile_n * tile_k
    ), "W copy pass larger than the W tile"

    sX_layout, sX_swizzle = make_smem_layout_ab(
        mX.element_type, copy_bits, (tile_m, tile_k, num_stages)
    )
    sW_layout, sW_swizzle = make_smem_layout_ab(
        mW.element_type, copy_bits, (tile_n, tile_k, num_stages)
    )

    atom_copy_g2s_X = cute.make_copy_atom(
        cute.nvgpu.cpasync.CopyG2SOp(cache_mode=cute.nvgpu.LoadCacheMode.GLOBAL),
        mX.element_type,
        num_bits_per_copy=copy_bits,
    )
    atom_copy_g2s_W = cute.make_copy_atom(
        cute.nvgpu.cpasync.CopyG2SOp(cache_mode=cute.nvgpu.LoadCacheMode.GLOBAL),
        mW.element_type,
        num_bits_per_copy=copy_bits,
    )
    tiled_copy_X = make_gmem_tiled_copy(
        atom_copy_g2s_X, mX.element_type, copy_bits, tile_k, num_threads
    )
    tiled_copy_W = make_gmem_tiled_copy(
        atom_copy_g2s_W, mW.element_type, copy_bits, tile_k, num_threads
    )

    fp8_linear_kernel(
        mX,
        mW,
        mScale,
        mY,
        sX_layout,
        sW_layout,
        sX_swizzle,
        sW_swizzle,
        tiled_copy_X,
        tiled_copy_W,
        tiled_mma,
        cta_tiler,
        num_stages,
    ).launch(
        grid=[cute.ceil_div(M, tile_m), cute.ceil_div(N, tile_n), 1],
        block=[num_threads, 1, 1],
        stream=stream,
    )


class WeightOnlyConfig(NamedTuple):
    cta_tiler: tuple[int, int, int]
    num_stages: int


# note (emre570): measured on the RTX 5080. Above 16 rows tile_m is 32, because 64 gives one CTA too much M work on small N.
CONFIG_BY_SHAPE: dict[tuple[int, int], tuple[WeightOnlyConfig, WeightOnlyConfig]] = {
    (4096, 2048): (
        WeightOnlyConfig((16, 32, 64), 3),
        WeightOnlyConfig((32, 32, 128), 2),
    ),  # qkv
    (2048, 2048): (
        WeightOnlyConfig((16, 32, 128), 3),
        WeightOnlyConfig((32, 32, 256), 2),
    ),  # o_proj
    (12288, 2048): (
        WeightOnlyConfig((16, 64, 64), 2),
        WeightOnlyConfig((32, 64, 64), 2),
    ),  # gate_up
    (2048, 6144): (
        WeightOnlyConfig((16, 64, 256), 4),
        WeightOnlyConfig((32, 32, 256), 2),
    ),  # down
    (151936, 2048): (
        WeightOnlyConfig((16, 32, 64), 2),
        WeightOnlyConfig((64, 64, 64), 2),
    ),  # lm_head
}
DEFAULT_CONFIGS = (
    WeightOnlyConfig((16, 64, 128), 2),
    WeightOnlyConfig((32, 64, 128), 2),
)
SMALL_BATCH_MAX_ROWS = 16
COMPILED_KERNELS: dict[tuple[int, int, WeightOnlyConfig], JitCompiledFunction] = {}


def pick_config(M: int, N: int, K: int) -> WeightOnlyConfig:
    small_batch_config, large_batch_config = CONFIG_BY_SHAPE.get(
        (N, K), DEFAULT_CONFIGS
    )
    if M <= SMALL_BATCH_MAX_ROWS:
        return small_batch_config
    else:
        return large_batch_config


def is_fp8_supported(weight: torch.Tensor) -> bool:
    """True when both kernels can serve this weight: bf16, and K fills every K tile."""
    N, K = weight.shape
    tile_k_values = [
        pick_config(1, N, K).cta_tiler[2],
        pick_config(WEIGHT_ONLY_MAX_ROWS, N, K).cta_tiler[2],
        w8a8_linear.pick_config(N, K).cta_tiler[2],
    ]
    if weight.dtype != torch.bfloat16:
        return False
    else:
        for tile_k in tile_k_values:
            if K % tile_k != 0:
                return False
            else:
                pass
        return True


def quantize_weight(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, float]:
    """bf16 [N, K] -> fp8 [N, K], fp32 scale vector [N] and the fp32 per-tensor scale it repeats."""
    amax = weight.float().abs().max().clamp(min=1e-12)
    scale = amax / FP8_MAX
    weight_fp8 = (weight.float() / scale).to(torch.float8_e4m3fn).contiguous()
    scale_vector = scale.expand(weight.shape[0]).contiguous()
    return weight_fp8, scale_vector, scale.item()


def wrap_rows(tensor: torch.Tensor) -> cute.Tensor:
    # note (emre570): only M is dynamic, so one compiled kernel serves every batch size.
    return from_dlpack(tensor, assumed_align=16).mark_compact_shape_dynamic(
        mode=0, stride_order=(0, 1), divisibility=1
    )


def wrap_weight(weight_fp8: torch.Tensor) -> cute.Tensor:
    # note (emre570): FP8 DLPack is not in every torch version, so pass a byte view and set the element type.
    mW = from_dlpack(weight_fp8.view(torch.uint8), assumed_align=16)
    mW.element_type = cutlass.Float8E4M3FN
    return mW


def get_compiled(M: int, N: int, K: int, device: torch.device) -> JitCompiledFunction:
    config = pick_config(M, N, K)
    cache_key = (N, K, config)
    if cache_key in COMPILED_KERNELS:
        return COMPILED_KERNELS[cache_key]
    else:
        pass
    x = torch.empty(M, K, device=device, dtype=torch.bfloat16)
    y = torch.empty(M, N, device=device, dtype=torch.bfloat16)
    weight_fp8 = torch.empty(N, K, device=device, dtype=torch.float8_e4m3fn)
    scale = torch.empty(N, device=device, dtype=torch.float32)
    COMPILED_KERNELS[cache_key] = cute.compile(
        fp8_linear_host,
        wrap_rows(x),
        wrap_weight(weight_fp8),
        from_dlpack(scale, assumed_align=16),
        wrap_rows(y),
        make_fake_stream(),
        config.cta_tiler,
        config.num_stages,
    )
    return COMPILED_KERNELS[cache_key]


def precompile(N: int, K: int, device: torch.device) -> None:
    # note (emre570): CUDA graph capture can not compile, so every kernel compiles at load.
    for M in [1, WEIGHT_ONLY_MAX_ROWS]:
        get_compiled(M, N, K, device)
    w8a8_linear.get_compiled(N, K, device)


def fp8_weight_only_linear(
    x: torch.Tensor,
    weight_fp8: torch.Tensor,
    scale_vector: torch.Tensor,
) -> torch.Tensor:
    """bf16 X [M, K] @ (fp8 W [N, K] * scale)^T -> bf16 Y [M, N]."""
    x = x.contiguous()
    M = x.shape[0]
    N, K = weight_fp8.shape
    y = torch.empty(M, N, device=x.device, dtype=torch.bfloat16)
    if M == 0:
        return y
    else:
        pass
    compiled = get_compiled(M, N, K, x.device)
    stream = cuda.CUstream(torch.cuda.current_stream(x.device).cuda_stream)
    compiled(
        wrap_rows(x),
        wrap_weight(weight_fp8),
        from_dlpack(scale_vector, assumed_align=16),
        wrap_rows(y),
        stream,
    )
    return y


class Fp8LinearMethod:
    """quant_method for a module that holds weight (fp8), weight_scale_vector and weight_scale."""

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        # note (emre570): the loader calls this on every quant_method, and set_fp8_weight already prepared the weights.
        pass

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        leading_shape = x.shape[:-1]
        x_2d = x.reshape(-1, x.shape[-1])
        if x_2d.shape[0] > WEIGHT_ONLY_MAX_ROWS:
            y = w8a8_linear.w8a8_linear(x_2d, layer.weight, layer.weight_scale)
        else:
            y = fp8_weight_only_linear(x_2d, layer.weight, layer.weight_scale_vector)
        if bias is not None:
            y = y + bias
        else:
            pass
        return y.reshape(*leading_shape, y.shape[-1])


def set_fp8_weight(module: nn.Module, weight: torch.Tensor) -> None:
    assert is_fp8_supported(
        weight
    ), f"no FP8 kernel for {weight.dtype} {tuple(weight.shape)}"
    weight_fp8, scale_vector, scale = quantize_weight(weight)
    module.register_buffer("weight", weight_fp8, persistent=False)
    module.register_buffer("weight_scale_vector", scale_vector, persistent=False)
    module.weight_scale = scale
    module.quant_method = Fp8LinearMethod()
    precompile(weight_fp8.shape[0], weight_fp8.shape[1], weight_fp8.device)


class Fp8LMHead(nn.Module):
    """FP8 copy of the tied lm_head. The embedding table itself stays bf16."""

    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        set_fp8_weight(self, weight.detach())


def install_fp8_linears(language_model: nn.Module) -> int:
    """Swap lm_head and the decoder linears to FP8 where is_fp8_supported allows. Returns the decoder count."""
    if is_fp8_supported(language_model.lm_head.weight):
        language_model.lm_head = Fp8LMHead(language_model.lm_head.weight)
    else:
        pass
    count = 0
    for name, module in language_model.model.named_modules():
        short_name = name.split(".")[-1]
        if short_name not in DECODER_LINEAR_NAMES:
            continue
        elif not is_fp8_supported(module.weight):
            continue
        else:
            pass
        weight = module.weight.detach()
        # note (emre570): drop the bf16 parameter, so its memory goes back to the KV cache.
        del module.weight
        set_fp8_weight(module, weight)
        count += 1
    for layer in language_model.model.layers:
        # note (emre570): SM100 DualGemm reads gate_up_proj.weight directly and skips quant_method.
        if isinstance(layer.mlp.gate_up_proj.quant_method, Fp8LinearMethod):
            layer.mlp.dual_gemm.mode = None
        else:
            pass
    repack_on_device(language_model)
    return count


def repack_on_device(module: nn.Module) -> None:
    """Round-trip every CUDA tensor of module through the host into fresh device memory.

    The FP8 weights land in the allocator segments of the freed bf16 weights, so empty_cache can not return those segments (about 1 GB on Qwen3-ASR-1.7B). After the round trip the KV cache gets that memory.
    """
    tensors = []
    seen_ids = set()
    candidates = list(module.parameters()) + list(module.buffers())
    for tensor in candidates:
        if not tensor.is_cuda or id(tensor) in seen_ids:
            continue
        else:
            seen_ids.add(id(tensor))
            tensors.append(tensor)

    host_copies = []
    for tensor in tensors:
        host_copies.append(tensor.data.to("cpu"))
        tensor.data = torch.empty(0, device=tensor.device, dtype=tensor.dtype)
    torch.cuda.empty_cache()

    for tensor, host_copy in zip(tensors, host_copies):
        tensor.data = host_copy.to(tensor.device)
