"""W8A8 FP8 linear for Qwen3-ASR prefill on SM120, and the tile helpers both FP8 kernels share."""

import math
from typing import NamedTuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils.blackwell_helpers as sm120_utils
import torch
from cutlass.base_dsl.jit_executor import JitCompiledFunction
from cutlass.cute.runtime import from_dlpack, make_fake_stream
from sgl_kernel import sgl_per_token_quant_fp8

MMA_SHAPE = (16, 8, 32)


@cute.jit
def copy_rows_tile(
    tiled_copy: cute.TiledCopy,
    source_tile: cute.Tensor,
    destination_stage: cute.Tensor,
    is_full_block: cutlass.Boolean,
    block_first_row: cutlass.Int32,
    thread_copy_row: cutlass.Int32,
    rows_per_copy_pass: cutlass.Constexpr,
    num_rows: cutlass.Int32,
):
    if is_full_block:
        cute.copy(tiled_copy, source_tile, destination_stage)
    else:
        # note (emre570): rows past num_rows stay stale in SMEM. They only feed outputs that are never stored.
        for copy_pass in cutlass.range_constexpr(
            cute.size(destination_stage, mode=[1])
        ):
            row = block_first_row + thread_copy_row + copy_pass * rows_per_copy_pass
            if row < num_rows:
                cute.copy(
                    tiled_copy,
                    source_tile[None, copy_pass, None],
                    destination_stage[None, copy_pass, None],
                )
            else:
                pass


@cute.jit
def store_tile(
    tCrY: cute.Tensor,
    tCgY: cute.Tensor,
    tCcY: cute.Tensor,
    is_full_tile: cutlass.Boolean,
    block_first_row: cutlass.Int32,
    block_first_column: cutlass.Int32,
    num_rows: cutlass.Int32,
    num_columns: cutlass.Int32,
):
    if is_full_tile:
        cute.autovec_copy(tCrY, tCgY)
    else:
        for index in cutlass.range(cute.size(tCrY), unroll_full=True):
            row = block_first_row + tCcY[index][0]
            column = block_first_column + tCcY[index][1]
            if row < num_rows:
                if column < num_columns:
                    tCgY[index] = tCrY[index]
                else:
                    pass
            else:
                pass


# note (emre570): the swizzle works on bytes, so shift = log2(row bytes) - base. This gives 3 for a 128-byte row.
def make_smem_layout_ab(
    dtype: type[cutlass.Numeric],
    copy_bits: int,
    smem_tiler: tuple[int, int, int],
) -> tuple[cute.Layout, cute.Swizzle]:
    major_size = min(smem_tiler[1], 128 * 8 // dtype.width)
    swizzle_bits = min(int(math.log2(major_size * dtype.width // copy_bits)), 3)
    base_bits = int(math.log2(copy_bits // 8))
    shift_bits = int(math.log2(major_size * dtype.width // 8)) - base_bits
    swizzle = cute.make_swizzle(swizzle_bits, base_bits, shift_bits)
    atom = cute.make_layout((8, major_size), stride=(major_size, 1))
    layout = cute.tile_to_shape(atom, smem_tiler, (0, 1, 2))
    return layout, swizzle


def make_gmem_tiled_copy(
    atom_copy: cute.CopyAtom,
    dtype: type[cutlass.Numeric],
    copy_bits: int,
    minor_size: int,
    num_threads: int,
) -> cute.TiledCopy:
    copy_elements = copy_bits // dtype.width
    threads_per_row = minor_size // copy_elements
    thread_layout = cute.make_layout(
        (num_threads // threads_per_row, threads_per_row),
        stride=(threads_per_row, 1),
    )
    value_layout = cute.make_layout((1, copy_elements))
    return cute.make_tiled_copy_tv(atom_copy, thread_layout, value_layout)


@cute.kernel
def w8a8_gemm_kernel(
    mX: cute.Tensor,
    mW: cute.Tensor,
    mXScale: cute.Tensor,
    mY: cute.Tensor,
    w_scale: cutlass.Float32,
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
    # note (emre570): mXScale is an (M, N) view with stride (1, 0), so every column sees the token scale of its row.
    gXScale = cute.local_tile(mXScale, cta_tiler, tile_coord, proj=(1, 1, None))

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
    tCgXScale = thr_mma.partition_C(gXScale)
    tCcY = thr_mma.partition_C(cute.make_identity_tensor((tile_m, tile_n)))

    tCrX = tiled_mma.make_fragment_A(tCsX[None, None, None, 0])
    tCrW = tiled_mma.make_fragment_B(tCsW[None, None, None, 0])
    tCrC = tiled_mma.make_fragment_C(tCgY)

    tCrC.fill(0.0)

    # note (emre570): block scale 0x7F is 2^0, so the block-scaled MMA gives the plain FP8 MMA bits at 2x the rate on SM120.
    scale_factor_dtype = cutlass.Float8E8M0FNU
    scale_factor_ref_X = cute.make_rmem_tensor(
        cute.make_layout((tile_m, tile_k), stride=(0, 0)),
        scale_factor_dtype,
    )
    scale_factor_ref_W = cute.make_rmem_tensor(
        cute.make_layout((tile_n, tile_k), stride=(0, 0)),
        scale_factor_dtype,
    )
    tCrSFX = sm120_utils.partition_fragment_SFA(
        scale_factor_ref_X, thr_mma, thread_index
    )
    tCrSFW = sm120_utils.partition_fragment_SFB(
        scale_factor_ref_W, thr_mma, thread_index
    )
    cute.recast_tensor(tCrSFX, cutlass.Uint8).fill(0x7F)
    cute.recast_tensor(tCrSFW, cutlass.Uint8).fill(0x7F)

    atom_copy_s2r_X = cute.make_copy_atom(
        cute.nvgpu.warp.LdMatrix8x16x8bOp(transpose=False, num_matrices=4),
        mX.element_type,
    )
    atom_copy_s2r_W = cute.make_copy_atom(
        cute.nvgpu.warp.LdMatrix8x16x8bOp(transpose=False, num_matrices=4),
        mW.element_type,
    )
    tiled_copy_s2r_X = cute.make_tiled_copy_A(atom_copy_s2r_X, tiled_mma)
    tiled_copy_s2r_W = cute.make_tiled_copy_B(atom_copy_s2r_W, tiled_mma)

    thr_copy_ldmatrix_X = tiled_copy_s2r_X.get_slice(thread_index)
    thr_copy_ldmatrix_W = tiled_copy_s2r_W.get_slice(thread_index)

    tCsX_copy_view = thr_copy_ldmatrix_X.partition_S(sX)
    tCrX_copy_view = thr_copy_ldmatrix_X.retile(tCrX)
    tCsW_copy_view = thr_copy_ldmatrix_W.partition_S(sW)
    tCrW_copy_view = thr_copy_ldmatrix_W.retile(tCrW)

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
        tCsW_p = tCsW_copy_view[None, None, None, slot]

        for k_block in cutlass.range(k_block_count, unroll_full=True):
            cute.copy(
                tiled_copy_s2r_X,
                tCsX_p[None, None, k_block],
                tCrX_copy_view[None, None, k_block],
            )
            cute.copy(
                tiled_copy_s2r_W,
                tCsW_p[None, None, k_block],
                tCrW_copy_view[None, None, k_block],
            )
            cute.gemm(
                tiled_mma,
                tCrC,
                [
                    tCrX[None, None, k_block],
                    tCrSFX[None, None, k_block],
                ],
                [
                    tCrW[None, None, k_block],
                    tCrSFW[None, (None, k_block)],
                ],
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

    # note (emre570): the host pads x_scale to whole M tiles, so this read stays in bounds without a row mask.
    tCrXScale = cute.make_fragment_like(tCrC)
    cute.autovec_copy(tCgXScale, tCrXScale)

    tCrY = cute.make_fragment_like(tCrC, mY.element_type)
    tCrY.store((tCrC.load() * tCrXScale.load() * w_scale).to(mY.element_type))

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
def w8a8_gemm_host(
    mX: cute.Tensor,
    mW: cute.Tensor,
    mXScale: cute.Tensor,
    mY: cute.Tensor,
    w_scale: cutlass.Float32,
    stream: cuda.CUstream,
    cta_tiler: cutlass.Constexpr,
    num_stages: cutlass.Constexpr,
    warps_m: cutlass.Constexpr,
    warps_n: cutlass.Constexpr,
):
    tile_m, tile_n, tile_k = cta_tiler
    M = cute.size(mY, mode=[0])
    N = cute.size(mY, mode=[1])
    atom_layout_mnk = (warps_m, warps_n, 1)
    num_threads = warps_m * warps_n * 32

    mXScale = cute.make_tensor(
        mXScale.iterator, cute.make_layout((M, N), stride=(1, 0))
    )

    mma_operation = cute.nvgpu.warp.MmaMXF8Op(
        mX.element_type, cutlass.Float32, cutlass.Float8E8M0FNU
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
    copy_elements = copy_bits // mX.element_type.width
    assert (
        num_threads * copy_elements <= tile_m * tile_k
    ), "X copy pass larger than the X tile"
    assert (
        num_threads * copy_elements <= tile_n * tile_k
    ), "W copy pass larger than the W tile"
    assert tile_k % MMA_SHAPE[2] == 0, "tile_k must be a multiple of 32"
    sX_layout, sX_swizzle = make_smem_layout_ab(
        mX.element_type, copy_bits, (tile_m, tile_k, num_stages)
    )
    sW_layout, sW_swizzle = make_smem_layout_ab(
        mW.element_type, copy_bits, (tile_n, tile_k, num_stages)
    )

    atom_copy_g2s = cute.make_copy_atom(
        cute.nvgpu.cpasync.CopyG2SOp(cache_mode=cute.nvgpu.LoadCacheMode.GLOBAL),
        mX.element_type,
        num_bits_per_copy=copy_bits,
    )
    tiled_copy_X = make_gmem_tiled_copy(
        atom_copy_g2s, mX.element_type, copy_bits, tile_k, num_threads
    )
    tiled_copy_W = make_gmem_tiled_copy(
        atom_copy_g2s, mW.element_type, copy_bits, tile_k, num_threads
    )

    w8a8_gemm_kernel(
        mX,
        mW,
        mXScale,
        mY,
        w_scale,
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


class W8A8Config(NamedTuple):
    cta_tiler: tuple[int, int, int]
    num_stages: int
    warps_m: int
    warps_n: int


# note (emre570): best of a 7 tile x 2 stage x 3 warp grid sweep on the RTX 5080, 2.7-3.1x over bf16 cuBLAS for M >= 512.
CONFIG_BY_SHAPE: dict[tuple[int, int], W8A8Config] = {
    (4096, 2048): W8A8Config((64, 128, 64), 2, 2, 2),  # qkv
    (2048, 2048): W8A8Config((64, 128, 64), 2, 2, 2),  # o_proj
    (12288, 2048): W8A8Config((128, 128, 64), 2, 2, 2),  # gate_up
    (2048, 6144): W8A8Config((64, 128, 64), 2, 2, 2),  # down
}
DEFAULT_CONFIG = W8A8Config((64, 128, 64), 2, 2, 2)
COMPILED_KERNELS: dict[tuple[int, int, W8A8Config], JitCompiledFunction] = {}


def pick_config(N: int, K: int) -> W8A8Config:
    return CONFIG_BY_SHAPE.get((N, K), DEFAULT_CONFIG)


def wrap_fp8_rows(tensor: torch.Tensor) -> cute.Tensor:
    view = from_dlpack(
        tensor.view(torch.uint8), assumed_align=16
    ).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1), divisibility=1)
    view.element_type = cutlass.Float8E4M3FN
    return view


def wrap_fp8(tensor: torch.Tensor) -> cute.Tensor:
    view = from_dlpack(tensor.view(torch.uint8), assumed_align=16)
    view.element_type = cutlass.Float8E4M3FN
    return view


def wrap_rows(tensor: torch.Tensor) -> cute.Tensor:
    if tensor.dim() == 1:
        return from_dlpack(tensor, assumed_align=16).mark_compact_shape_dynamic(
            mode=0, divisibility=1
        )
    else:
        return from_dlpack(tensor, assumed_align=16).mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1), divisibility=1
        )


def get_compiled(N: int, K: int, device: torch.device) -> JitCompiledFunction:
    config = pick_config(N, K)
    cache_key = (N, K, config)
    if cache_key in COMPILED_KERNELS:
        return COMPILED_KERNELS[cache_key]
    else:
        pass
    M = 128
    x_fp8 = torch.empty(M, K, device=device, dtype=torch.float8_e4m3fn)
    w_fp8 = torch.empty(N, K, device=device, dtype=torch.float8_e4m3fn)
    x_scale = torch.empty(M, device=device, dtype=torch.float32)
    y = torch.empty(M, N, device=device, dtype=torch.bfloat16)
    COMPILED_KERNELS[cache_key] = cute.compile(
        w8a8_gemm_host,
        wrap_fp8_rows(x_fp8),
        wrap_fp8(w_fp8),
        wrap_rows(x_scale),
        wrap_rows(y),
        1.0,
        make_fake_stream(),
        *config,
    )
    return COMPILED_KERNELS[cache_key]


def w8a8_linear(
    x: torch.Tensor,
    weight_fp8: torch.Tensor,
    weight_scale: float,
) -> torch.Tensor:
    """bf16 X [M, K] -> per-token FP8 -> W8A8 GEMM -> bf16 Y [M, N]."""
    x = x.contiguous()
    M, K = x.shape
    N = weight_fp8.shape[0]
    tile_m = pick_config(N, K).cta_tiler[0]
    padded_rows = (M + tile_m - 1) // tile_m * tile_m
    x_fp8 = torch.empty(M, K, device=x.device, dtype=torch.float8_e4m3fn)
    x_scale = torch.empty(padded_rows, 1, device=x.device, dtype=torch.float32)
    sgl_per_token_quant_fp8(x, x_fp8, x_scale[:M])
    y = torch.empty(M, N, device=x.device, dtype=torch.bfloat16)
    compiled = get_compiled(N, K, x.device)
    stream = cuda.CUstream(torch.cuda.current_stream(x.device).cuda_stream)
    compiled(
        wrap_fp8_rows(x_fp8),
        wrap_fp8(weight_fp8),
        wrap_rows(x_scale.view(-1)),
        wrap_rows(y),
        weight_scale,
        stream,
    )
    return y
