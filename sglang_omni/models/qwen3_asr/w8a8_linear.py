"""W8A8 FP8 linear for Qwen3-ASR prefill on SM120.

X is quantized per token with sgl_kernel, W is FP8 with one per-tensor scale, and
the GEMM runs on the block-scaled FP8 MMA. See fp8_linear.py for the dispatch.
"""

import math

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils.blackwell_helpers as sm120_utils
import torch
from cutlass.cute.runtime import from_dlpack, make_fake_stream

# W8A8: Y = (X_fp8 * x_scale[m]) @ (W_fp8 * w_scale)^T
# X [M, K] fp8 e4m3 + x_scale [M] fp32 (one per token)
# W [N, K] fp8 e4m3 + w_scale fp32 (one per tensor)
# Y [M, N] bf16
# The MMA is the block-scaled FP8 MMA (m16n8k32) with every block scale = 1.0.
# On SM120 it runs at 2x the plain FP8 MMA rate and gives the same bits.
_CTA_TILER = (64, 64, 64)
_NUM_STAGES = 2
_MMA_SHAPE = (16, 8, 32)


@cute.jit
def copy_X_tile(
    tiled_copy_X: cute.TiledCopy,
    tXgX_tile: cute.Tensor,
    tXsX_stage: cute.Tensor,
    is_full_block: cutlass.Boolean,
    block_first_row: cutlass.Int32,
    thread_row: cutlass.Int32,
    thread_rows: cutlass.Constexpr,
    num_rows: cutlass.Int32,
):
    if is_full_block:
        cute.copy(tiled_copy_X, tXgX_tile, tXsX_stage)
    else:
        # Rows past M stay stale in SMEM, and their output is never stored
        for copy_row in cutlass.range_constexpr(cute.size(tXsX_stage, mode=[1])):
            row = block_first_row + thread_row + copy_row * thread_rows
            if row < num_rows:
                cute.copy(
                    tiled_copy_X,
                    tXgX_tile[None, copy_row, None],
                    tXsX_stage[None, copy_row, None],
                )
            else:
                pass


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
    cta_tiler: cutlass.Constexpr = _CTA_TILER,
    num_stages: cutlass.Constexpr = _NUM_STAGES,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()

    gX = cute.local_tile(mX, cta_tiler, (bidx, bidy, None), proj=(1, None, 1))
    gW = cute.local_tile(mW, cta_tiler, (bidx, bidy, None), proj=(None, 1, 1))
    gY = cute.local_tile(mY, cta_tiler, (bidx, bidy, None), proj=(1, 1, None))
    # mXScale is an (M, N) view with stride (1, 0): every column sees the same token scale
    gXScale = cute.local_tile(mXScale, cta_tiler, (bidx, bidy, None), proj=(1, 1, None))

    @cute.struct
    class SharedStorage:
        x: cute.struct.Align[
            cute.struct.MemRange[mX.element_type, cute.cosize(sX_layout)], 16
        ]
        w: cute.struct.Align[
            cute.struct.MemRange[mW.element_type, cute.cosize(sW_layout)], 16
        ]

    smem = cutlass.utils.SmemAllocator()
    storage = smem.allocate(SharedStorage.size_in_bytes(), byte_alignment=16)
    sX = SharedStorage(storage).x.get_tensor(sX_layout, swizzle=sX_swizzle)
    sW = SharedStorage(storage).w.get_tensor(sW_layout, swizzle=sW_swizzle)

    thr_copy_X = tiled_copy_X.get_slice(tidx)
    thr_copy_W = tiled_copy_W.get_slice(tidx)

    tXgX = thr_copy_X.partition_S(gX)
    tWgW = thr_copy_W.partition_S(gW)

    tXsX = thr_copy_X.partition_D(sX)
    tWsW = thr_copy_W.partition_D(sW)

    k_tile_count = cute.size(tXgX, mode=[3])

    thr_mma = tiled_mma.get_slice(tidx)

    tCsX = thr_mma.partition_A(sX)
    tCsW = thr_mma.partition_B(sW)
    tCgY = thr_mma.partition_C(gY)
    tCgXScale = thr_mma.partition_C(gXScale)

    tCrX = tiled_mma.make_fragment_A(tCsX[None, None, None, 0])
    tCrW = tiled_mma.make_fragment_B(tCsW[None, None, None, 0])
    tCrC = tiled_mma.make_fragment_C(tCgY)

    tCrC.fill(0.0)

    # The block-scaled MMA wants one UE8M0 scale per 32 k elements.
    # 0x7F = 2^0 = 1.0, so the math is the plain FP8 MMA (same bits, 2x the rate on SM120).
    # The real scales (x_scale, w_scale) do not depend on k and go to the epilogue.
    sf_dtype = cutlass.Float8E8M0FNU
    sf_ref_X = cute.make_rmem_tensor(
        cute.make_layout((cta_tiler[0], cta_tiler[2]), stride=(0, 0)),
        sf_dtype,
    )
    sf_ref_W = cute.make_rmem_tensor(
        cute.make_layout((cta_tiler[1], cta_tiler[2]), stride=(0, 0)),
        sf_dtype,
    )
    tCrSFX = sm120_utils.partition_fragment_SFA(sf_ref_X, thr_mma, tidx)
    tCrSFW = sm120_utils.partition_fragment_SFB(sf_ref_W, thr_mma, tidx)
    cute.recast_tensor(tCrSFX, cutlass.Uint8).fill(0x7F)
    cute.recast_tensor(tCrSFW, cutlass.Uint8).fill(0x7F)

    # X and W are both 8-bit and k-major: ldmatrix 8x16x8b
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

    thr_copy_ldmatrix_X = tiled_copy_s2r_X.get_slice(tidx)
    thr_copy_ldmatrix_W = tiled_copy_s2r_W.get_slice(tidx)

    tCsX_copy_view = thr_copy_ldmatrix_X.partition_S(sX)
    tCrX_copy_view = thr_copy_ldmatrix_X.retile(tCrX)
    tCsW_copy_view = thr_copy_ldmatrix_W.partition_S(sW)
    tCrW_copy_view = thr_copy_ldmatrix_W.retile(tCrW)

    num_k_block = cute.size(tCrX, mode=[2])

    # X rows owned by this thread are thread_row + copy_row * thread_rows
    thread_rows = cta_tiler[0] // cute.size(tXsX, mode=[1])
    thread_row = tidx // (cute.size(tiled_mma) // thread_rows)
    block_first_row = bidx * cta_tiler[0]
    is_full_block = block_first_row + cta_tiler[0] <= mX.shape[0]

    # prologue: tiles t0..t(S-1) -> slots 0..S-1, one commit group per tile
    for stage in cutlass.range_constexpr(num_stages):
        if stage < k_tile_count:
            copy_X_tile(
                tiled_copy_X,
                tXgX[None, None, None, stage],
                tXsX[None, None, None, stage],
                is_full_block,
                block_first_row,
                thread_row,
                thread_rows,
                mX.shape[0],
            )
            cute.copy(
                tiled_copy_W,
                tWgW[None, None, None, stage],
                tWsW[None, None, None, stage],
            )
        else:
            pass
        cute.arch.cp_async_commit_group()

    # SYNC-A: tile t0 is in SMEM
    cute.arch.cp_async_wait_group(num_stages - 1)
    cute.arch.sync_threads()

    for k_tile in range(k_tile_count):
        slot = k_tile % num_stages
        tCsX_p = tCsX_copy_view[None, None, None, slot]
        tCsW_p = tCsW_copy_view[None, None, None, slot]

        for k_block in cutlass.range(num_k_block, unroll_full=True):
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

        # SYNC-B: every thread is done with this slot
        cute.arch.sync_threads()

        if k_tile + num_stages < k_tile_count:
            copy_X_tile(
                tiled_copy_X,
                tXgX[None, None, None, k_tile + num_stages],
                tXsX[None, None, None, slot],
                is_full_block,
                block_first_row,
                thread_row,
                thread_rows,
                mX.shape[0],
            )
            cute.copy(
                tiled_copy_W,
                tWgW[None, None, None, k_tile + num_stages],
                tWsW[None, None, None, slot],
            )
        else:
            pass
        cute.arch.cp_async_commit_group()

        # SYNC-C: tile t(k+1) is in SMEM
        cute.arch.cp_async_wait_group(num_stages - 1)
        cute.arch.sync_threads()

    # epilogue: x_scale per row, w_scale one number. Neither depends on k.
    tCrXScale = cute.make_fragment_like(tCrC)
    cute.autovec_copy(tCgXScale, tCrXScale)

    tCrY = cute.make_fragment_like(tCrC, mY.element_type)
    tCrY.store((tCrC.load() * tCrXScale.load() * w_scale).to(mY.element_type))

    tCgCoord = thr_mma.partition_C(
        cute.make_identity_tensor((cta_tiler[0], cta_tiler[1]))
    )
    if is_full_block:
        cute.autovec_copy(tCrY, tCgY)
    else:
        for index in cutlass.range(cute.size(tCrY), unroll_full=True):
            row = block_first_row + tCgCoord[index][0]
            if row < mY.shape[0]:
                tCgY[index] = tCrY[index]
            else:
                pass


# Same helper as fp8_linear.py (byte swizzle, shift = log2(row bytes) - base)
def make_smem_layout_ab(dtype, copy_bits, smem_tiler):
    major_size = min(smem_tiler[1], 128 * 8 // dtype.width)
    swizzle_bits = min(int(math.log2(major_size * dtype.width // copy_bits)), 3)
    base_bits = int(math.log2(copy_bits // 8))
    shift_bits = int(math.log2(major_size * dtype.width // 8)) - base_bits
    swizzle = cute.make_swizzle(swizzle_bits, base_bits, shift_bits)
    atom = cute.make_layout((8, major_size), stride=(major_size, 1))
    layout = cute.tile_to_shape(atom, smem_tiler, (0, 1, 2))
    return layout, swizzle


def make_gmem_tiled_copy(atom_copy, dtype, copy_bits, minor_size, num_threads):
    copy_elems = copy_bits // dtype.width
    shape_dim_1 = minor_size // copy_elems
    thread_layout = cute.make_layout(
        (num_threads // shape_dim_1, shape_dim_1), stride=(shape_dim_1, 1)
    )
    value_layout = cute.make_layout((1, copy_elems))
    return cute.make_tiled_copy_tv(atom_copy, thread_layout, value_layout)


@cute.jit
def w8a8_gemm_host(
    mX: cute.Tensor,
    mW: cute.Tensor,
    mXScale: cute.Tensor,
    mY: cute.Tensor,
    w_scale: cutlass.Float32,
    stream: cuda.CUstream,
    cta_tiler: cutlass.Constexpr = _CTA_TILER,
    num_stages: cutlass.Constexpr = _NUM_STAGES,
    warps_m: cutlass.Constexpr = 1,
    warps_n: cutlass.Constexpr = 4,
):
    bM, bN, bK = cta_tiler
    M = cute.size(mY, mode=[0])
    N = cute.size(mY, mode=[1])
    atom_layout_mnk = (warps_m, warps_n, 1)
    num_threads = warps_m * warps_n * 32

    # 1. x_scale [M] -> [M, N] view, stride 0 along N
    mXScale = cute.make_tensor(
        mXScale.iterator, cute.make_layout((M, N), stride=(1, 0))
    )

    # 2. Tiled MMA: block-scaled FP8 atom, warp grid (warps_m, warps_n)
    op = cute.nvgpu.warp.MmaMXF8Op(
        mX.element_type, cutlass.Float32, cutlass.Float8E8M0FNU
    )
    permutation_mnk = (
        atom_layout_mnk[0] * _MMA_SHAPE[0],
        atom_layout_mnk[1] * _MMA_SHAPE[1] * 2,
        atom_layout_mnk[2] * _MMA_SHAPE[2],
    )
    tiled_mma = cute.make_tiled_mma(
        op, cute.make_layout(atom_layout_mnk), permutation_mnk=permutation_mnk
    )

    # 3. SMEM per stage: X bM x bK fp8 + W bN x bK fp8
    copy_bits = 128
    copy_elems = copy_bits // mX.element_type.width
    assert num_threads * copy_elems <= bM * bK, "X copy pass larger than the X tile"
    assert num_threads * copy_elems <= bN * bK, "W copy pass larger than the W tile"
    assert bK % _MMA_SHAPE[2] == 0, "bK must be a multiple of 32"
    sX_layout, sX_swizzle = make_smem_layout_ab(
        mX.element_type, copy_bits, (bM, bK, num_stages)
    )
    sW_layout, sW_swizzle = make_smem_layout_ab(
        mW.element_type, copy_bits, (bN, bK, num_stages)
    )

    # 4. g2s async copy: 16 bytes = 16 fp8 elements
    atom_g2s = cute.make_copy_atom(
        cute.nvgpu.cpasync.CopyG2SOp(cache_mode=cute.nvgpu.LoadCacheMode.GLOBAL),
        mX.element_type,
        num_bits_per_copy=copy_bits,
    )
    tiled_copy_X = make_gmem_tiled_copy(
        atom_g2s, mX.element_type, copy_bits, bK, num_threads
    )
    tiled_copy_W = make_gmem_tiled_copy(
        atom_g2s, mW.element_type, copy_bits, bK, num_threads
    )

    # 5. Launch: grid = output tiles. M is dynamic, the last tile can be partial.
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
        grid=[cute.ceil_div(M, bM), N // bN, 1],
        block=[num_threads, 1, 1],
        stream=stream,
    )


# Prefill configs, measured on the 5080 (best of a 7 tile x 2 stage x 3 warp grid sweep).
# The W8A8 GEMM is 2.7-3.1x faster than bf16 cuBLAS for M >= 512 on all four shapes.
_CONFIG_BY_SHAPE = {
    (4096, 2048): ((64, 128, 64), 2, 2, 2),  # qkv
    (2048, 2048): ((64, 128, 64), 2, 2, 2),  # o_proj
    (12288, 2048): ((128, 128, 64), 2, 2, 2),  # gate_up
    (2048, 6144): ((64, 128, 64), 2, 2, 2),  # down
}
_DEFAULT_CONFIG = ((64, 128, 64), 2, 2, 2)
_COMPILED_KERNELS: dict[tuple, object] = {}


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


def get_compiled(N: int, K: int, device: torch.device):
    config = _CONFIG_BY_SHAPE.get((N, K), _DEFAULT_CONFIG)
    key = (N, K, config)
    if key in _COMPILED_KERNELS:
        return _COMPILED_KERNELS[key]
    else:
        pass
    M = 128
    x_fp8 = torch.empty(M, K, device=device, dtype=torch.float8_e4m3fn)
    w_fp8 = torch.empty(N, K, device=device, dtype=torch.float8_e4m3fn)
    x_scale = torch.empty(M, device=device, dtype=torch.float32)
    y = torch.empty(M, N, device=device, dtype=torch.bfloat16)
    _COMPILED_KERNELS[key] = cute.compile(
        w8a8_gemm_host,
        wrap_fp8_rows(x_fp8),
        wrap_fp8(w_fp8),
        wrap_rows(x_scale),
        wrap_rows(y),
        1.0,
        make_fake_stream(),
        *config,
    )
    return _COMPILED_KERNELS[key]


def precompile(N: int, K: int, device: torch.device) -> None:
    get_compiled(N, K, device)


def w8a8_linear(
    x: torch.Tensor,
    weight_fp8: torch.Tensor,
    weight_scale: float,
) -> torch.Tensor:
    """bf16 X [M, K] -> per-token FP8 (sgl_kernel) -> W8A8 GEMM -> bf16 Y [M, N]."""
    from sgl_kernel import sgl_per_token_quant_fp8

    x = x.contiguous()
    M, K = x.shape
    N = weight_fp8.shape[0]
    x_fp8 = torch.empty(M, K, device=x.device, dtype=torch.float8_e4m3fn)
    x_scale = torch.empty(M, 1, device=x.device, dtype=torch.float32)
    sgl_per_token_quant_fp8(x, x_fp8, x_scale)
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
